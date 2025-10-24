"""Team-level ranking pipeline (isolated from player pipeline).

Builds team-vs-team edges from per-PA statcast data by aggregating final
game scores per game_pk, uses run ratio (runs_for / max(runs_against,1))
as the directed edge weight winner->loser, and solves a tethered Laplacian
system to produce team rankings per season.

Design choices:
- No validations; CSV-only output of rankings.
- Chi-term is zero (no covariates), no temporal leakage concerns.
- Tether is kept via lambda_reg; R_map is uniformly 0.0 so D_reg=lambda*I.
- No harmonic scaling (use_harmonic=False) as R_map has no semantics for teams.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Reuse the aware solver for unipartite edges
from pipeline import aware_rank_with_tether


MLB_TEAMS = {
    'ARI','ATL','BAL','BOS','CHC','CIN','CLE','COL','CWS','DET','HOU','KC','LAA','LAD','MIA','MIL','MIN','NYM','NYY','OAK','PHI','PIT','SD','SEA','SF','STL','TB','TEX','TOR','WSH'
}

# Common aliases observed in raw data
TEAM_ALIASES: Dict[str, str] = {
    'AZ': 'ARI',  # Arizona Diamondbacks
    'CHW': 'CWS', # Chicago White Sox
    'WSN': 'WSH', # Washington Nationals
    'ATH': 'OAK', # Oakland Athletics alias in some feeds
}

def _canon_team(code: str) -> str:
    c = str(code).strip().upper()
    return TEAM_ALIASES.get(c, c)


def _read_pa_csv(year: int, raw_data_dir: str) -> pd.DataFrame | None:
    path = os.path.join(raw_data_dir, f"at_bat_data_{year}.csv")
    if not os.path.isfile(path):
        return None
    # We only need a handful of columns; load selectively if possible.
    usecols = [
        'game_pk','game_date','game_type','type',
        'home_team','away_team','home_score','away_score'
    ]
    try:
        df = pd.read_csv(path, usecols=lambda c: c in set(usecols))
    except Exception:
        df = pd.read_csv(path)
    if df is None or df.empty:
        return None
    return df


def _games_from_pas(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-PA rows to one row per game with final scores and teams.

    Returns columns: game_pk, home_team, away_team, home_final, away_final
    """
    # Normalize columns
    if 'game_pk' not in df.columns:
        # Fallback path: derive per-game segments from score resets within contiguous
        # (home_team, away_team) blocks. We assume rows are in chronological order
        # for each game (common for scraped statcast dumps) and that home/away scores
        # are non-decreasing during a game.
        for c in ('home_team','away_team','home_score','away_score'):
            if c not in df.columns:
                return pd.DataFrame(columns=['game_pk','home_team','away_team','home_final','away_final'])
        # Normalize team codes and restrict to MLB
        df = df.copy()
        df['home_team'] = df['home_team'].astype(str).map(_canon_team)
        df['away_team'] = df['away_team'].astype(str).map(_canon_team)
        df = df[df['home_team'].isin(MLB_TEAMS) & df['away_team'].isin(MLB_TEAMS)]
        if df.empty:
            return pd.DataFrame(columns=['game_pk','home_team','away_team','home_final','away_final'])
        # Ensure numeric scores
        hs = pd.to_numeric(df['home_score'], errors='coerce').fillna(0).astype(int)
        as_ = pd.to_numeric(df['away_score'], errors='coerce').fillna(0).astype(int)
        inn = pd.to_numeric(df.get('inning', pd.Series(0, index=df.index)), errors='coerce').fillna(0).astype(int)
        # Iterate sequentially to find game boundaries
        rows = []
        last_key = None
        last_h = last_a = last_inn = None
        game_idx = 0
        # We'll borrow the original index order as-seen
        for i, r in enumerate(df.itertuples(index=False)):
            hteam = getattr(r, 'home_team')
            ateam = getattr(r, 'away_team')
            h = int(hs.iloc[i])
            a = int(as_.iloc[i])
            inning = int(inn.iloc[i])
            key = (hteam, ateam)
            # Start of first game or new teams block
            start_new = False
            if last_key is None or key != last_key:
                start_new = True
            else:
                # Boundary if scores reset or inning goes backward (new game)
                if (last_h is not None and (h < last_h)) or (last_a is not None and (a < last_a)):
                    start_new = True
                elif (last_inn is not None and inning < last_inn):
                    start_new = True
            if start_new and last_key is not None:
                # Commit previous game's final for previous key
                rows.append({
                    'game_pk': f"{last_key[0]}_{last_key[1]}_{game_idx}",
                    'home_team': last_key[0],
                    'away_team': last_key[1],
                    'home_final': last_h if last_h is not None else 0,
                    'away_final': last_a if last_a is not None else 0,
                })
                game_idx += 1
            if start_new:
                # Reset counters at start of a new game
                last_key = key
                last_h, last_a, last_inn = h, a, inning
            else:
                # Continue within same game: update last scores/inning
                last_h, last_a, last_inn = h, a, inning
        # Commit trailing game
        if last_key is not None:
            rows.append({
                'game_pk': f"{last_key[0]}_{last_key[1]}_{game_idx}",
                'home_team': last_key[0],
                'away_team': last_key[1],
                'home_final': last_h if last_h is not None else 0,
                'away_final': last_a if last_a is not None else 0,
            })
        g = pd.DataFrame(rows, columns=['game_pk','home_team','away_team','home_final','away_final'])
        # Remove ties if any
        g = g[g['home_final'] != g['away_final']].copy()
        return g.reset_index(drop=True)
    for c in ('home_team','away_team','home_score','away_score'):
        if c not in df.columns:
            # Missing required columns; return empty
            return pd.DataFrame(columns=['game_pk','home_team','away_team','home_final','away_final'])
    # Normalize team codes using aliases
    df['home_team'] = df['home_team'].astype(str).map(_canon_team)
    df['away_team'] = df['away_team'].astype(str).map(_canon_team)
    # Filter to MLB games when possible
    if 'game_type' in df.columns:
        df = df[df['game_type'].isin(['R','P'])].copy()
    elif 'type' in df.columns:
        df = df[df['type'].isin(['R','P'])].copy()
    # Fallback: filter by known MLB franchises
    if {'home_team','away_team'}.issubset(df.columns):
        df = df[df['home_team'].isin(MLB_TEAMS) & df['away_team'].isin(MLB_TEAMS)].copy()
    # Compute final scores per game_pk using max observed scores
    g = df.groupby('game_pk', as_index=False).agg({
        'home_team': 'first',
        'away_team': 'first',
        'home_score': 'max',
        'away_score': 'max',
    }).rename(columns={'home_score':'home_final','away_score':'away_final'})
    # Drop games missing scores
    g = g[pd.notna(g['home_final']) & pd.notna(g['away_final'])].copy()
    # Ensure numeric
    g['home_final'] = pd.to_numeric(g['home_final'], errors='coerce').fillna(-1).astype(int)
    g['away_final'] = pd.to_numeric(g['away_final'], errors='coerce').fillna(-1).astype(int)
    g = g[(g['home_final'] >= 0) & (g['away_final'] >= 0)].copy()
    # Remove ties if any
    g = g[g['home_final'] != g['away_final']].copy()
    return g.reset_index(drop=True)


def _team_edges_from_games(games: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Build directed winner->loser edges with weight = run ratio.

    Returns (edges_df, node_list). edges_df has columns: winner, loser, score
    """
    rows: List[Tuple[str, str, float]] = []
    teams: set[str] = set()
    for _, r in games.iterrows():
        ht = str(r['home_team'])
        at = str(r['away_team'])
        hs = int(r['home_final'])
        as_ = int(r['away_final'])
        if hs > as_:
            w, l = ht, at
            rf, ra = hs, as_
        else:
            w, l = at, ht
            rf, ra = as_, hs
        denom = max(int(ra), 1)
        score = float(rf) / float(denom)
        rows.append((w, l, score))
        teams.add(w); teams.add(l)
    edf = pd.DataFrame(rows, columns=['winner','loser','score'])
    nodes = sorted(list(teams))
    return edf, nodes


def _regular_season_win_pct(games: pd.DataFrame) -> Dict[str, float]:
    """
    Compute regular season win percentage per team from game rows.
    Expects columns: home_team, away_team, home_final, away_final, and optionally game_type.
    """
    if games is None or games.empty:
        return {}
    g = games.copy()
    if 'game_type' in g.columns:
        g = g[g['game_type'] == 'R'].copy()
    # Tally wins and losses
    W: Dict[str, int] = {}
    L: Dict[str, int] = {}
    for _, r in g.iterrows():
        ht = str(r['home_team']); at = str(r['away_team'])
        hs = int(r['home_final']); as_ = int(r['away_final'])
        if hs == as_:
            continue
        win, lose = (ht, at) if hs > as_ else (at, ht)
        W[win] = W.get(win, 0) + 1
        L[lose] = L.get(lose, 0) + 1
        # Ensure both keys exist
        W.setdefault(lose, W.get(lose, 0))
        L.setdefault(win, L.get(win, 0))
    pct: Dict[str, float] = {}
    for t in set(list(W.keys()) + list(L.keys())):
        w = int(W.get(t, 0)); l = int(L.get(t, 0))
        n = w + l
        pct[t] = (w / n) if n > 0 else 0.0
    return pct


def _adjacency_from_edges(edges: pd.DataFrame, nodes: List[str]) -> np.ndarray:
    idx = {name: i for i, name in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=float)
    if edges is None or edges.empty:
        return A
    for _, r in edges.iterrows():
        i = idx.get(str(r['winner']))
        j = idx.get(str(r['loser']))
        if i is None or j is None or i == j:
            continue
        try:
            w = float(r['score'])
        except Exception:
            w = 0.0
        if w > 0:
            A[i, j] += w
    return A


def compute_team_rankings(cfg: Dict) -> None:
    years = [int(y) for y in cfg.get('years', [])]
    raw_dir = cfg.get('paths', {}).get('raw_data_dir', 'At Bats/general_data')
    out_dir = cfg.get('paths', {}).get('output_dir', 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    teams_cfg = cfg.get('teams', {}) if isinstance(cfg.get('teams'), dict) else {}
    lambda_reg = float(teams_cfg.get('lambda', 1.0))
    progress = bool(cfg.get('logging', {}).get('progress', False))

    all_rows = []
    for y in years:
        if progress:
            print(f"[teams] building team rankings for {y} from {raw_dir}")
        pa = _read_pa_csv(int(y), raw_dir)
        if pa is None or pa.empty:
            if progress:
                print(f"[teams] skipped {y}: no per-PA CSV found")
            continue
        games = _games_from_pas(pa)
        if games is None or games.empty:
            if progress:
                print(f"[teams] skipped {y}: no MLB games with final scores")
            continue
        edges, nodes = _team_edges_from_games(games)
        if progress:
            print(f"[teams] {y}: games={len(games)}, edges={len(edges)}, teams={len(nodes)}")
        A = _adjacency_from_edges(edges, nodes)
        # Uniform R_map=0.0 to keep tether as lambda*I; no harmonic scaling.
        R_map: Dict[str, float] = {name: 0.0 for name in nodes}
        ranks_vec, pairs = aware_rank_with_tether(A, nodes, R_map, lambda_reg=lambda_reg, use_harmonic=False)
        # Compute regular-season win percentage
        winpct_map = _regular_season_win_pct(games)
        # Convert to DataFrame rows
        for name, score in pairs:
            wp = float(winpct_map.get(str(name), 0.0))
            all_rows.append([int(y), str(name), float(score), wp])

    if not all_rows:
        if progress:
            print("[teams] no rankings produced (no data across requested years)")
        return
    rdf = pd.DataFrame(all_rows, columns=['Year','Team','RankScore','WinPct'])
    # Add ordinal rank within year
    rdf['Rank'] = rdf.groupby('Year')['RankScore'].rank(ascending=False, method='dense').astype(int)
    rdf['WinPctRank'] = rdf.groupby('Year')['WinPct'].rank(ascending=False, method='dense').astype(int)
    rdf = rdf.sort_values(['Year','Rank']).reset_index(drop=True)
    out_path = os.path.join(out_dir, 'team_rankings.csv')
    try:
        rdf.to_csv(out_path, index=False)
        if progress:
            print(f"[teams] wrote {len(rdf)} rows -> {out_path}")
    except Exception as e:
        # Fallback: write to a default outputs folder
        os.makedirs('outputs', exist_ok=True)
        fallback = os.path.join('outputs', 'team_rankings.csv')
        rdf.to_csv(fallback, index=False)
        if progress:
            print(f"[teams] wrote {len(rdf)} rows -> {fallback} (fallback due to {e})")
