"""Unified pipeline functions for MLB-Network-Analysis driven by JSON config.

This module avoids altering original research scripts while exposing a stable
API for programmatic execution.
"""
from __future__ import annotations
import os
import json
import time
import hashlib
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from datetime import date

# External deps (assumed installed in original project)
import baseball_scraper as pyb  # type: ignore
import networkx as nx
# SpringRank import (handle both package name casings) - import module, not class
try:
    import SpringRank as sr  # type: ignore
except ImportError:  # pragma: no cover - fallback for lowercase package name
    import springrank as sr  # type: ignore
from typing import Optional

# Debug: identify which module path is being imported at runtime
try:
    print(f"[debug] importing pipeline from: {__file__}")
    try:
        os.makedirs('outputs', exist_ok=True)
        with open(os.path.join('outputs','debug_import_pipeline.txt'), 'a', encoding='utf-8') as _fh:
            _fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} :: {__file__}\n")
    except Exception:
        pass
except Exception:
    pass

# Reuse logic from Rankings.py minimally (avoid import side-effects)
# (Potential improvement: refactor Rankings.py into functions and import.)

ALLOWED_PITCH_TYPES = ["CH","CU","FC","FF","FS","FT","SI","SL"]

# ------------------------- Aware scoring (new score_type) ------------------------- #
def _detect_bool(v: Any) -> bool:
    try:
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, float, np.integer, np.floating)):
            return bool(int(v))
        s = str(v).strip().lower()
        return s in ("1","true","t","yes","y")
    except Exception:
        return False

def _aware_pa_score(row: Dict[str, Any], weights: Dict[str, float]) -> Optional[float]:
    # Ignore intentional walks
    for key in ("ibb","is_intentional_walk","intent_walk","intentional_walk"):
        if key in row and _detect_bool(row[key]):
            return None
    ev = (str(row.get('event', row.get('events',''))).strip().lower())
    # Strikeouts
    if ev in ("strikeout","strikeout_double_play","strikeout_triple_play","strikeout - dp","strikeout - tp","k"):
        return float(weights.get('wK', -0.28))
    # Walk / HBP
    if ev in ("walk","bb","base on balls"):
        return float(weights.get('wBB', 0.69))
    if ev in ("hit_by_pitch","hbp"):
        return float(weights.get('wHBP', 0.72))
    # Batted balls -> xwOBA if available
    for c in ("xwOBA","estimated_woba_using_speedangle","woba_expectation"):
        if c in row and row[c] is not None and not pd.isna(row[c]):
            try:
                return float(row[c])
            except Exception:
                pass
    # Fallback: wOBA value
    for c in ("woba_value","wOBA","woba"):
        if c in row and row[c] is not None and not pd.isna(row[c]):
            try:
                return float(row[c])
            except Exception:
                pass
    return None

def _aware_covariates(df: pd.DataFrame, *, basic: bool = False) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    # Batter handedness
    if 'stand' in df.columns:
        out = out.join(pd.get_dummies(df['stand'].astype(str).str.upper(), prefix='stand', drop_first=True))
    # Pitcher throws
    if 'p_throws' in df.columns:
        out = out.join(pd.get_dummies(df['p_throws'].astype(str).str.upper(), prefix='pthr', drop_first=True))
    if basic:
        return out.fillna(0.0)
    # Home/park
    if 'home_team' in df.columns:
        out = out.join(pd.get_dummies(df['home_team'].astype(str), prefix='home_team', drop_first=True))
    elif 'home' in df.columns:
        out['home'] = df['home'].apply(_detect_bool).astype(int)
    # Pitch count
    if 'pitch_number' in df.columns:
        out['pitch_count'] = pd.to_numeric(df['pitch_number'], errors='coerce').fillna(0.0)
    elif 'n_pitches' in df.columns:
        out['pitch_count'] = pd.to_numeric(df['n_pitches'], errors='coerce').fillna(0.0)
    return out.fillna(0.0)

def _aware_pa_score_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized per-PA score: prefer xwOBA/wOBA columns; fallback to events K/BB/HBP."""
    y = pd.Series(np.nan, index=df.index, dtype='float64')
    # Prefer outcome expectation columns first
    for c in ("xwOBA","estimated_woba_using_speedangle","woba_expectation","woba_value","wOBA","woba"):
        if c in df.columns:
            v = pd.to_numeric(df[c], errors='coerce')
            y = y.where(~y.isna(), v)
    # Fallback to events when still NaN
    if y.isna().any():
        ev = df.get('event', df.get('events', pd.Series('', index=df.index))).astype(str).str.strip().str.lower()
        is_k = ev.isin(["strikeout","strikeout_double_play","strikeout_triple_play","strikeout - dp","strikeout - tp","k"])
        is_bb = ev.isin(["walk","bb","base on balls"]) | df.get('bb', pd.Series(False, index=df.index)).astype(bool)
        is_hbp = ev.isin(["hit_by_pitch","hbp"]) | df.get('hbp', pd.Series(False, index=df.index)).astype(bool)
        y = y.where(~(y.isna() & is_k), -0.28)
        y = y.where(~(y.isna() & is_bb), 0.69)
        y = y.where(~(y.isna() & is_hbp), 0.72)
    return y

def _aware_fit_two_way_ridge(df: pd.DataFrame, y: np.ndarray, alpha_ridge: float = 1.0, *, use_covariates: bool = True, basic_covariates: bool = False) -> Tuple[float, Dict[str,float], Dict[str,float], np.ndarray, Dict[str,float], Dict[str,float]]:
    from sklearn.linear_model import Ridge  # type: ignore
    bat = df['batter'].astype(str)
    pit = df['pitcher'].astype(str)
    B = pd.get_dummies(bat, prefix='b', drop_first=True)
    P = pd.get_dummies(pit, prefix='p', drop_first=True)
    if B.shape[1] == 0:
        B = pd.DataFrame(index=df.index)
    if P.shape[1] == 0:
        P = pd.DataFrame(index=df.index)
    Xcov = _aware_covariates(df, basic=basic_covariates) if use_covariates else pd.DataFrame(index=df.index)
    X = pd.concat([B, -1.0*P, Xcov], axis=1).astype(float)
    X_mat = X.values.astype(float)
    n, p = X_mat.shape
    model = Ridge(alpha=alpha_ridge, fit_intercept=True)
    model.fit(X_mat, y)
    mu = float(model.intercept_)
    coef = pd.Series(model.coef_, index=X.columns)
    alpha_map: Dict[str,float] = {}
    sigma_map: Dict[str,float] = {}
    for c in B.columns:
        alpha_map[c.replace('b_','',1)] = float(coef.get(c, 0.0))
    omitted_b = list(set(bat.unique()) - set([c.replace('b_','',1) for c in B.columns]))
    for bname in omitted_b:
        alpha_map[str(bname)] = 0.0
    for c in P.columns:
        sigma_map[c.replace('p_','',1)] = float(-coef.get(c, 0.0))
    omitted_p = list(set(pit.unique()) - set([c.replace('p_','',1) for c in P.columns]))
    for pname in omitted_p:
        sigma_map[str(pname)] = 0.0
    # Enforce sum-to-zero constraints by centering and adjusting intercept
    mean_alpha = float(np.mean(list(alpha_map.values()))) if alpha_map else 0.0
    mean_sigma = float(np.mean(list(sigma_map.values()))) if sigma_map else 0.0
    if alpha_map:
        for k in list(alpha_map.keys()):
            alpha_map[k] -= mean_alpha
    if sigma_map:
        for k in list(sigma_map.keys()):
            sigma_map[k] -= mean_sigma
    mu = mu + mean_alpha + mean_sigma
    # covariate linear predictor per row
    beta_cov = coef[Xcov.columns].values if len(Xcov.columns) else np.zeros((0,), dtype=float)
    chi_beta = (Xcov.values @ beta_cov) if len(beta_cov) else np.zeros((len(df),), dtype=float)
    # Approximate coefficient SEs for shrinkage
    se_alpha: Dict[str,float] = {}
    se_sigma: Dict[str,float] = {}
    try:
        import numpy.linalg as npl
        XTX = X_mat.T @ X_mat
        A = XTX + alpha_ridge * np.eye(p, dtype=float)
        A_inv = npl.inv(A)
        M = A_inv @ XTX @ A_inv
        resid = y - (X_mat @ model.coef_.reshape(-1) + mu)
        dof = max(n - p, 1)
        s2 = float((resid @ resid) / dof)
        var_diag = np.clip(np.diag(M) * s2, a_min=0.0, a_max=None)
        col_to_se = {col: float(np.sqrt(var_diag[i])) for i, col in enumerate(X.columns)}
        if B.shape[1] > 0:
            b_cols = list(B.columns)
            b_ses = [col_to_se.get(c, np.nan) for c in b_cols]
            b_med = float(np.nanmedian(b_ses)) if np.isfinite(np.nanmedian(b_ses)) else 1.0
            for c in b_cols:
                se_alpha[c.replace('b_','',1)] = float(col_to_se.get(c, b_med))
            for bname in omitted_b:
                se_alpha[str(bname)] = b_med
        if P.shape[1] > 0:
            p_cols = list(P.columns)
            p_ses = [col_to_se.get(c, np.nan) for c in p_cols]
            p_med = float(np.nanmedian(p_ses)) if np.isfinite(np.nanmedian(p_ses)) else 1.0
            for c in p_cols:
                se_sigma[c.replace('p_','',1)] = float(col_to_se.get(c, p_med))
            for pname in omitted_p:
                se_sigma[str(pname)] = p_med
    except Exception:
        # Fallback uniform SEs
        for k in alpha_map.keys():
            se_alpha[k] = 1.0
        for k in sigma_map.keys():
            se_sigma[k] = 1.0
    return mu, alpha_map, sigma_map, chi_beta, se_alpha, se_sigma
def _aware_fit_two_way_ridge_fast(df: pd.DataFrame, y: np.ndarray, alpha_ridge: float = 1.0, *, use_covariates: bool = True, basic_covariates: bool = False):
    """Sparse/fast two-way ridge fit for large datasets (no SEs).

    Model: y ≈ mu + alpha_b + (-1)*sigma_p.
    Returns: (mu, alpha_map, sigma_map, chi_beta, se_alpha, se_sigma)
    where chi_beta is zeros (no extra covariates) and SE dicts are empty.
    """
    try:
        from scipy.sparse import csr_matrix, hstack
        from sklearn.linear_model import Ridge
    except Exception:
        # Fallback to original if sparse stack not available
        return _aware_fit_two_way_ridge(df, y, alpha_ridge=alpha_ridge, use_covariates=use_covariates, basic_covariates=basic_covariates)
    # Indices
    bat = df['batter'].astype(str).values
    pit = df['pitcher'].astype(str).values
    bats = sorted(pd.unique(bat).tolist())
    pits = sorted(pd.unique(pit).tolist())
    nb = len(bats); npit = len(pits); n = len(df)
    b_index = {name: i for i, name in enumerate(bats)}
    p_index = {name: i for i, name in enumerate(pits)}
    # Build one-hot sparse matrices
    b_rows = np.arange(n, dtype=int)
    b_cols = np.array([b_index.get(b, -1) for b in bat], dtype=int)
    mask_b = b_cols >= 0
    B = csr_matrix((np.ones(mask_b.sum(), dtype=float), (b_rows[mask_b], b_cols[mask_b])), shape=(n, nb))
    p_rows = np.arange(n, dtype=int)
    p_cols = np.array([p_index.get(p, -1) for p in pit], dtype=int)
    mask_p = p_cols >= 0
    P = csr_matrix((np.ones(mask_p.sum(), dtype=float), (p_rows[mask_p], p_cols[mask_p])), shape=(n, npit))
    # Optional covariates
    Xcov_df = _aware_covariates(df, basic=basic_covariates) if use_covariates else pd.DataFrame(index=df.index)
    Xcov = csr_matrix(Xcov_df.values.astype(float)) if Xcov_df.shape[1] > 0 else None
    # Concatenate [B | -P | Xcov]
    X = hstack([m for m in ([B, -P] + ([Xcov] if Xcov is not None else []))], format='csr')
    model = Ridge(alpha=float(alpha_ridge), fit_intercept=True)
    model.fit(X, y)
    coefs = model.coef_.reshape(-1)
    mu = float(model.intercept_)
    split1 = nb
    split2 = nb + npit
    coef_b = coefs[:split1]
    coef_pneg = coefs[split1:split2]
    coef_cov = coefs[split2:]
    alpha_map = {bats[i]: float(coef_b[i]) for i in range(nb)}
    sigma_map = {pits[j]: float(-coef_pneg[j]) for j in range(npit)}
    chi_beta = (Xcov_df.values @ coef_cov) if (use_covariates and Xcov_df.shape[1] > 0) else np.zeros((n,), dtype=float)
    se_alpha: Dict[str, float] = {}
    se_sigma: Dict[str, float] = {}
    return mu, alpha_map, sigma_map, chi_beta, se_alpha, se_sigma

def _aware_edges_from_scores(df: pd.DataFrame, score_col: str, row_b_col: str, row_p_col: str, winners_role: str, year: int, chunk_p: int = 64) -> pd.DataFrame:
    # Aggregate per (batter,pitcher): mean, variance, and count
    grp = df.groupby([row_b_col, row_p_col])[score_col]
    mean_df = grp.mean().rename('mean').reset_index()
    var_df = grp.var(ddof=1).rename('var').reset_index()
    cnt_df = grp.size().rename('n').reset_index()
    stats = mean_df.merge(var_df, on=[row_b_col, row_p_col], how='left').merge(cnt_df, on=[row_b_col, row_p_col], how='left')
    # Establish a sensible variance floor to avoid exploding weights with n=1
    try:
        global_var = float(np.nanvar(df[score_col].astype(float).values))
    except Exception:
        global_var = 1.0
    # Median positive variance across observed pairs; fallback to global var or 1.0
    med_pair_var = float(np.nanmedian(stats['var'][stats['var'] > 0])) if np.any(stats['var'] > 0) else (global_var if np.isfinite(global_var) and global_var>0 else 1.0)
    var_floor = max(med_pair_var, 1e-3)
    # Variance of the mean ~ var / n; for n<2, use floor
    stats['n'] = stats['n'].fillna(0).astype(float)
    stats['var'] = stats['var'].astype(float)
    stats['var_mean'] = np.where(stats['n'] >= 2, stats['var'] / stats['n'].clip(lower=1.0), var_floor)
    stats['var_mean'] = stats['var_mean'].fillna(var_floor).clip(lower=var_floor)
    # Pivot into player x opponent matrices
    if winners_role == 'batter':
        idx_col, col_col = row_b_col, row_p_col
    else:
        idx_col, col_col = row_p_col, row_b_col
    players = sorted(stats[idx_col].unique().tolist())
    opps = sorted(stats[col_col].unique().tolist())
    if not players or not opps:
        return pd.DataFrame(columns=['winner','loser','score'])
    m = pd.pivot_table(stats, index=idx_col, columns=col_col, values='mean').reindex(index=players, columns=opps).astype(float).fillna(0.0)
    v = pd.pivot_table(stats, index=idx_col, columns=col_col, values='var_mean').reindex(index=players, columns=opps).astype(float).fillna(var_floor)
    B = len(players)
    D = np.zeros((B, B), dtype=np.float32)
    # chunk over opponent columns
    for start in range(0, len(opps), chunk_p):
        end = min(start + chunk_p, len(opps))
        m_blk = m.iloc[:, start:end].values.astype(np.float32, copy=False)
        v_blk = v.iloc[:, start:end].values.astype(np.float32, copy=False)
        # Vectorized accumulation across all opponents in the block:
        # For each opponent p in block, contribution C_p = (a[:,p] - a[:,p]^T) / (s[:,p] + s[:,p]^T + var_floor)
        # Compute as a 3D tensor then sum along p-axis, but do it block-wise to control memory.
        Ai = m_blk[:, None, :]  # (B,1,Pblk)
        Aj = m_blk[None, :, :]  # (1,B,Pblk)
        Si = v_blk[:, None, :]  # (B,1,Pblk)
        Sj = v_blk[None, :, :]  # (1,B,Pblk)
        denom = Si + Sj
        if np.isscalar(var_floor):
            denom = denom + np.float32(var_floor)
        else:
            denom = denom + np.float32(var_floor)
        contrib = (Ai - Aj) / denom  # (B,B,Pblk)
        D += np.sum(contrib, axis=2).astype(np.float32)
    np.fill_diagonal(D, 0.0)
    wi, li = np.where(D > 0.0)
    scores = D[wi, li].astype(float)
    winners = [players[i] for i in wi]
    losers = [players[j] for j in li]
    return pd.DataFrame({'winner': winners, 'loser': losers, 'score': scores})

def ensure_aware_edges(year: int, raw_data_dir: str, alpha_ridge: float = 1.0, progress: bool = False, force: bool = False, *, use_shrink: bool = True, shrink_mode: str = 'se_based', shrink_k: int = 150, use_covariates: bool = True, shrink_k_batter: Optional[int] = None, shrink_k_pitcher: Optional[int] = None, basic_covariates: bool = False):
    # Reuse existing aware edges when not forcing regeneration
    b_out_dir = os.path.join('At Bats','batter_data','aware_scores')
    p_out_dir = os.path.join('At Bats','pitcher_data','aware_scores')
    os.makedirs(b_out_dir, exist_ok=True)
    os.makedirs(p_out_dir, exist_ok=True)
    b_edge_out = os.path.join(b_out_dir, f"{year}_batter_edges.csv")
    p_edge_out = os.path.join(p_out_dir, f"{year}_pitcher_edges.csv")
    if (not force) and os.path.isfile(b_edge_out) and os.path.isfile(p_edge_out):
        if progress:
            print(f"[aware] existing edges found for {year}; force={force} -> reuse and skip regeneration")
        return
    in_path = os.path.join(raw_data_dir, f"at_bat_data_{year}.csv")
    if not os.path.isfile(in_path):
        if progress: print(f"[aware] missing input: {in_path}")
        return
    df = pd.read_csv(in_path)
    # Defensive MLB-only filter if available
    try:
        for gt_col in ("game_type","type"):
            if gt_col in df.columns:
                before = len(df)
                df = df[df[gt_col].isin(["R","P"])].copy()
                if progress:
                    print(f"[aware] filtered MLB games by {gt_col} in {{'R','P'}}: {before} -> {len(df)} rows")
                break
        else:
            # Fallback: restrict to rows where both teams are MLB franchises
            MLB_TEAMS = {
                'ARI','ATL','BAL','BOS','CHC','CIN','CLE','COL','CWS','DET','HOU','KC','LAA','LAD','MIA','MIL','MIN','NYM','NYY','OAK','PHI','PIT','SD','SEA','SF','STL','TB','TEX','TOR','WSH'
            }
            if {'home_team','away_team'}.issubset(df.columns):
                before2 = len(df)
                df = df[df['home_team'].isin(MLB_TEAMS) & df['away_team'].isin(MLB_TEAMS)].copy()
                if progress:
                    print(f"[aware] fallback MLB team filter applied: {before2} -> {len(df)} rows")
    except Exception:
        pass
    # Normalize batter/pitcher columns
    if 'batter_name' in df.columns:
        df['batter'] = df['batter_name']
    if 'pitcher_name' in df.columns:
        df['pitcher'] = df['pitcher_name']
    if 'batter' not in df.columns or 'pitcher' not in df.columns:
        if progress: print("[aware] batter/pitcher columns missing")
        return
    # Compute per-PA aware YB/YP values via helper
    # Compute YB/YP (aware). If use_shrink is False, compute without shrinkage.
    if use_shrink:
        df, a_shrunk, s_shrunk, R_b_map, R_p_map = _aware_compute_y(
            df,
            alpha_ridge=alpha_ridge,
            progress=progress,
            shrink_mode=shrink_mode,
            shrink_k=shrink_k,
            use_covariates=use_covariates,
            shrink_k_batter=shrink_k_batter,
            shrink_k_pitcher=shrink_k_pitcher,
            basic_covariates=basic_covariates,
        )
    else:
        df, a_shrunk, s_shrunk, R_b_map, R_p_map = _aware_compute_y_no_shrink(
            df,
            alpha_ridge=alpha_ridge,
            progress=progress,
            use_covariates=use_covariates,
            basic_covariates=basic_covariates,
        )
    if df is None or df.empty or ('YB' not in df.columns) or ('YP' not in df.columns):
        if progress: print(f"[aware] no scored PAs for {year}")
        return
    # Output aware unipartite edges
    if force or (not os.path.isfile(b_edge_out)) or (not os.path.isfile(p_edge_out)):
        bedges = _aware_edges_from_scores(df[['batter','pitcher','YB']].rename(columns={'YB':'score'}), 'score', 'batter', 'pitcher', 'batter', year)
        try:
            bedges = _resolve_names_in_edges_df(bedges)
        except Exception:
            pass
        # Optional pruning for performance / robustness
        try:
            from typing import cast
            # Read pruning settings from a global-like cfg is not trivial here; instead, infer from env vars if set
            topk = os.environ.get('AWARE_TOPK_PER_NODE')
            qmin = os.environ.get('AWARE_MIN_QUANTILE')
            if topk is not None:
                try:
                    k = int(topk)
                    if k > 0 and {'winner','score'}.issubset(bedges.columns):
                        bedges = bedges.sort_values(['winner','score'], ascending=[True,False]).groupby('winner').head(k).reset_index(drop=True)
                except Exception:
                    pass
            if qmin is not None:
                try:
                    q = float(qmin)
                    if 0.0 <= q < 1.0 and 'score' in bedges.columns:
                        thr = float(bedges['score'].quantile(q))
                        bedges = bedges[bedges['score'] >= thr].copy()
                except Exception:
                    pass
        except Exception:
            pass
        bedges.to_csv(b_edge_out, index=False)
        pedges = _aware_edges_from_scores(df[['batter','pitcher','YP']].rename(columns={'YP':'score'}), 'score', 'batter', 'pitcher', 'pitcher', year)
        try:
            pedges = _resolve_names_in_edges_df(pedges)
        except Exception:
            pass
        try:
            topk = os.environ.get('AWARE_TOPK_PER_NODE')
            qmin = os.environ.get('AWARE_MIN_QUANTILE')
            if topk is not None:
                try:
                    k = int(topk)
                    if k > 0 and {'winner','score'}.issubset(pedges.columns):
                        pedges = pedges.sort_values(['winner','score'], ascending=[True,False]).groupby('winner').head(k).reset_index(drop=True)
                except Exception:
                    pass
            if qmin is not None:
                try:
                    q = float(qmin)
                    if 0.0 <= q < 1.0 and 'score' in pedges.columns:
                        thr = float(pedges['score'].quantile(q))
                        pedges = pedges[pedges['score'] >= thr].copy()
                except Exception:
                    pass
        except Exception:
            pass
        pedges.to_csv(p_edge_out, index=False)
        # Persist R shrink factors for later aware ranking
        try:
            rb = pd.DataFrame({'Player': list(a_shrunk.keys())})
            rb['R'] = 1.0 if (not use_shrink) else rb['Player'].map(R_b_map).fillna(0.0)
            rp = pd.DataFrame({'Player': list(s_shrunk.keys())})
            rp['R'] = 1.0 if (not use_shrink) else rp['Player'].map(R_p_map).fillna(0.0)
            for dfR, odir in ((rb, b_out_dir), (rp, p_out_dir)):
                try:
                    tmp = dfR.rename(columns={'Player':'winner'}).copy(); tmp['loser']=tmp['winner']; tmp['score']=1.0
                    mapped = _resolve_names_in_edges_df(tmp)
                    out = mapped[['winner']].rename(columns={'winner':'Player'})
                    out['R'] = dfR['R'].values
                except Exception:
                    out = dfR
                out.to_csv(os.path.join(odir, f"{year}_R.csv"), index=False)
        except Exception:
            pass
    if progress: print(f"[aware] edges written for {year} (shrink={'on' if use_shrink else 'off'}, mode={shrink_mode}, covariates={'on' if use_covariates else 'off'})")

# Base handcrafted scoring (from add_edgeinfo.py) and base pitcher scoring
BASE_BATTER_SCORING = {'hit_by_pitch':1,'walk':2,'single':3,'double':6,'triple':9,'home_run':12}
BASE_PITCHER_SCORING = {'fielders_choice':1,'fielders_choice_out':1,'other_out':1,'field_out':1,'force_out':2,'grounded_into_double_play':2,'strikeout':6}

# ------------------------- Scraping ---------------------------------------- #

def _aware_compute_y(df_in: pd.DataFrame, alpha_ridge: float = 1.0, progress: bool = False, *, shrink_mode: str = 'se_based', shrink_k: int = 150, use_covariates: bool = True, shrink_k_batter: Optional[int] = None, shrink_k_pitcher: Optional[int] = None, basic_covariates: bool = False) -> Tuple[pd.DataFrame, Dict[str,float], Dict[str,float], Dict[str,float], Dict[str,float]]:
    try:
        df = df_in.copy()
        # Normalize batter/pitcher columns
        if 'batter_name' in df.columns:
            df['batter'] = df['batter_name']
        if 'pitcher_name' in df.columns:
            df['pitcher'] = df['pitcher_name']
        if 'batter' not in df.columns or 'pitcher' not in df.columns:
            return pd.DataFrame(), {}, {}, {}, {}
        # Compute per-PA score
        # Vectorized per-PA score
        df['y_bp'] = _aware_pa_score_series(df)
        df = df[~df['y_bp'].isna()].copy()
        if df.empty:
            return df, {}, {}, {}, {}
        # Use fast sparse fit when using n/(n+k) shrink (SEs not required)
        if shrink_mode == 'n_over_n_plus_k':
            mu, alpha_map, sigma_map, chi_beta, se_alpha, se_sigma = _aware_fit_two_way_ridge_fast(df, df['y_bp'].astype(float).values, alpha_ridge=alpha_ridge, use_covariates=use_covariates, basic_covariates=basic_covariates)
        else:
            mu, alpha_map, sigma_map, chi_beta, se_alpha, se_sigma = _aware_fit_two_way_ridge(df, df['y_bp'].astype(float).values, alpha_ridge=alpha_ridge, use_covariates=use_covariates, basic_covariates=basic_covariates)
        # Across-player variances
        rho_b2 = float(np.var(list(alpha_map.values()))) if alpha_map else 0.0
        rho_p2 = float(np.var(list(sigma_map.values()))) if sigma_map else 0.0
        eps = 1e-12
        # Shrinkage factors R per player
        if shrink_mode == 'n_over_n_plus_k':
            # Count MLB PAs per player (batter and pitcher separately)
            n_b = df['batter'].astype(str).value_counts().to_dict()
            n_p = df['pitcher'].astype(str).value_counts().to_dict()
            kb = shrink_k_batter if (shrink_k_batter is not None) else shrink_k
            kp = shrink_k_pitcher if (shrink_k_pitcher is not None) else shrink_k
            R_b_map = {k: float(n_b.get(k,0)) / float(n_b.get(k,0) + kb) for k in alpha_map.keys()}
            R_p_map = {k: float(n_p.get(k,0)) / float(n_p.get(k,0) + kp) for k in sigma_map.keys()}
        else:
            R_b_map = {k: (rho_b2 / (rho_b2 + (se_alpha.get(k,1.0)**2) + eps)) if (rho_b2 + (se_alpha.get(k,1.0)**2))>0 else 0.0 for k in alpha_map.keys()}
            R_p_map = {k: (rho_p2 / (rho_p2 + (se_sigma.get(k,1.0)**2) + eps)) if (rho_p2 + (se_sigma.get(k,1.0)**2))>0 else 0.0 for k in sigma_map.keys()}
        # Apply shrinkage and re-center sums to zero
        alpha_shrunk = {k: float(R_b_map.get(k,0.0))*float(v) for k,v in alpha_map.items()}
        sigma_shrunk = {k: float(R_p_map.get(k,0.0))*float(v) for k,v in sigma_map.items()}
        if alpha_shrunk:
            m_a = float(np.mean(list(alpha_shrunk.values())))
            for k in list(alpha_shrunk.keys()):
                alpha_shrunk[k] -= m_a
        if sigma_shrunk:
            m_s = float(np.mean(list(sigma_shrunk.values())))
            for k in list(sigma_shrunk.keys()):
                sigma_shrunk[k] -= m_s
        # Map into dataframe and compute YB/YP
        df['alpha_b'] = df['batter'].astype(str).map(alpha_shrunk).fillna(0.0)
        df['sigma_p'] = df['pitcher'].astype(str).map(sigma_shrunk).fillna(0.0)
        df['chi_beta'] = chi_beta if use_covariates else 0.0
        df['YB'] = df['y_bp'] - (mu + df['sigma_p'] + df['chi_beta'])
        df['YP'] = df['y_bp'] - (mu + df['alpha_b'] + df['chi_beta'])
        return df, alpha_shrunk, sigma_shrunk, R_b_map, R_p_map
    except Exception:
        return pd.DataFrame(), {}, {}, {}, {}

def _aware_compute_y_no_shrink(df_in: pd.DataFrame, alpha_ridge: float = 1.0, progress: bool = False, use_covariates: bool = True, basic_covariates: bool = False) -> Tuple[pd.DataFrame, Dict[str,float], Dict[str,float], Dict[str,float], Dict[str,float]]:
    try:
        df = df_in.copy()
        if 'batter_name' in df.columns:
            df['batter'] = df['batter_name']
        if 'pitcher_name' in df.columns:
            df['pitcher'] = df['pitcher_name']
        if 'batter' not in df.columns or 'pitcher' not in df.columns:
            return pd.DataFrame(), {}, {}, {}, {}
        # Vectorized per-PA score
        df['y_bp'] = _aware_pa_score_series(df)
        df = df[~df['y_bp'].isna()].copy()
        if df.empty:
            return df, {}, {}, {}, {}
        mu, alpha_map, sigma_map, chi_beta, _se_alpha, _se_sigma = _aware_fit_two_way_ridge(
            df,
            df['y_bp'].astype(float).values,
            alpha_ridge=alpha_ridge,
            use_covariates=use_covariates,
            basic_covariates=basic_covariates,
        )
        # No shrinkage: use raw alpha/sigma, but recenter to sum zero to respect identifiability
        alpha_raw = dict(alpha_map)
        sigma_raw = dict(sigma_map)
        if alpha_raw:
            m_a = float(np.mean(list(alpha_raw.values())))
            for k in list(alpha_raw.keys()):
                alpha_raw[k] -= m_a
        if sigma_raw:
            m_s = float(np.mean(list(sigma_raw.values())))
            for k in list(sigma_raw.keys()):
                sigma_raw[k] -= m_s
        df['alpha_b'] = df['batter'].astype(str).map(alpha_raw).fillna(0.0)
        df['sigma_p'] = df['pitcher'].astype(str).map(sigma_raw).fillna(0.0)
        df['chi_beta'] = chi_beta if use_covariates else 0.0
        df['YB'] = df['y_bp'] - (mu + df['sigma_p'] + df['chi_beta'])
        df['YP'] = df['y_bp'] - (mu + df['alpha_b'] + df['chi_beta'])
        # R maps are all ones when shrinkage is disabled
        ones_b = {k: 1.0 for k in alpha_raw.keys()}
        ones_p = {k: 1.0 for k in sigma_raw.keys()}
        return df, alpha_raw, sigma_raw, ones_b, ones_p
    except Exception:
        return pd.DataFrame(), {}, {}, {}, {}

def season_date_range(year: int) -> Tuple[str,str]:
    # Prefer known regular-season windows; otherwise fall back to a generic range
    ranges = {
        2019:("2019-03-20","2019-10-30"),
        2018:("2018-03-29","2018-10-28"),
        2017:("2017-04-02","2017-11-01"),
        2016:("2016-04-03","2016-11-02"),
        2015:("2015-04-05","2015-11-01"),
        2014:("2014-03-22","2014-10-29"),
        2013:("2013-03-31","2013-10-30"),
        2012:("2012-03-28","2012-10-28"),
        2011:("2011-03-31","2011-10-28"),
        2010:("2010-04-04","2010-11-01"),
        2009:("2009-04-05","2009-11-04"),
    }
    # For years outside the table, default to March–November to allow new scraping
    return ranges.get(year, (f"{year}-03-01", f"{year}-11-30"))

SCRAPE_COLUMNS = ['pitch_type','player_name','batter','events','description',
                  'home_team','away_team','inning','stand','p_throws',
                  'home_score','away_score']


def scrape_year(year: int, out_dir: str, force: bool=False, progress: bool=True) -> str:
    """Ensure a season CSV exists by delegating to the dedicated scraper.

    Uses a conservative March–November fallback for unknown years to allow
    scraping new seasons without code changes.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"at_bat_data_{year}.csv")
    if os.path.isfile(out_path) and not force:
        if progress: print(f"[scrape] {year} exists -> skip")
        return out_path
    # Delegate to At Bats/at_bat_scraper.py for consistent preprocessing
    try:
        import importlib.util as _importlib_util
        from pathlib import Path as _Path
        _scraper_path = os.path.join(os.path.dirname(__file__), 'At Bats', 'at_bat_scraper.py')
        _spec = _importlib_util.spec_from_file_location('mlb_at_bat_scraper', _scraper_path)
        if _spec is None or _spec.loader is None:
            raise ImportError('Could not load at_bat_scraper module spec')
        _mod = _importlib_util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
        _scrape_year = getattr(_mod, 'scrape_year')
    except Exception:
        # Fallback to local logic if import path with space causes issues
        # (kept for backward compatibility)
        start, end = season_date_range(year)
        if progress: print(f"[scrape] Fetching Statcast {year} {start}..{end}")
        data = pyb.statcast(start, end)
        data = data.dropna(subset=['events'])
        # Try to filter to MLB regular/postseason if game_type/type exists
        for gt_col in ('game_type','type'):
            if gt_col in data.columns:
                before = len(data)
                data = data[data[gt_col].isin(['R','P'])].copy()
                print(f"[scrape:fallback] Filtered MLB games by {gt_col}: {before} -> {len(data)} rows")
                break
        data.astype({'batter': 'int32'}).dtypes
        all_data = data[SCRAPE_COLUMNS]
        player_ids = [int(n) for n in all_data['batter'].to_numpy()]
        retrieved_names = pyb.playerid_reverse_lookup(player_ids, key_type='mlbam')
        batter_names = retrieved_names.loc[:,('key_mlbam','name_first','name_last')]
        batter_names.loc[:,('name_first')] = batter_names.loc[:,('name_first')].str.capitalize()
        batter_names.loc[:,('name_last')]  = batter_names.loc[:,('name_last')].str.capitalize()
        batter_names.loc[:,('batter_name')] = batter_names.loc[:,('name_first','name_last')].agg(' '.join, axis=1)
        merged = all_data.join(batter_names[['key_mlbam','batter_name']].set_index('key_mlbam'), on='batter')
        if 'batter' in merged.columns:
            del merged['batter']
        merged.to_csv(out_path, index=False)
        if progress: print(f"[scrape] Wrote {out_path}")
        return out_path
    # Use the enhanced scraper with default or custom date ranges
    start, end = season_date_range(year)
    try:
        res_path = _scrape_year(year, output_dir=_Path(out_dir), overwrite=force, date_range=(start, end))
    except TypeError as e:
        # If date_range parameter is not supported by function signature, retry without it
        if "date_range" in str(e):
            res_path = _scrape_year(year, output_dir=_Path(out_dir), overwrite=force)
        else:
            raise
    except Exception:
        # Propagate other exceptions (e.g., merge failures) to avoid duplicate attempts
        raise
    # The scraper writes only when non-empty; verify existence
    if os.path.isfile(out_path):
        if progress: print(f"[scrape] Wrote {out_path}")
        return out_path
    else:
        if progress: print(f"[scrape] No data written for {year} ({start}..{end})")
        # Return path that would have been written, but signal upstream via missing file
        return out_path


# ------------------------- Edge Generation -------------------------------- #

def _extract_mlbam_id_edges(val: str):
    try:
        import re
        s = str(val).strip()
        m = re.search(r'(?i)mlbam[\s_-]*(\d+)$', s)
        if m:
            return int(m.group(1))
        if s.isdigit():
            return int(s)
    except Exception:
        return None
    return None


def _resolve_names_in_edges_df(edf: pd.DataFrame) -> pd.DataFrame:
    """Replace MLBAM_<id> style names in winner/loser with real names where possible.

    Uses vectorized extraction of MLBAM ids to avoid slow per-row regex calls.
    Safe no-op if none found or lookups fail.
    """
    try:
        cols = ['winner','loser']
        # Vectorized extraction: support either 'MLBAM_<id>'-style or plain numeric strings
        pid_series_map: dict[str, pd.Series] = {}
        all_ids: list[int] = []
        for c in cols:
            s = edf[c].astype(str).str.strip()
            id_from_tag = s.str.extract(r'(?i)mlbam[\s_-]*(\d+)$', expand=False)
            id_from_digits = s.where(s.str.fullmatch(r'\d+'), other=pd.NA)
            pid_series = pd.to_numeric(id_from_tag.fillna(id_from_digits), errors='coerce').astype('Int64')
            pid_series_map[c] = pid_series
            if pid_series.notna().any():
                # Extend with unique ids from this column
                all_ids.extend([int(x) for x in pd.unique(pid_series.dropna())])
        ids = sorted(set(all_ids))
        if not ids:
            return edf
        name_map: dict[int,str] = {}
        # Load persistent cache first
        cache_path = os.path.join('At Bats','general_data','player_name_cache.csv')
        try:
            if os.path.isfile(cache_path):
                cdf = pd.read_csv(cache_path)
                if not cdf.empty and all(c in cdf.columns for c in ['mlbam','name']):
                    for k, v in cdf[['mlbam','name']].dropna().itertuples(index=False, name=None):
                        try:
                            name_map[int(k)] = str(v)
                        except Exception:
                            continue
        except Exception:
            pass
        # Try pybaseball
        try:
            import importlib as _il
            _pb = _il.import_module('pybaseball')
            df1 = _pb.playerid_reverse_lookup(ids, key_type='mlbam')
            if df1 is not None and not df1.empty and all(c in df1.columns for c in ['key_mlbam','name_first','name_last']):
                df1 = df1[['key_mlbam','name_first','name_last']].copy()
                df1['key_mlbam'] = pd.to_numeric(df1['key_mlbam'], errors='coerce').astype('Int64')
                df1['name_first'] = df1['name_first'].astype(str).str.capitalize()
                df1['name_last'] = df1['name_last'].astype(str).str.capitalize()
                df1['full'] = df1[['name_first','name_last']].agg(' '.join, axis=1)
                name_map.update({int(k): v for k,v in df1.set_index('key_mlbam')['full'].dropna().items()})
        except Exception:
            pass
        # Try baseball_scraper
        if len(name_map) < len(ids):
            try:
                import baseball_scraper as _bs
                df2 = _bs.playerid_reverse_lookup(ids, key_type='mlbam')
                if df2 is not None and not df2.empty and all(c in df2.columns for c in ['key_mlbam','name_first','name_last']):
                    df2 = df2[['key_mlbam','name_first','name_last']].copy()
                    df2['key_mlbam'] = pd.to_numeric(df2['key_mlbam'], errors='coerce').astype('Int64')
                    df2['name_first'] = df2['name_first'].astype(str).str.capitalize()
                    df2['name_last'] = df2['name_last'].astype(str).str.capitalize()
                    df2['full'] = df2[['name_first','name_last']].agg(' '.join, axis=1)
                    name_map.update({int(k): v for k,v in df2.set_index('key_mlbam')['full'].dropna().items()})
            except Exception:
                pass
        # Fallback to MLB Stats API
        if len(name_map) < len(ids):
            try:
                from urllib.request import urlopen as _urlopen
                import json as _json
                remaining = [i for i in ids if i not in name_map]
                for i0 in range(0, len(remaining), 50):
                    batch = remaining[i0:i0+50]
                    url = 'https://statsapi.mlb.com/api/v1/people?personIds=' + ','.join(str(x) for x in batch)
                    with _urlopen(url, timeout=10) as resp:
                        data = _json.loads(resp.read().decode('utf-8'))
                    for p in data.get('people', []):
                        pid = p.get('id'); full = p.get('fullName')
                        if isinstance(pid, int) and isinstance(full, str) and full:
                            name_map[pid] = full
            except Exception:
                pass
        if not name_map:
            pass
        # Merge resolved results; optionally persist new ones to cache
        def _map_val(v: str) -> str:
            pid = _extract_mlbam_id_edges(v)
            if pid is not None:
                return name_map.get(pid, v)
            return v
        out = edf.copy()
        for c in cols:
            # Use vectorized replace when id present
            s = out[c].astype(str)
            pids = pid_series_map[c]
            # Map pid -> name (or keep original if not found)
            mapped = pids.map(lambda v: name_map.get(int(v)) if pd.notna(v) and int(v) in name_map else None)
            out[c] = np.where(mapped.notna(), mapped.astype(str), s)
        # Persist any new resolutions to cache
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            existing = {}
            if os.path.isfile(cache_path):
                try:
                    cdf2 = pd.read_csv(cache_path)
                    if not cdf2.empty and all(c in cdf2.columns for c in ['mlbam','name']):
                        existing = {int(k): str(v) for k, v in cdf2[['mlbam','name']].dropna().itertuples(index=False, name=None)}
                except Exception:
                    existing = {}
            merged = existing.copy()
            merged.update(name_map)
            # Only write if new keys were added
            if len(merged) > len(existing):
                pd.DataFrame(sorted([(k, v) for k, v in merged.items()], key=lambda x: x[0]), columns=['mlbam','name']).to_csv(cache_path, index=False)
        except Exception:
            pass
        return out
    except Exception:
        return edf

def _score_event(batter_name: str, pitcher_name: str, event: str, b_dict, p_dict):
    if event in b_dict:
        s = b_dict[event]
        return [batter_name, pitcher_name, s, 'batter'] if s >= 0 else [pitcher_name, batter_name, abs(s), 'pitcher']
    if event in p_dict:
        s = -p_dict[event]
        return [pitcher_name, batter_name, abs(s), 'pitcher'] if s < 0 else [batter_name, pitcher_name, s, 'batter']
    return None

def _frequency_scaling(df: pd.DataFrame, base_dict: dict, column: str='events') -> dict:
    counts = df[column].value_counts()
    total = counts.sum()
    scaled = {}
    for k,v in base_dict.items():
        c = counts.get(k,0)
        scaled[k] = v * (c/total) if total>0 else 0
    return scaled

def ensure_edge_only(year: int, score_type: str, raw_data_dir: str, progress: bool, pitch_types=None, innings=None, stand_filter=None, pthrows_filter=None, force: bool=False):
    """Create edge-only files under general_data/<type>/... if missing.
    score_type in {handmade, frequency, pitch_type, inning}
    """
    base_dir = raw_data_dir  # expected 'At Bats/general_data'
    raw_file = os.path.join(base_dir, f"at_bat_data_{year}.csv")
    if not os.path.isfile(raw_file):
        if progress: print(f"[edges] raw file missing {raw_file}")
        return []
    if progress: print(f"[edges] {year}:{score_type} reading raw file -> {raw_file}")
    df = pd.read_csv(raw_file)
    # Defensive filter: if the raw scraper preserved a game type column, restrict to MLB regular/postseason only
    try:
        for gt_col in ("game_type", "type"):
            if gt_col in df.columns:
                before = len(df)
                df = df[df[gt_col].isin(["R","P"])].copy()
                if progress:
                    print(f"[edges] filtered MLB games by {gt_col} in {{'R','P'}}: {before} -> {len(df)} rows")
                break
    except Exception:
        pass
    # Apply handedness filters if provided
    orig_rows = len(df)
    if stand_filter:
        df = df[df['stand'].isin(stand_filter)]
    if pthrows_filter:
        df = df[df['p_throws'].isin(pthrows_filter)]
    if progress:
        flt = []
        if stand_filter: flt.append(f"stand in {stand_filter}")
        if pthrows_filter: flt.append(f"p_throws in {pthrows_filter}")
        fdesc = ("; ".join(flt)) if flt else "none"
        print(f"[edges] {year}:{score_type} rows: {orig_rows} -> {len(df)} after filters ({fdesc})")
    # Harmonize schema differences: scraper may emit 'pitcher_name' (preferred) or legacy 'player_name'
    if 'pitcher_name' in df.columns:
        _pcol = 'pitcher_name'
    elif 'player_name' in df.columns:
        _pcol = 'player_name'
    elif 'pitcher' in df.columns:
        _pcol = 'pitcher'
    else:
        missing_cols = "pitcher_name/player_name/pitcher"
        raise KeyError(f"Required pitcher column missing from {raw_file}; expected one of: {missing_cols}. Columns present: {list(df.columns)}")
    if progress:
        print(f"[edges] {year}:{score_type} using pitcher column: '{_pcol}'")

    created = []
    if score_type == 'handmade':
        out_dir = os.path.join(base_dir, 'handmade')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{year}_edges_only.csv")
        # If existing file has MLBAM codes, regenerate
        if (not force) and os.path.isfile(out_path):
            try:
                tmp = pd.read_csv(out_path, usecols=['winner','loser']).astype(str)
                if tmp['winner'].str.contains(r'^MLBAM_\d+$', regex=True).any() or tmp['loser'].str.contains(r'^MLBAM_\d+$', regex=True).any():
                    if progress: print(f"[edges] {year}:handmade contains MLBAM codes -> regenerating")
                    force = True
            except Exception:
                pass
        if force or (not os.path.isfile(out_path)):
            if progress: print(f"[edges] {year}:handmade generating -> {out_path}")
            b_dict = BASE_BATTER_SCORING
            p_dict = BASE_PITCHER_SCORING
            rows = []
            for batter_name, pitcher_name, event in df[['batter_name', _pcol, 'events']].itertuples(index=False):
                r = _score_event(batter_name, pitcher_name, event, b_dict, p_dict)
                if r: rows.append(r)
            if rows:
                edf = pd.DataFrame(rows, columns=['winner','loser','score','who_won'])
                edf = edf.groupby(['winner','loser','who_won']).sum()
                # Resolve any MLBAM codes before writing
                edf = _resolve_names_in_edges_df(edf.reset_index()).set_index(['winner','loser','who_won'])
                edf.to_csv(out_path)
                created.append(out_path)
                if progress: print(f"[edges] wrote {out_path}")
        else:
            if progress: print(f"[edges] {year}:handmade exists -> skip ({out_path})")
        return created
    if score_type == 'frequency':
        out_dir = os.path.join(base_dir, 'frequency')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{year}_edges_only.csv")
        # If existing file has MLBAM codes, regenerate
        if (not force) and os.path.isfile(out_path):
            try:
                tmp = pd.read_csv(out_path, usecols=['winner','loser']).astype(str)
                if tmp['winner'].str.contains(r'^MLBAM_\d+$', regex=True).any() or tmp['loser'].str.contains(r'^MLBAM_\d+$', regex=True).any():
                    if progress: print(f"[edges] {year}:frequency contains MLBAM codes -> regenerating")
                    force = True
            except Exception:
                pass
        if force or (not os.path.isfile(out_path)):
            if progress: print(f"[edges] {year}:frequency generating -> {out_path}")
            b_freq = _frequency_scaling(df, BASE_BATTER_SCORING)
            p_freq = _frequency_scaling(df, BASE_PITCHER_SCORING)
            rows = []
            for batter_name, pitcher_name, event in df[['batter_name', _pcol, 'events']].itertuples(index=False):
                r = _score_event(batter_name, pitcher_name, event, b_freq, p_freq)
                if r: rows.append(r)
            if rows:
                edf = pd.DataFrame(rows, columns=['winner','loser','score','who_won'])
                edf = edf.groupby(['winner','loser','who_won']).sum()
                # Resolve any MLBAM codes before writing
                edf = _resolve_names_in_edges_df(edf.reset_index()).set_index(['winner','loser','who_won'])
                edf.to_csv(out_path)
                created.append(out_path)
                if progress: print(f"[edges] wrote {out_path}")
        else:
            if progress: print(f"[edges] {year}:frequency exists -> skip ({out_path})")
        return created
    if score_type == 'pitch_type':
        # Create per pitch_type subdirectories
        out_parent = os.path.join(base_dir, 'pitch_type')
        os.makedirs(out_parent, exist_ok=True)
        if pitch_types is None:
            pitch_types = ALLOWED_PITCH_TYPES
        for pt in pitch_types:
            sub_dir = os.path.join(out_parent, pt)
            os.makedirs(sub_dir, exist_ok=True)
            out_path = os.path.join(sub_dir, f"{year}_edges_only.csv")
            if force or (not os.path.isfile(out_path)):
                rows = []
                pt_df = df[df['pitch_type']==pt]
                if pt_df.empty: continue
                if progress: print(f"[edges] {year}:pitch_type[{pt}] rows={len(pt_df)} generating -> {out_path}")
                for batter_name, pitcher_name, event in pt_df[['batter_name', _pcol, 'events']].itertuples(index=False):
                    r = _score_event(batter_name, pitcher_name, event, BASE_BATTER_SCORING, BASE_PITCHER_SCORING)
                    if r: rows.append(r)
                if rows:
                    edf = pd.DataFrame(rows, columns=['winner','loser','score','who_won'])
                    edf = edf.groupby(['winner','loser','who_won']).sum()
                    edf.to_csv(out_path)
                    created.append(out_path)
                    if progress: print(f"[edges] wrote {out_path}")
            else:
                if progress: print(f"[edges] {year}:pitch_type[{pt}] exists -> skip ({out_path})")
        return created
    if score_type == 'inning':
        out_parent = os.path.join(base_dir, 'inning')
        os.makedirs(out_parent, exist_ok=True)
        if innings is None:
            innings = list(range(1,10))
        for inn in innings:
            sub_dir = os.path.join(out_parent, str(inn))
            os.makedirs(sub_dir, exist_ok=True)
            out_path = os.path.join(sub_dir, f"{year}_edges_only.csv")
            if force or (not os.path.isfile(out_path)):
                rows = []
                inn_df = df[df['inning']==inn]
                if inn_df.empty: continue
                if progress: print(f"[edges] {year}:inning[{inn}] rows={len(inn_df)} generating -> {out_path}")
                for batter_name, pitcher_name, event in inn_df[['batter_name', _pcol, 'events']].itertuples(index=False):
                    r = _score_event(batter_name, pitcher_name, event, BASE_BATTER_SCORING, BASE_PITCHER_SCORING)
                    if r: rows.append(r)
                if rows:
                    edf = pd.DataFrame(rows, columns=['winner','loser','score','who_won'])
                    edf = edf.groupby(['winner','loser','who_won']).sum()
                    edf.to_csv(out_path)
                    created.append(out_path)
                    if progress: print(f"[edges] wrote {out_path}")
            else:
                if progress: print(f"[edges] {year}:inning[{inn}] exists -> skip ({out_path})")
        return created
    return []

def _unipartite_vectorized(
    group_df: pd.DataFrame,
    *,
    metric: str = 'sum',
    year: int | None = None,
    raw_data_dir: str = 'At Bats/general_data',
    winners_role: str | None = None,
    opponent_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    # Vectorized i->j edge computation using a dense opponent matrix
    # Rows: players (winners in group), Cols: opponents (losers in group)
    # Edge(i->j) = sum(max(W[i,:] - W[j,:], 0)) over common opponents (non-neg implied by max)
    # 1) Aggregate to ensure unique (winner, loser)
    g = group_df.groupby(['winner','loser'], as_index=False)['score'].sum()
    # Rows should be winners only (batters for batter-group, pitchers for pitcher-group)
    players = g['winner'].unique()
    losers = g['loser'].unique()
    # Map players and opponents to indices
    p_to_i = {p:i for i,p in enumerate(players)}
    opps = np.unique(losers)
    o_to_j = {o:j for j,o in enumerate(opps)}
    # Build dense matrix (float32 to reduce memory), default zeros
    W = np.zeros((players.size, opps.size), dtype=np.float32)
    # Fill values
    pi = g['winner'].map(p_to_i).to_numpy()
    pj = g['loser'].map(o_to_j).to_numpy()
    vals = g['score'].astype(np.float32).to_numpy()
    W[pi, pj] = vals
    # Build opponent weight vector aligned to opps order (defaults 1.0)
    opp_w = None
    if opponent_weights:
        try:
            opp_w = np.array([float(opponent_weights.get(str(o), 1.0)) for o in opps], dtype=np.float32)
        except Exception:
            opp_w = None
    # Optional rate normalization: divide each player's per-opponent sum by their PAs vs that opponent
    def _dbg_log(msg: str):
        try:
            # Mirror to console and append to debug log
            print(msg)
            if metric == 'rate':
                os.makedirs(os.path.join('outputs'), exist_ok=True)
                log_path = os.path.join('outputs', 'debug_unipartite_rate.log')
                ts = time.strftime('%Y-%m-%d %H:%M:%S')
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f"[{ts}] year={year} role={winners_role} players={players.size} opps={opps.size} :: {msg}\n")
        except Exception:
            # Best-effort only; ignore logging failures
            pass
    presence_mask = None  # used later to mark common opponents in rate mode
    if metric == 'rate':
        _dbg_log(f"enter rate-mode: players={players.size}, opps={opps.size}")
        if year is None:
            raise ValueError("metric 'rate' requires year")
        raw_path = os.path.join(raw_data_dir, f"at_bat_data_{year}.csv")
        if not os.path.isfile(raw_path):
            raise FileNotFoundError(raw_path)
        raw = pd.read_csv(raw_path, usecols=['batter_name','pitcher_name','events'])
        # Build PA counts per (batter, pitcher)
        pa = raw.groupby(['batter_name','pitcher_name'], as_index=False)['events'].count()
        pa.rename(columns={'events':'pa'}, inplace=True)
        # Best-effort: resolve MLBAM_<id> names to 'First Last' to match edges-only naming (both batter and pitcher if present)
        try:
            # Reuse edges mapper which is proven to work in this repo
            tmp = pa[['batter_name','pitcher_name']].copy()
            mapped = _resolve_names_in_edges_df(tmp.rename(columns={'batter_name':'winner','pitcher_name':'loser'}))
            if mapped is not None and not mapped.empty and all(c in mapped.columns for c in ['winner','loser']):
                pa['batter_name'] = mapped['winner']
                pa['pitcher_name'] = mapped['loser']
                _dbg_log("applied _resolve_names_in_edges_df to PA names")
        except Exception:
            pass
        try:
            import re as _re
            def _resolve_ids(series: pd.Series) -> dict[int,str]:
                mask = series.astype(str).str.match(r'^MLBAM[_\s-]?(\d+)$')
                if not mask.any():
                    return {}
                ids = series.loc[mask].astype(str).str.extract(r'(\d+)')[0].dropna().astype(int).unique().tolist()
                name_map: dict[int,str] = {}
                try:
                    import importlib as _il
                    _pb = _il.import_module('pybaseball')
                    dfm = _pb.playerid_reverse_lookup(ids, key_type='mlbam')
                    if dfm is not None and not dfm.empty and all(c in dfm.columns for c in ['key_mlbam','name_first','name_last']):
                        dfm = dfm[['key_mlbam','name_first','name_last']].copy()
                        dfm['key_mlbam'] = pd.to_numeric(dfm['key_mlbam'], errors='coerce').astype('Int64')
                        dfm['name_first'] = dfm['name_first'].astype(str).str.capitalize()
                        dfm['name_last'] = dfm['name_last'].astype(str).str.capitalize()
                        dfm['full'] = dfm[['name_first','name_last']].agg(' '.join, axis=1)
                        name_map.update({int(k): v for k,v in dfm.set_index('key_mlbam')['full'].dropna().items()})
                except Exception:
                    pass
                if not name_map:
                    try:
                        import baseball_scraper as _bs
                        dfm2 = _bs.playerid_reverse_lookup(ids, key_type='mlbam')
                        if dfm2 is not None and not dfm2.empty and all(c in dfm2.columns for c in ['key_mlbam','name_first','name_last']):
                            dfm2 = dfm2[['key_mlbam','name_first','name_last']].copy()
                            dfm2['key_mlbam'] = pd.to_numeric(dfm2['key_mlbam'], errors='coerce').astype('Int64')
                            dfm2['name_first'] = dfm2['name_first'].astype(str).str.capitalize()
                            dfm2['name_last'] = dfm2['name_last'].astype(str).str.capitalize()
                            dfm2['full'] = dfm2[['name_first','name_last']].agg(' '.join, axis=1)
                            name_map.update({int(k): v for k,v in dfm2.set_index('key_mlbam')['full'].dropna().items()})
                    except Exception:
                        pass
                # Final fallback to MLB Stats API for any remaining ids
                try:
                    remaining = [i for i in ids if i not in name_map]
                    if remaining:
                        from urllib.request import urlopen as _urlopen
                        import json as _json
                        for i0 in range(0, len(remaining), 50):
                            batch = remaining[i0:i0+50]
                            url = 'https://statsapi.mlb.com/api/v1/people?personIds=' + ','.join(str(x) for x in batch)
                            with _urlopen(url, timeout=10) as resp:
                                data = _json.loads(resp.read().decode('utf-8'))
                            for p in data.get('people', []):
                                pid = p.get('id'); full = p.get('fullName')
                                if isinstance(pid, int) and isinstance(full, str) and full:
                                    name_map[pid] = full
                except Exception:
                    pass
                return name_map
            # Apply for batter_name
            bmap = _resolve_ids(pa['batter_name'])
            if bmap:
                def _map_b(v: str) -> str:
                    m = _re.search(r'(\d+)$', str(v))
                    if m:
                        return bmap.get(int(m.group(1)), v)
                    return v
                pa['batter_name'] = pa['batter_name'].map(_map_b)
            # Apply for pitcher_name
            pmap = _resolve_ids(pa['pitcher_name'])
            if pmap:
                def _map_p(v: str) -> str:
                    m = _re.search(r'(\d+)$', str(v))
                    if m:
                        return pmap.get(int(m.group(1)), v)
                    return v
                pa['pitcher_name'] = pa['pitcher_name'].map(_map_p)
        except Exception:
            pass
        # Build normalization helpers for robust name matching (diacritics/case/spacing)
        import unicodedata as _ud
        def _norm(s: Any) -> str:
            t = str(s) if not pd.isna(s) else ''
            t = t.strip()
            # Normalize unicode to strip accents
            t = _ud.normalize('NFKD', t)
            t = ''.join(c for c in t if not _ud.combining(c))
            # Collapse inner whitespace and lowercase
            t = ' '.join(t.split())
            return t.lower()
        players_norm_map = {_norm(p): i for i, p in enumerate(players)}
        opps_norm_map = {_norm(o): j for j, o in enumerate(opps)}
        # Normalized PA name columns
        pa_b_norm = pa['batter_name'].apply(_norm)
        pa_p_norm = pa['pitcher_name'].apply(_norm)
        # Determine orientation for mapping based on winners_role
        # - For batter group: players=batter_name, opps=pitcher_name
        # - For pitcher group: players=pitcher_name, opps=batter_name
        if winners_role is None:
            # Heuristic: choose orientation with more valid mappings
            pi_bat = pa_b_norm.map(players_norm_map)
            pj_bat = pa_p_norm.map(opps_norm_map)
            cnt_bat = int((~pd.isna(pi_bat) & ~pd.isna(pj_bat)).sum())
            pi_pit = pa_p_norm.map(players_norm_map)
            pj_pit = pa_b_norm.map(opps_norm_map)
            cnt_pit = int((~pd.isna(pi_pit) & ~pd.isna(pj_pit)).sum())
            winners_role = 'batter' if cnt_bat >= cnt_pit else 'pitcher'
            _dbg_log(f"auto-orientation: cand_batter_pairs={cnt_bat}, cand_pitcher_pairs={cnt_pit} -> choose {winners_role}")
        # Log intersection diagnostics
        try:
            edges_players_set = set(players_norm_map.keys())
            edges_opps_set = set(opps_norm_map.keys())
            pa_b_set = set(pa_b_norm.unique())
            pa_p_set = set(pa_p_norm.unique())
            inter_players = len(edges_players_set & pa_b_set)
            inter_opps_bp = len(edges_opps_set & pa_p_set)
            inter_players_alt = len(edges_players_set & pa_p_set)
            inter_opps_alt = len(edges_opps_set & pa_b_set)
            _dbg_log(
                f"name coverage: edges_players={len(edges_players_set)}, edges_opps={len(edges_opps_set)}, "
                f"pa_bat={len(pa_b_set)}, pa_pit={len(pa_p_set)}, "
                f"inter(players,pa_b)={inter_players}, inter(opps,pa_p)={inter_opps_bp}, "
                f"alt inter(players,pa_p)={inter_players_alt}, alt inter(opps,pa_b)={inter_opps_alt}"
            )
            if inter_players == 0 or inter_opps_bp == 0:
                # Show a few example names from each side to eyeball formatting issues
                samp_players = list(sorted(list(edges_players_set))[:5])
                samp_pa_b = list(sorted(list(pa_b_set))[:5])
                samp_opps = list(sorted(list(edges_opps_set))[:5])
                samp_pa_p = list(sorted(list(pa_p_set))[:5])
                _dbg_log(f"samples players={samp_players}")
                _dbg_log(f"samples pa_bat={samp_pa_b}")
                _dbg_log(f"samples opps={samp_opps}")
                _dbg_log(f"samples pa_pit={samp_pa_p}")
        except Exception:
            pass
        if winners_role == 'batter':
            pai = pa_b_norm.map(players_norm_map).to_numpy()
            paj = pa_p_norm.map(opps_norm_map).to_numpy()
        elif winners_role == 'pitcher':
            pai = pa_p_norm.map(players_norm_map).to_numpy()
            paj = pa_b_norm.map(opps_norm_map).to_numpy()
        else:
            raise ValueError("winners_role must be 'batter' or 'pitcher'")
        pav = pa['pa'].astype(np.float32).to_numpy()
        # Only apply where indices are valid (drop NaNs)
        mask = (~pd.isna(pai)) & (~pd.isna(paj))
        mapped_rows = int(mask.sum())
        _dbg_log(f"mapped PA rows: {mapped_rows} / {len(pa)}")
        pai = pai[mask].astype(int)
        paj = paj[mask].astype(int)
        pav = pav[mask]
        if mapped_rows:
            _dbg_log(f"unique mapped players={int(np.unique(pai).size)}, unique mapped opps={int(np.unique(paj).size)}")
        # Build PA matrix and normalize; avoid division by zero
        PA = np.zeros_like(W)
        PA[pai, paj] = pav
        presence = (PA > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            before_nz = int((W > 0).sum())
            valid_pa = int(presence.sum())
            W = np.where(presence, W / np.where(PA>0, PA, 1), 0.0)
            after_nz = int((W > 0).sum())
            _dbg_log(f"rate normalization: W>0 before={before_nz}, PA>0 cells={valid_pa}, W>0 after={after_nz}")
        presence_mask = presence
        # Basic stats on presence per player/opponent
        try:
            row_presence = presence.sum(axis=1)
            col_presence = presence.sum(axis=0)
            _dbg_log(f"presence rows: any={int((row_presence>0).sum())}/{players.size}, mean={float(row_presence.mean()):.3f}, median={float(np.median(row_presence)):.3f}")
            _dbg_log(f"presence cols: any={int((col_presence>0).sum())}/{opps.size}, mean={float(col_presence.mean()):.3f}, median={float(np.median(col_presence)):.3f}")
        except Exception:
            pass
    # Compute i->j = sum_k max(W[i,k] - W[j,k], 0) only over common opponents where both W>0
    n = players.size
    m = opps.size
    results = []
    row_chunk = 128  # rows per chunk (players)
    col_chunk = 128  # opponent columns per block
    last_pct = -1
    # For sum metric use W>0 to mark activity; for rate metric use presence (PA>0) so common opponents means both faced the opponent
    if metric == 'rate' and presence_mask is not None:
        NZ = presence_mask
    else:
        NZ = W > 0
    for start in range(0, n, row_chunk):
        end = min(n, start + row_chunk)
        Wc = W[start:end, :]            # (a, m)
        NZc = NZ[start:end, :]          # (a, m)
        # Accumulator for this chunk
        relu_sum = np.zeros((end - start, n), dtype=np.float32)
        for k0 in range(0, m, col_chunk):
            k1 = min(m, k0 + col_chunk)
            Wc_blk = Wc[:, k0:k1]       # (a, b)
            Wa_blk = W[:, k0:k1]        # (n, b)
            NZc_blk = NZc[:, k0:k1]     # (a, b)
            NZa_blk = NZ[:, k0:k1]      # (n, b)
            # Apply opponent weights per column if provided
            if opp_w is not None:
                w_blk = opp_w[k0:k1][None, :]  # shape (1, b)
                Wc_blk = Wc_blk * w_blk
                Wa_blk = Wa_blk * w_blk
            # Common opponent mask per (i,j,k): (a,b,n)
            common = NZc_blk[:, :, None] & NZa_blk.T[None, :, :]
            # Differences per (i,j,k)
            diff = Wc_blk[:, :, None] - Wa_blk.T[None, :, :]
            # ReLU and mask to common opponents; sum over k-block
            relu_block = np.maximum(diff, 0.0)
            relu_block = np.where(common, relu_block, 0.0)
            relu_sum += relu_block.sum(axis=1)
        # Emit edges where total > 0, excluding i==j
        ii, jj = np.where(relu_sum > 0)
        valid = (ii + start) != jj
        ii = ii[valid]
        jj = jj[valid]
        scores = relu_sum[ii, jj]
        for a, b, s in zip(ii + start, jj, scores):
            results.append([players[a], players[b], float(s)])
        # Coarse progress
        pct = int((end * 100) / max(n, 1))
        if pct // 10 != last_pct // 10:
            _dbg_log(f"vectorized progress: {pct}% ({end}/{n})")
            last_pct = pct
    edf = pd.DataFrame(results, columns=['winner','loser','score'])
    _dbg_log(f"vectorized produced {len(edf)} edges (players={n}, opps={m})")
    return edf


def to_2_unipartite(edge_only_path: str, save_batter: str, save_pitcher: str, *, metric: str = 'sum', raw_data_dir: str = 'At Bats/general_data'):
    # Re-implement minimal variant of BipartiteTo2Unipartite.to2Unipartite for internal use, with progress prints
    if os.path.isfile(edge_only_path):
        print(f"[unipartite] reading bipartite edges -> {edge_only_path}")
    df = pd.read_csv(edge_only_path)
    bwe = df[df.who_won=='batter'][['winner','loser','score']].sort_values(['winner','loser'])
    pwe = df[df.who_won=='pitcher'][['winner','loser','score']].sort_values(['winner','loser'])
    print(f"[unipartite] split: batter_edges={len(bwe)}, pitcher_edges={len(pwe)}")
    # Determine run-time settings passed indirectly via environment: use default config
    # If available, prefer processing.unipartite_metric; default to 'sum'.
    # metric and raw_data_dir are provided by caller; defaults keep behavior stable
    def _extract_mlbam_id(val: str):
        try:
            import re
            s = str(val).strip()
            m = re.search(r'(?i)mlbam[\s_-]*(\d+)$', s)
            if m:
                return int(m.group(1))
            # Also accept plain numeric strings
            if s.isdigit():
                return int(s)
        except Exception:
            return None
        return None

    def _resolve_winner_names(gdf: pd.DataFrame) -> pd.DataFrame:
        # Replace winner names like MLBAM_<id> with resolved 'First Last' when possible
        try:
            w = gdf['winner'].astype(str)
            mask = w.apply(lambda x: _extract_mlbam_id(x) is not None)
            if not mask.any():
                return gdf
            ids = list({ _extract_mlbam_id(x) for x in w[mask] if _extract_mlbam_id(x) is not None })
            if not ids:
                return gdf
            name_map = {}
            # Try pybaseball first
            try:
                import importlib as _il
                _pb = _il.import_module('pybaseball')
                dfm = _pb.playerid_reverse_lookup(ids, key_type='mlbam')
                if dfm is not None and not dfm.empty and all(c in dfm.columns for c in ['key_mlbam','name_first','name_last']):
                    dfm = dfm[['key_mlbam','name_first','name_last']].copy()
                    dfm['key_mlbam'] = pd.to_numeric(dfm['key_mlbam'], errors='coerce').astype('Int64')
                    dfm['name_first'] = dfm['name_first'].astype(str).str.capitalize()
                    dfm['name_last'] = dfm['name_last'].astype(str).str.capitalize()
                    dfm['full'] = dfm[['name_first','name_last']].agg(' '.join, axis=1)
                    name_map.update({int(k): v for k,v in dfm.set_index('key_mlbam')['full'].dropna().items()})
            except Exception:
                pass
            # Fallback to baseball_scraper
            if len(name_map) < len(ids):
                try:
                    import baseball_scraper as _bs
                    dfm2 = _bs.playerid_reverse_lookup(ids, key_type='mlbam')
                    if dfm2 is not None and not dfm2.empty and all(c in dfm2.columns for c in ['key_mlbam','name_first','name_last']):
                        dfm2 = dfm2[['key_mlbam','name_first','name_last']].copy()
                        dfm2['key_mlbam'] = pd.to_numeric(dfm2['key_mlbam'], errors='coerce').astype('Int64')
                        dfm2['name_first'] = dfm2['name_first'].astype(str).str.capitalize()
                        dfm2['name_last'] = dfm2['name_last'].astype(str).str.capitalize()
                        dfm2['full'] = dfm2[['name_first','name_last']].agg(' '.join, axis=1)
                        name_map.update({int(k): v for k,v in dfm2.set_index('key_mlbam')['full'].dropna().items()})
                except Exception:
                    pass
            if name_map:
                def _map_winner(val: str) -> str:
                    pid = _extract_mlbam_id(val)
                    if pid is not None:
                        return name_map.get(pid, val)
                    return val
                before = int(mask.sum())
                gdf = gdf.copy()
                gdf.loc[mask, 'winner'] = gdf.loc[mask, 'winner'].map(_map_winner)
                after = int(gdf['winner'].astype(str).apply(lambda x: _extract_mlbam_id(x) is not None).sum())
                fixed = before - after
                if fixed > 0:
                    print(f"[unipartite] resolved {fixed}/{before} MLBAM_<id> winner names")
        except Exception:
            pass
        return gdf

    def _resolve_names_in_edges(edf: pd.DataFrame) -> pd.DataFrame:
        try:
            cols = ['winner','loser']
            masks = [edf[c].astype(str).apply(lambda x: _extract_mlbam_id(x) is not None) for c in cols]
            ids = []
            for m,c in zip(masks, cols):
                if m.any():
                    ids.extend([ _extract_mlbam_id(x) for x in edf.loc[m,c].astype(str) if _extract_mlbam_id(x) is not None ])
            ids = sorted(set(ids))
            if not ids:
                return edf
            name_map = {}
            # Try pybaseball
            try:
                import importlib as _il
                _pb = _il.import_module('pybaseball')
                df1 = _pb.playerid_reverse_lookup(ids, key_type='mlbam')
                if df1 is not None and not df1.empty and all(c in df1.columns for c in ['key_mlbam','name_first','name_last']):
                    df1 = df1[['key_mlbam','name_first','name_last']].copy()
                    df1['key_mlbam'] = pd.to_numeric(df1['key_mlbam'], errors='coerce').astype('Int64')
                    df1['name_first'] = df1['name_first'].astype(str).str.capitalize()
                    df1['name_last'] = df1['name_last'].astype(str).str.capitalize()
                    df1['full'] = df1[['name_first','name_last']].agg(' '.join, axis=1)
                    name_map.update({int(k): v for k,v in df1.set_index('key_mlbam')['full'].dropna().items()})
            except Exception:
                pass
            # Try baseball_scraper
            if len(name_map) < len(ids):
                try:
                    import baseball_scraper as _bs
                    df2 = _bs.playerid_reverse_lookup(ids, key_type='mlbam')
                    if df2 is not None and not df2.empty and all(c in df2.columns for c in ['key_mlbam','name_first','name_last']):
                        df2 = df2[['key_mlbam','name_first','name_last']].copy()
                        df2['key_mlbam'] = pd.to_numeric(df2['key_mlbam'], errors='coerce').astype('Int64')
                        df2['name_first'] = df2['name_first'].astype(str).str.capitalize()
                        df2['name_last'] = df2['name_last'].astype(str).str.capitalize()
                        df2['full'] = df2[['name_first','name_last']].agg(' '.join, axis=1)
                        name_map.update({int(k): v for k,v in df2.set_index('key_mlbam')['full'].dropna().items()})
                except Exception:
                    pass
            # Fallback to MLB Stats API
            if len(name_map) < len(ids):
                try:
                    from urllib.request import urlopen as _urlopen
                    import json as _json
                    # batch up to 50 ids
                    remaining = [i for i in ids if i not in name_map]
                    for i0 in range(0, len(remaining), 50):
                        batch = remaining[i0:i0+50]
                        url = 'https://statsapi.mlb.com/api/v1/people?personIds=' + ','.join(str(x) for x in batch)
                        with _urlopen(url, timeout=10) as resp:
                            data = _json.loads(resp.read().decode('utf-8'))
                        for p in data.get('people', []):
                            pid = p.get('id')
                            full = p.get('fullName')
                            if isinstance(pid, int) and isinstance(full, str) and full:
                                name_map[pid] = full
                except Exception:
                    pass
            if name_map:
                def _map_val(v: str) -> str:
                    pid = _extract_mlbam_id(v)
                    if pid is not None:
                        return name_map.get(pid, v)
                    return v
                before = sum(m.sum() for m in masks)
                edf = edf.copy()
                for c in cols:
                    edf[c] = edf[c].map(_map_val)
                after = sum(edf[c].astype(str).apply(lambda x: _extract_mlbam_id(x) is not None).sum() for c in cols)
                fixed = before - after
                if fixed > 0:
                    print(f"[unipartite] resolved {fixed}/{before} MLBAM_<id> names in edges")
        except Exception:
            pass
        return edf

    def group_edges(gwe_df, out_path, *, winners_role: str | None = None):
        # Ensure we work with aggregated edges per (winner, loser) pair
        gwe_df = gwe_df.groupby(['winner','loser'], as_index=False)['score'].sum()
        # Best-effort resolve of winner names from MLBAM_<id>
        gwe_df = _resolve_winner_names(gwe_df)
        # Choose processing mode
        from_config = True  # default vectorized enabled via loader
        try:
            # Late import to avoid circular; cfg injected via closure in generate_edges/run_pipeline
            from config.loader import load_config  # not used here; just to indicate context
        except Exception:
            pass
        # Determine year from out_path filename when available
        _yr = None
        try:
            base = os.path.basename(out_path)
            _yr = int(base.split('_')[0])
        except Exception:
            _yr = None
        def _pairwise_compute(df_in: pd.DataFrame) -> pd.DataFrame:
            grouped = {}
            for w, sub in df_in.groupby('winner'):
                arr = sub[['loser','score']].to_numpy()
                grouped[w] = (arr[:,0], arr[:,1].astype(float))
            group_players = np.unique(np.hstack((df_in['winner'].unique(), df_in['loser'].unique())))
            total_players = len(group_players)
            print(f"[unipartite] computing group graph for {total_players} players -> {out_path}")
            player_edgelist = []
            last_pct = -1
            for i, p1 in enumerate(group_players):
                pct = int((i+1) * 100 / max(total_players, 1))
                if pct // 10 != last_pct // 10:
                    print(f"[unipartite] progress: {pct}% ({i+1}/{total_players}) for {out_path}")
                    last_pct = pct
                if p1 not in grouped:
                    continue
                p1_opp, p1_sc = grouped[p1]
                if p1_opp.size == 0:
                    continue
                for j, p2 in enumerate(group_players):
                    if i == j:
                        continue
                    if p2 not in grouped:
                        continue
                    p2_opp, p2_sc = grouped[p2]
                    if p2_opp.size == 0:
                        continue
                    inter, idx1, idx2 = np.intersect1d(p1_opp, p2_opp, assume_unique=True, return_indices=True)
                    if inter.size == 0:
                        continue
                    a = p1_sc[idx1]
                    b = p2_sc[idx2]
                    relu = np.maximum(a - b, 0.0)
                    total = float(relu.sum())
                    if total > 0.0:
                        player_edgelist.append([p1, p2, total])
            return pd.DataFrame(player_edgelist, columns=['winner','loser','score'])

        edf: pd.DataFrame | None = None
        # 1) Try vectorized path first (honors metric='rate' with PA normalization internally)
        try:
            edf = _unipartite_vectorized(gwe_df, metric=metric, year=_yr, raw_data_dir=raw_data_dir, winners_role=winners_role)
        except Exception as e:
            print(f"[unipartite] vectorized path failed ({e}); will try pairwise next")
            edf = None
        # 2) If vectorized returned empty or failed, fallback to pairwise. For 'rate', normalize via PAs first.
        if edf is None or edf.empty:
            if edf is not None and edf.empty:
                print("[unipartite] vectorized returned 0 edges; retrying with pairwise computation")
            try:
                df_in = gwe_df
                if metric == 'rate':
                    if _yr is None:
                        raise ValueError("metric 'rate' requires identifiable year in output path")
                    raw_path = os.path.join(raw_data_dir, f"at_bat_data_{_yr}.csv")
                    raw = pd.read_csv(raw_path, usecols=['batter_name','pitcher_name','events'])
                    pa = raw.groupby(['batter_name','pitcher_name'], as_index=False)['events'].count().rename(columns={'events':'pa'})
                    if winners_role == 'batter':
                        merged = gwe_df.merge(pa.rename(columns={'batter_name':'winner','pitcher_name':'loser'}), on=['winner','loser'], how='left')
                    else:
                        merged = gwe_df.merge(pa.rename(columns={'pitcher_name':'winner','batter_name':'loser'}), on=['winner','loser'], how='left')
                    merged['pa'] = pd.to_numeric(merged['pa'], errors='coerce').fillna(0)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        merged['score'] = np.where(merged['pa'] > 0, merged['score'] / merged['pa'], 0.0)
                    df_in = merged[merged['score'] > 0]
                edf = _pairwise_compute(df_in)
            except Exception as e:
                print(f"[unipartite] pairwise fallback failed ({e}); using unnormalized pairwise")
                edf = _pairwise_compute(gwe_df)
        # If rate produced nothing, retry with sum metric as safety
        if metric == 'rate' and (edf is None or edf.empty):
            try:
                print("[unipartite] rate produced 0 edges; retrying with metric='sum'")
                # Recompute using sum metric (vectorized)
                edf = _unipartite_vectorized(gwe_df, metric='sum', year=_yr, raw_data_dir=raw_data_dir, winners_role=winners_role)
            except Exception:
                edf = _pairwise_compute(gwe_df)
        # Post-process names in both columns (winner/loser) and write
        edf = _resolve_names_in_edges(edf)
        print(f"[unipartite] writing -> {out_path} (edges={len(edf)})")
        edf.to_csv(out_path, index=False)
    group_edges(bwe, save_batter, winners_role='batter')
    group_edges(pwe, save_pitcher, winners_role='pitcher')


def ensure_scraped(years: List[int], raw_data_dir: str, force: bool, progress: bool=True) -> List[str]:
    paths = []
    for y in years:
        paths.append(scrape_year(y, raw_data_dir, force=force, progress=progress))
    return paths

# ------------------------- Graph / Ranking --------------------------------- #

def load_edge_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def make_graph_from_edge_csv(path: str, weights: bool=True, validation_folds: int=0, *, seed: int | None = None, sample_as_train: bool = True, index_base: int = 0, fast: bool = False):
    df = pd.read_csv(path)
    # Best-effort: resolve MLBAM-coded names in the edges for human-readable files and stable downstream joins
    try:
        if fast:
            # Skip expensive MLBAM detection and remapping in fast mode (e.g., k-sweep)
            raise Exception('skip_mlbam_detection_fast')
        if 'winner' in df.columns and 'loser' in df.columns:
            s_w = df['winner'].astype(str)
            s_l = df['loser'].astype(str)
            has_tags = s_w.str.contains(r'(?i)mlbam[\s_-]*\d+$', regex=True).any() or s_l.str.contains(r'(?i)mlbam[\s_-]*\d+$', regex=True).any()
            # Faster numeric check than full regex fullmatch
            try:
                has_digits = s_w.str.isnumeric().any() or s_l.str.isnumeric().any()
            except Exception:
                has_digits = s_w.str.match(r'^\d+$', regex=True).any() or s_l.str.match(r'^\d+$', regex=True).any()
            if has_tags or has_digits:
                mapped = _resolve_names_in_edges_df(df[['winner','loser']].copy().assign(score=df.get('score', 1.0)))
                # Preserve original score column if present
                if 'score' in df.columns:
                    mapped['score'] = pd.to_numeric(df['score'], errors='coerce')
                df[['winner','loser']] = mapped[['winner','loser']]
                # Persist mapped names back to the edge CSV for user convenience
                try:
                    df.to_csv(path, index=False)
                except Exception:
                    pass
    except Exception:
        pass
    # Guard: if no edges, return empty graph/adjacency and let caller skip
    if df is None or len(df) == 0:
        G = nx.DiGraph()
        try:
            import scipy.sparse as sp  # type: ignore
            A = sp.csr_matrix((0, 0))
        except Exception:
            A = np.zeros((0, 0), dtype=float)
        return G, A, [], None, None
    # Expect columns winner, loser, score
    cols = [c for c in ['winner','loser','score'] if c in df.columns]
    if len(cols) != 3:
        # Fallback: assume first 3 columns are the expected ones
        cols = list(df.columns[:3])
    # Use list of tuples to avoid numpy row unpacking quirks in networkx
    raw_edge_list = list(df[cols].itertuples(index=False, name=None))
    # Sanitize: ensure every element is a (u,v,w) 3-tuple with numeric weight
    edge_list = []
    for row in raw_edge_list:
        if row is None:
            continue
        # Accept tuples/lists of len>=2
        if isinstance(row, (list, tuple)):
            if len(row) >= 3:
                u, v, w = row[0], row[1], row[2]
            elif len(row) == 2:
                # If only (u,v) provided, default weight to 1.0
                u, v = row[0], row[1]
                w = 1.0
            else:
                continue
        else:
            # Unexpected type
            continue
        # Coerce node names to strings and weight to float; drop NaNs
        if pd.isna(u) or pd.isna(v):
            continue
        try:
            w = float(w)
        except Exception:
            # Skip non-numeric weights
            continue
        edge_list.append((str(u), str(v), float(w)))
    G = nx.DiGraph()
    if validation_folds>0:
        m = len(edge_list)
        if m == 0:
            train_edges = edge_list
            test_edges = None
        else:
            # Use a deterministic RNG by default to ensure reproducible splits across runs
            rng = np.random.RandomState(seed if seed is not None else 42)
            sel_inds = rng.choice(np.arange(m, dtype=int), int(m*(1-(1/validation_folds))), replace=False)
            not_sel = np.setdiff1d(np.arange(m, dtype=int), sel_inds)
            # Old code path hypothesis: sel could be train or test depending on where slicing was applied.
            if sample_as_train:
                train_edges = [edge_list[i] for i in sel_inds]
                test_edges = [edge_list[i] for i in not_sel]
            else:
                test_edges = [edge_list[i] for i in sel_inds]
                train_edges = [edge_list[i] for i in not_sel]
    else:
        train_edges = edge_list
        test_edges = None
    if weights:
        # Add edges explicitly to avoid tuple-unpacking issues within NetworkX
        for (u, v, w) in train_edges:
            # Skip non-finite weights
            if w is None or (isinstance(w, float) and (np.isnan(w) or np.isinf(w))):
                continue
            try:
                G.add_edge(u, v, weight=float(w))
            except Exception:
                continue
    else:
        # Add only (u,v) when ignoring weights
        G.add_edges_from((u, v) for (u, v, _w) in train_edges)
    node_list = list(G.nodes())
    # NetworkX 3.x removed to_scipy_sparse_matrix; prefer the array version when available
    try:
        A = nx.to_scipy_sparse_matrix(G, dtype=float, nodelist=node_list)  # NetworkX <3.0
    except AttributeError:  # NetworkX >=3.0
        A = nx.to_scipy_sparse_array(G, dtype=float, nodelist=node_list)
    # Normalize to CSR matrix for springrank compatibility
    try:
        import scipy.sparse as sp
        A = sp.csr_matrix(A)
    except Exception as e:
        # Keep A as-is; downstream SpringRank accepts sparse arrays too
        try:
            print(f"[graph][warn] failed to convert adjacency to CSR: {e}")
        except Exception:
            pass
    return G, A, node_list, train_edges, test_edges


def spring_rank(A, node_list: List[str]):
    # Use estimator pattern from springrank package
    model = sr.SpringRank(alpha=0)
    model.fit(A)
    # Some versions expose ranks on attribute 'ranks' (list-like)
    sr_rank = np.asarray(getattr(model, 'ranks', getattr(model, 'ranks_', None)))
    if sr_rank is None:
        # As a fallback, try model.get_rescaled_ranks with target_scale=0.5 to retrieve relative ordering
        sr_rank = np.asarray(model.get_rescaled_ranks(target_scale=0.5))
    sr_sorted = [[node_list[i], float(sr_rank[i])] for i in range(len(node_list))]
    sr_sorted.sort(reverse=True, key=lambda x: x[1])
    return sr_rank, sr_sorted


def scale_ranks(A, raw_ranks, a=0.01, b=20, scale=0.75):
    # Prefer the package's built-in rescaling if available
    model = sr.SpringRank(alpha=0)
    # Attach the adjacency and existing ranks to mirror fitted state
    try:
        model.A = A
        model.ranks = np.asarray(raw_ranks)
        return np.asarray(model.get_rescaled_ranks(target_scale=scale))
    except Exception:
        # Fallback to identity if rescaling not available
        return np.asarray(raw_ranks)

# ------------------------- Validation metrics (ACC/AUC) -------------------- #

def _compute_acc_auc(
    sorted_ranks: List[List[Any]],
    test_edges: Optional[np.ndarray],
    *,
    auc_mode: str = "balanced-negatives",
    k_neg: int = 1,
    auto_flip: bool = False,
    acc_mode: str = "positive-only",
    seed: Optional[int] = 42,
) -> Optional[Tuple[float, float, int]]:
    """Compute ACC/AUC against held-out directed edges using rank differences.

    Modes:
    - balanced-negatives (default): For each held-out positive (u->v), score = r[u]-r[v] with label 1,
      and add k_neg negatives for the same source u by sampling v' not in the positive set, label 0.
      This tests whether rank differences discriminate winners from random non-winners.
        - legacy: Match legacy_auc_demo.py behavior on the raw edge CSV. Treat ground-truth label as
            1 if the held-out edge weight > 0 else 0. Predict label as 1 if dv = r[u]-r[v] > 0 else 0.
            Report ACC(obs, preds) and AUC(obs, preds) where the AUC is computed on binary predictions
            (as in the legacy script), not on continuous scores.

    Returns (accuracy, auc, used_test_edges) or None if no edges usable.
    """
    # Initialize safe defaults to avoid UnboundLocalError in any early-return path
    acc: float = 0.0
    auc: float = 0.5
    used: int = 0
    if test_edges is None or len(test_edges) == 0:
        return None
    # Map player -> rank score
    rank_map = {name: float(score) for name, score in sorted_ranks}
    # Legacy mode: ACC/AUC on binary predictions vs weight-derived labels (edge weight > 0)
    if str(auc_mode).lower() == "legacy":
        acc_preds: List[int] = []
        acc_obs: List[int] = []
        used = 0
        for edge in test_edges:
            try:
                u, v, w = str(edge[0]), str(edge[1]), edge[2]
                si = rank_map.get(u)
                sj = rank_map.get(v)
                if si is None or sj is None:
                    continue
                dv = float(si - sj)
                pred = 1 if dv > 0 else 0
                obs = 1 if (float(w) > 0) else 0
                acc_preds.append(pred)
                acc_obs.append(obs)
                used += 1
            except Exception:
                continue
        if used == 0:
            return None
        acc_arr = (np.array(acc_preds) == np.array(acc_obs))
        acc_l = float(np.mean(acc_arr))
        if auto_flip and acc_l < 0.5:
            # Flip orientation and recompute once
            acc_preds = [1 - p for p in acc_preds]
            acc_l = float(np.mean(np.array(acc_preds) == np.array(acc_obs)))
        # If only one class in obs, roc_auc_score is undefined; fall back to 0.5
        try:
            from sklearn.metrics import roc_auc_score  # type: ignore
            if len(set(acc_obs)) < 2:
                auc_l = 0.5
            else:
                auc_l = float(roc_auc_score(acc_obs, acc_preds))
        except Exception:
            auc_l = 0.5
        return acc_l, auc_l, used

    # Pairwise-reversal: for each held-out positive (u->v), create a matched negative (v->u)
    if str(auc_mode).lower() == "pairwise-reversal":
        auc_scores: List[float] = []
        auc_labels: List[int] = []
        acc_preds: List[int] = []
        acc_obs: List[int] = []
        used = 0
        for edge in test_edges:
            try:
                u, v, _w = str(edge[0]), str(edge[1]), edge[2]
                si = rank_map.get(u)
                sj = rank_map.get(v)
                if si is None or sj is None:
                    continue
                # Positive dv and its reverse negative
                dv_pos = float(si - sj)   # label 1
                dv_neg = float(sj - si)   # label 0
                auc_scores.extend([dv_pos, dv_neg])
                auc_labels.extend([1, 0])
                # Accuracy on positives matches legacy correctness
                acc_preds.append(1 if dv_pos > 0 else 0)
                acc_obs.append(1)
                used += 1
            except Exception:
                continue
        if used == 0:
            return None
        acc_arr = (np.array(acc_preds) == np.array(acc_obs))
        if str(acc_mode).lower() == "balanced":
            # In pairwise-reversal, we created an equal number of positives and negatives in auc_{scores,labels}
            # Use threshold at 0 to form predictions across both classes
            acc_all = float(np.mean((np.array(auc_scores) > 0).astype(int) == np.array(auc_labels)))
            acc_p = acc_all
        else:
            acc_p = float(np.mean(acc_arr))
        if auto_flip and acc_p < 0.5:
            # Flip both the positive correctness and swap labels
            acc_preds = [1 - p for p in acc_preds]
            # For symmetric pairs, flipping labels is equivalent to flipping signs of dv; just swap labels
            auc_labels = [1 - l for l in auc_labels]
            acc_p = float(np.mean(np.array(acc_preds) == np.array(acc_obs)))
        try:
            from sklearn.metrics import roc_auc_score  # type: ignore
            auc_p = float(roc_auc_score(auc_labels, auc_scores))
        except Exception:
            # With symmetric pairs, threshold-at-0 approximation is fine
            scores = np.array(auc_scores)
            labels = np.array(auc_labels)
            pos = labels == 1
            neg = ~pos
            tpr = (scores[pos] > 0).mean() if pos.any() else 0.5
            fpr = (scores[neg] > 0).mean() if neg.any() else 0.5
            auc_p = 0.5 * (tpr + (1 - fpr))
        return acc_p, auc_p, used
    # Default (balanced-negatives) mode
    acc_preds2: List[int] = []
    acc_obs2: List[int] = []
    auc_scores2: List[float] = []
    auc_labels2: List[int] = []
    used2 = 0
    # Build quick lookup structures from test set for negative sampling per source
    test_pos_by_u: Dict[str, set] = {}
    losers_pool: List[str] = []
    _hb = 0
    for edge in test_edges:
        try:
            u, v = str(edge[0]), str(edge[1])
        except Exception:
            continue
        test_pos_by_u.setdefault(u, set()).add(v)
        losers_pool.append(v)
    unique_nodes = np.array(sorted(rank_map.keys()))
    unique_nodes_set = set(unique_nodes.tolist())
    rng = np.random.RandomState(seed if seed is not None else 42)
    k_neg = int(max(0, k_neg))
    for edge in test_edges:
        try:
            u, v, _w = str(edge[0]), str(edge[1]), edge[2]
            si = rank_map.get(u)
            sj = rank_map.get(v)
            if si is None or sj is None:
                continue
            dv = float(si - sj)
            acc_preds2.append(1 if dv > 0 else 0)
            acc_obs2.append(1)
            auc_scores2.append(dv)
            auc_labels2.append(1)
            if k_neg > 0 and unique_nodes.size > 1:
                base_cand = unique_nodes_set.difference({u})
                ban = test_pos_by_u.get(u, set())
                cand_set = base_cand.difference(ban)
                if v in cand_set:
                    cand_set.discard(v)
                if not cand_set:
                    cand_set = base_cand.difference({v})
                if cand_set:
                    cand_list = list(cand_set)
                    take = min(k_neg, len(cand_list))
                    if take > 0:
                        if take == len(cand_list):
                            sample_idx = np.arange(take)
                        else:
                            sample_idx = rng.choice(np.arange(len(cand_list)), size=take, replace=False)
                        for idx in np.atleast_1d(sample_idx):
                            v_neg = str(cand_list[int(idx)])
                            sjn = rank_map.get(v_neg)
                            if sjn is None:
                                continue
                            auc_scores2.append(float(si - sjn))
                            auc_labels2.append(0)
            used2 += 1
            _hb += 1
            if _hb % 200000 == 0:
                try:
                    print(f"[auc] processed {used2} held-out edges (+negs)")
                except Exception:
                    pass
        except Exception:
            continue
    if used2 == 0:
        return None
    pos_acc_arr = (np.array(acc_preds2) == np.array(acc_obs2))
    if str(acc_mode).lower() == "balanced" and len(auc_scores2) == len(auc_labels2) and len(auc_labels2) > 0:
        acc_bn = float(np.mean((np.array(auc_scores2) > 0).astype(int) == np.array(auc_labels2)))
    else:
        acc_bn = float(np.mean(pos_acc_arr))
    if auto_flip and acc_bn < 0.5:
        acc_preds2 = [1 - p for p in acc_preds2]
        auc_labels2 = [1 - l for l in auc_labels2]
        if str(acc_mode).lower() == "balanced" and len(auc_scores2) == len(auc_labels2) and len(auc_labels2) > 0:
            acc_bn = float(np.mean((np.array(auc_scores2) > 0).astype(int) == np.array(auc_labels2)))
        else:
            acc_bn = float(np.mean(pos_acc_arr))
    obs_arr = np.array(auc_labels2)
    if len(np.unique(obs_arr)) < 2:
        auc_bn = 0.5
    else:
        try:
            from sklearn.metrics import roc_auc_score  # type: ignore
            auc_bn = float(roc_auc_score(auc_labels2, auc_scores2))
        except Exception:
            # Fallback to a rough approximation using zero threshold on dv
            scores = np.array(auc_scores2)
            pos = obs_arr == 1
            neg = ~pos
            tpr = (scores[pos] > 0).mean() if pos.any() else 0.5
            fpr = (scores[neg] > 0).mean() if neg.any() else 0.5
            auc_bn = 0.5 * (tpr + (1 - fpr))
    return acc_bn, auc_bn, used2

# --- Extra validation helpers ---
def _fit_temperature_beta(diffs: np.ndarray, labels: np.ndarray) -> float:
    # Fit beta > 0 to minimize negative log-likelihood on labels in {0,1}
    diffs = np.asarray(diffs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    grid = np.logspace(-3, 3, 25)
    def nll(beta: float) -> float:
        z = np.clip(beta * diffs, -40, 40)
        p = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-12
        return float(-(labels*np.log(p+eps) + (1-labels)*np.log(1-p+eps)).mean())
    scores = [nll(b) for b in grid]
    best = grid[int(np.argmin(scores))]
    return float(best)

def _logloss_brier_from_diffs(diffs: np.ndarray, labels: np.ndarray, beta: float) -> Tuple[float,float]:
    z = np.clip(beta * np.asarray(diffs, dtype=float), -40, 40)
    p = 1.0 / (1.0 + np.exp(-z))
    eps = 1e-12
    logloss = float(-(labels*np.log(p+eps) + (1-labels)*np.log(1-p+eps)).mean())
    brier = float(((labels - p)**2).mean())
    return logloss, brier

# ------------------------- High-level run ---------------------------------- #

def _write_multi(df: pd.DataFrame, base_path: str, formats: List[str]):
    # Ensure parent directory exists
    try:
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
    except Exception:
        pass
    if 'csv' in formats:
        try:
            tmp = base_path + '.csv.tmp'
            df.to_csv(tmp, index=False)
            try:
                os.replace(tmp, base_path + '.csv')
            except Exception:
                # Fallback to direct write if replace fails
                try:
                    df.to_csv(base_path + '.csv', index=False)
                except PermissionError as pe:
                    # Likely file is open/locked on Windows (e.g., Excel). Write a timestamped fallback.
                    import time
                    ts = time.strftime('%Y%m%d_%H%M%S')
                    alt = f"{base_path}.{ts}.csv"
                    try:
                        df.to_csv(alt, index=False)
                        print(f"[warn] csv locked; wrote fallback: {alt} ({pe})")
                    except Exception as e2:
                        print(f"[warn] csv write failed (fallback also failed): {e2}")
                except Exception as e:
                    print(f"[warn] csv write failed: {e}")
        except Exception as e:
            print(f"[warn] csv write failed: {e}")
    if 'parquet' in formats:
        try:
            df.to_parquet(base_path + '.parquet', index=False)
        except Exception as e:
            print(f"[warn] parquet write failed: {e}")
    if 'json' in formats:
        try:
            tmpj = base_path + '.json.tmp'
            df.to_json(tmpj, orient='records')
            try:
                os.replace(tmpj, base_path + '.json')
            except Exception:
                df.to_json(base_path + '.json', orient='records')
        except Exception as e:
            print(f"[warn] json write failed: {e}")

# Helper to compute baseline AUC/logloss rows from a stats table
def _collect_baseline_rows(stats_df: pd.DataFrame, metrics: List[Tuple[str, int]], test_edges: List[Tuple[str,str,float]], _norm_name2, st: str, group: str, y: int, v_extra: Dict[str, Any]):
    rows_auc: List[List[Any]] = []
    rows_ll: List[List[Any]] = []
    try:
        if stats_df is None or len(stats_df) == 0 or not metrics:
            return rows_auc, rows_ll
        name_col = None
        for c in ('Name','name','player_name','Player','player'):
            if c in stats_df.columns:
                name_col = c
                break
        if name_col is None:
            return rows_auc, rows_ll
        s = stats_df.copy()
        s['k'] = s[name_col].apply(_norm_name2)
        for mcol, direction in metrics:
            try:
                smap = {k: float(v) for k, v in s[['k', mcol]].dropna().itertuples(index=False, name=None)}
            except Exception:
                continue
            scores: List[float] = []
            labels2: List[int] = []
            for (u, v, w) in test_edges:
                ku = _norm_name2(u); kv = _norm_name2(v)
                if ku in smap and kv in smap:
                    diff = (smap[ku] - smap[kv]) * (1.0 if direction > 0 else -1.0)
                    scores.append(float(diff))
                    labels2.append(1 if (w > 0) else 0)
            if scores:
                from sklearn.metrics import roc_auc_score  # type: ignore
                try:
                    auc_b = 0.5 if len(set(labels2)) < 2 else float(roc_auc_score(labels2, scores))
                except Exception:
                    auc_b = 0.5
                acc_b = float(np.mean((np.array(scores) > 0).astype(int) == np.array(labels2)))
                rows_auc.append([st, group, y, mcol, auc_b, acc_b, len(scores)])
                if bool(v_extra.get('statcast_logloss', False)):
                    d = np.array(scores, dtype=float)
                    l = np.array(labels2, dtype=int)
                    # Filter out non-finite diffs to avoid NaNs in logloss/Brier (e.g., ERA+=inf when ERA=0)
                    m = np.isfinite(d)
                    d2 = d[m]
                    l2 = l[m]
                    if d2.size > 0:
                        beta_b = _fit_temperature_beta(d2, l2)
                        ll_b, br_b = _logloss_brier_from_diffs(d2, l2, beta_b)
                        rows_ll.append([st, group, None, y, beta_b, ll_b, br_b, int(d2.size), mcol])
                    else:
                        # No finite pairs left; emit placeholder row to indicate coverage
                        rows_ll.append([st, group, None, y, None, None, None, 0, mcol])
            else:
                # Emit a placeholder row to indicate metric coverage even if no pairs matched
                rows_auc.append([st, group, y, mcol, None, None, 0])
    except Exception:
        # Be fail-safe for baseline collection; don't block the pipeline
        return rows_auc, rows_ll
    return rows_auc, rows_ll

# Helper: orientation fraction on positive (winner->loser) edges
def _rank_orientation_fraction(sorted_r: List[List[Any]], test_edges: List[Tuple[str,str,float]]) -> Tuple[float, int]:
    try:
        rmap = {n: s for n, s in sorted_r}
    except Exception:
        rmap = {}
    if not rmap or not test_edges:
        return 0.0, 0
    ok = 0
    tot = 0
    for (u, v, w) in test_edges:
        if w <= 0:
            continue
        if (u in rmap) and (v in rmap):
            tot += 1
            if rmap[u] > rmap[v]:
                ok += 1
    frac = (float(ok) / float(tot)) if tot else 0.0
    return frac, tot

def _harmonic_mean(a: float, b: float) -> float:
    s = a + b
    if s <= 0:
        return 0.0
    return 2.0 * a * b / s

def _write_R_nk(raw: pd.DataFrame, group: str, year: int, k: float, *, out_dir: str = None):
    """Compute and write n/(n+k) shrinkage weights per player for inspection.

    raw: PA-level raw DataFrame with columns 'batter_name','pitcher_name'
    group: 'batter' or 'pitcher'
    year: season
    k: shrinkage constant
    out_dir: optional override for output folder; defaults to At Bats/<group>_data/aware_scores
    """
    try:
        col = 'batter_name' if group=='batter' else 'pitcher_name'
        c = raw[col].astype(str).str.strip()
        # Robust across pandas versions: set axis name before reset_index
        n = c.value_counts().rename_axis('Player').reset_index(name='n')
        n['R_nk'] = n['n'].astype(float) / (n['n'].astype(float) + float(k))
        out_base = out_dir or os.path.join('At Bats', f'{group}_data','aware_scores')
        os.makedirs(out_base, exist_ok=True)
        path = os.path.join(out_base, f"{year}_R_nk_k{int(k)}.csv")
        n[['Player','R_nk','n']].to_csv(path, index=False)
        try:
            print(f"[k_sweep] R_nk written -> {path} (players={len(n)})")
        except Exception:
            pass
    except Exception:
        pass

def _run_k_sweep_section(cfg: Dict[str, Any], output_dir: str, formats: List[str], progress: bool) -> None:
    try:
        ks_cfg = (cfg.get('analysis', {}) or {}).get('k_sweep', {}) or {}
        if not ks_cfg.get('enabled'):
            return
        bats = ks_cfg.get('batter_k', [100,125,175,200,225])
        pits = ks_cfg.get('pitcher_k', [300,350,450])
        # Optional: write extra R_nk snapshots even if not in sweep
        extra_R = ks_cfg.get('extra_R', {}) or {}
        extra_bats = extra_R.get('batter', []) or []
        extra_pits = extra_R.get('pitcher', []) or []
        years = cfg.get('years', [])
        out_rows: List[List[Any]] = []
        se_done: set[tuple[int,str]] = set()  # track (year, group) for which se_based row was added
        for y in years:
            # Load raw PAs once and compute R_nk snapshots
            try:
                raw_path = os.path.join(cfg['paths']['raw_data_dir'], f"at_bat_data_{y}.csv")
                raw = pd.read_csv(raw_path) if os.path.isfile(raw_path) else None
            except Exception:
                raw = None
            if raw is not None:
                # Main sweep R_nk
                for kb in bats:
                    _write_R_nk(raw, 'batter', y, float(kb))
                for kp in pits:
                    _write_R_nk(raw, 'pitcher', y, float(kp))
                # Extra snapshots requested
                for kb in extra_bats:
                    _write_R_nk(raw, 'batter', y, float(kb))
                for kp in extra_pits:
                    _write_R_nk(raw, 'pitcher', y, float(kp))
                # Strict preflight: ensure all required R_nk files exist before evaluating
                missing = []
                for kb in bats:
                    p = os.path.join('At Bats', 'batter_data','aware_scores', f"{y}_R_nk_k{int(kb)}.csv")
                    if not os.path.isfile(p):
                        missing.append(p)
                for kp in pits:
                    p = os.path.join('At Bats', 'pitcher_data','aware_scores', f"{y}_R_nk_k{int(kp)}.csv")
                    if not os.path.isfile(p):
                        missing.append(p)
                for kb in extra_bats:
                    p = os.path.join('At Bats', 'batter_data','aware_scores', f"{y}_R_nk_k{int(kb)}.csv")
                    if not os.path.isfile(p):
                        missing.append(p)
                for kp in extra_pits:
                    p = os.path.join('At Bats', 'pitcher_data','aware_scores', f"{y}_R_nk_k{int(kp)}.csv")
                    if not os.path.isfile(p):
                        missing.append(p)
                if missing:
                    try:
                        print("[k_sweep][error] required R_nk snapshots missing; aborting sweep. Missing:")
                        for mp in missing:
                            print("  - ", mp)
                    except Exception:
                        pass
                    return
            for kb in bats:
                for kp in pits:
                    for group in ['batter','pitcher']:
                        try:
                            k_val = float(kb) if group=='batter' else float(kp)
                            Rnk_path = os.path.join('At Bats', f'{group}_data','aware_scores', f"{y}_R_nk_k{int(k_val)}.csv")
                            R_df = pd.read_csv(Rnk_path) if os.path.isfile(Rnk_path) else None
                            if R_df is None:
                                # Attempt to generate the missing snapshot on-demand, then retry
                                try:
                                    raw_path2 = os.path.join(cfg['paths']['raw_data_dir'], f"at_bat_data_{y}.csv")
                                    if os.path.isfile(raw_path2):
                                        raw2 = pd.read_csv(raw_path2)
                                        _write_R_nk(raw2, group, y, float(k_val))
                                        # Retry a couple of times in case AV or indexing delays file visibility
                                        import time as _t
                                        for _ in range(3):
                                            if os.path.isfile(Rnk_path):
                                                try:
                                                    R_df = pd.read_csv(Rnk_path)
                                                    break
                                                except Exception:
                                                    _t.sleep(0.1)
                                            _t.sleep(0.1)
                                except Exception as _rnk_e:
                                    try:
                                        print(f"[k_sweep][warn] failed to create missing R_nk snapshot: {Rnk_path} -> {_rnk_e}")
                                    except Exception:
                                        pass
                                if R_df is None:
                                    try:
                                        print(f"[k_sweep][warn] missing R_nk snapshot: {Rnk_path} (evaluating with empty R_map)")
                                    except Exception:
                                        pass
                            R_map = {str(n): float(r) for n, r in (R_df[['Player','R_nk']].itertuples(index=False, name=None) if R_df is not None and not R_df.empty else [])}
                            edge_path = os.path.join('At Bats', f'{group}_data','aware_scores', f"{y}_{group}_edges.csv")
                            if not os.path.isfile(edge_path):
                                continue
                            val_cfg = cfg.get('validation', {})
                            G,A,node_list,train_edges,test_edges = make_graph_from_edge_csv(
                                edge_path,
                                validation_folds=cfg['validation_folds'],
                                seed=val_cfg.get('seed'),
                                sample_as_train=bool(val_cfg.get('sample_as_train', True)),
                                index_base=int(val_cfg.get('index_base', 0)),
                                fast=True
                            )
                            if not node_list or not test_edges:
                                continue
                            # Optional: limit test edges for faster sweep evaluation
                            try:
                                ks_max = int((cfg.get('analysis', {}) or {}).get('k_sweep', {}).get('max_test_edges') or 0)
                            except Exception:
                                ks_max = 0
                            if ks_max and len(test_edges) > ks_max:
                                test_edges = test_edges[:ks_max]
                            try:
                                import scipy.sparse as sp
                                A = sp.csr_matrix(A)
                            except Exception:
                                pass
                            aware_lambda = float(cfg.get('ranking', {}).get('aware_lambda', 1.0))
                            aware_harm = bool(cfg.get('ranking', {}).get('aware_harmonic', True))
                            _, sorted_r = aware_rank_with_tether(A, node_list, R_map, lambda_reg=aware_lambda, use_harmonic=aware_harm)
                            # Compute balanced accuracy (default for sweep) using n/(n+k)
                            # Clamp negatives per positive to at most 1 for sweep speed unless user set 0
                            _kneg = int(val_cfg.get('negatives_per_positive', 1))
                            if _kneg is None:
                                _kneg = 1
                            if _kneg > 1:
                                _kneg = 1
                            # Diagnostics: orientation fraction and rank spread
                            try:
                                orient_frac, orient_tot = _rank_orientation_fraction(sorted_r, list(test_edges))
                            except Exception:
                                orient_frac, orient_tot = 0.0, 0
                            try:
                                rank_std = float(np.std([s for _, s in sorted_r])) if sorted_r else 0.0
                            except Exception:
                                rank_std = 0.0
                            auc_mode_str = str(val_cfg.get('aucMode','balanced-negatives'))
                            res_bal = _compute_acc_auc(
                                sorted_r,
                                list(test_edges),
                                auc_mode=auc_mode_str,
                                k_neg=int(_kneg),
                                auto_flip=bool(val_cfg.get('auto_flip', False)),
                                seed=val_cfg.get('seed'),
                                acc_mode="balanced",
                            )
                            # Compute positive-only accuracy for apples-to-apples with older runs
                            res_pos = _compute_acc_auc(
                                sorted_r,
                                list(test_edges),
                                auc_mode=str(val_cfg.get('aucMode','balanced-negatives')),
                                k_neg=int(_kneg),
                                auto_flip=bool(val_cfg.get('auto_flip', False)),
                                seed=val_cfg.get('seed'),
                                acc_mode="positive-only",
                            )
                            if res_bal:
                                acc_bal, auc, used = res_bal
                                # Use res_pos if available; else None
                                acc_pos = float(res_pos[0]) if res_pos else None
                                out_rows.append([y, int(kb), int(kp), group, 'n_over_n_plus_k', float(acc_pos) if acc_pos is not None else None, float(acc_bal), float(auc), int(used), float(orient_frac), float(rank_std), auc_mode_str])

                            # Add a single se_based reference per year/group
                            key = (y, group)
                            if key not in se_done:
                                try:
                                    Rse_path = os.path.join('At Bats', f'{group}_data','aware_scores', f"{y}_R.csv")
                                    Rse_df = pd.read_csv(Rse_path) if os.path.isfile(Rse_path) else None
                                    Rse_map = {str(n): float(r) for n, r in (Rse_df[['Player','R']].itertuples(index=False, name=None) if Rse_df is not None and not Rse_df.empty else [])}
                                    if Rse_map:
                                        # Recompute ranks using se-based R tether
                                        _, sorted_r_se = aware_rank_with_tether(A, node_list, Rse_map, lambda_reg=aware_lambda, use_harmonic=aware_harm)
                                        # Diagnostics for se_based as well
                                        try:
                                            orient_frac_se, orient_tot_se = _rank_orientation_fraction(sorted_r_se, list(test_edges))
                                        except Exception:
                                            orient_frac_se, orient_tot_se = 0.0, 0
                                        try:
                                            rank_std_se = float(np.std([s for _, s in sorted_r_se])) if sorted_r_se else 0.0
                                        except Exception:
                                            rank_std_se = 0.0
                                        res_bal_se = _compute_acc_auc(
                                            sorted_r_se,
                                            list(test_edges),
                                            auc_mode=auc_mode_str,
                                            k_neg=int(val_cfg.get('negatives_per_positive', 1)),
                                            auto_flip=bool(val_cfg.get('auto_flip', False)),
                                            seed=val_cfg.get('seed'),
                                            acc_mode="balanced",
                                        )
                                        res_pos_se = _compute_acc_auc(
                                            sorted_r_se,
                                            list(test_edges),
                                            auc_mode=str(val_cfg.get('aucMode','balanced-negatives')),
                                            k_neg=int(val_cfg.get('negatives_per_positive', 1)),
                                            auto_flip=bool(val_cfg.get('auto_flip', False)),
                                            seed=val_cfg.get('seed'),
                                            acc_mode="positive-only",
                                        )
                                        if res_bal_se:
                                            acc_bal_se, auc_se, used_se = res_bal_se
                                            acc_pos_se = float(res_pos_se[0]) if res_pos_se else None
                                            # Kb/Kp not applicable for se_based; set None
                                            out_rows.append([y, None, None, group, 'se_based', float(acc_pos_se) if acc_pos_se is not None else None, float(acc_bal_se), float(auc_se), int(used_se), float(orient_frac_se), float(rank_std_se), auc_mode_str])
                                            se_done.add(key)
                                except Exception:
                                    pass
                        except Exception as _ks_e:
                            try:
                                if progress:
                                    print(f"[k_sweep] skip {y}:{group} Kb={kb} Kp={kp} due to: {_ks_e}")
                            except Exception:
                                pass
                            continue
        if out_rows:
            # Columns: add diagnostics OrientFrac and RankStd plus AucMode used
            kdf = pd.DataFrame(out_rows, columns=['Year','Kb','Kp','Group','Shrink','AccuracyPos','Accuracy','AUC','TestEdges','OrientFrac','RankStd','AucMode'])
            _write_multi(kdf, os.path.join(output_dir, 'k_sweep_summary'), formats)
            try:
                if progress:
                    print(f"[pipeline] k_sweep_summary written ({len(out_rows)} rows)")
            except Exception:
                pass
        else:
            try:
                if progress:
                    print("[pipeline] k_sweep produced 0 rows (check R_nk and edges availability)")
            except Exception:
                pass
    except Exception:
        pass

def aware_rank_with_tether(A, node_list: List[str], R_map: Dict[str, float], lambda_reg: float = 1.0, *, use_harmonic: bool = True):
    """Solve (L_W + D_reg) x = b with W_ij scaled by h(R_i,R_j) and D_reg = lambda*diag(1-R).

    W_ij = A_ij * h(R_i, R_j), with h the harmonic mean. b accumulates +w for outgoing and -w for incoming.
    """
    import numpy as np
    try:
        import scipy.sparse as sp  # type: ignore
        from scipy.sparse.linalg import spsolve  # type: ignore
        use_sparse = True
    except Exception:
        use_sparse = False
    n = len(node_list)
    if n == 0:
        return np.zeros((0,), dtype=float), []
    if use_sparse:
        A = sp.csr_matrix(A)
        rows, cols = A.nonzero()
        data = A.data.copy()
        # Optionally scale by harmonic mean of R_i and R_j
        if use_harmonic:
            for k in range(len(data)):
                i = int(rows[k]); j = int(cols[k])
                ri = float(R_map.get(node_list[i], 1.0))
                rj = float(R_map.get(node_list[j], 1.0))
                data[k] = float(data[k]) * _harmonic_mean(ri, rj)
        W = sp.csr_matrix((data, (rows, cols)), shape=A.shape)
        d = np.asarray(W.sum(axis=1)).ravel()
        L = sp.diags(d, offsets=0, shape=(n, n), dtype=float) - W
        b = np.zeros((n,), dtype=float)
        for k in range(len(data)):
            i = int(rows[k]); j = int(cols[k]); w = float(data[k])
            b[i] += w
            b[j] -= w
        R_vec = np.array([float(R_map.get(name, 1.0)) for name in node_list], dtype=float)
        # Always include a tiny numerical ridge epsilon to stabilize the solve, even when lambda_reg>0.
        deg_scale = float(np.mean(d)) if d.size else 1.0
        eps = max(1e-8, 1e-6 * deg_scale)
        if lambda_reg > 0:
            Dreg = sp.diags(lambda_reg * (1.0 - R_vec), offsets=0, shape=(n, n), dtype=float)
            # Add epsilon ridge to avoid exact singular blocks (e.g., nodes with R=1 or isolates)
            Dreg = Dreg + sp.eye(n, format='csr', dtype=float) * eps
        else:
            # Pure epsilon ridge without data-dependent tether
            Dreg = sp.eye(n, format='csr', dtype=float) * eps
        M = L + Dreg
        try:
            x = spsolve(M, b)
        except Exception:
            x = np.linalg.lstsq(M.toarray(), b, rcond=None)[0]
    else:
        A = np.asarray(A, dtype=float)
        n = A.shape[0]
        W = A.copy()
        if use_harmonic:
            for i in range(n):
                for j in range(n):
                    if W[i, j] != 0:
                        ri = float(R_map.get(node_list[i], 1.0))
                        rj = float(R_map.get(node_list[j], 1.0))
                        W[i, j] *= _harmonic_mean(ri, rj)
        d = W.sum(axis=1)
        L = np.diag(d) - W
        b = np.zeros((n,), dtype=float)
        for i in range(n):
            for j in range(n):
                w = W[i, j]
                if w != 0:
                    b[i] += w
                    b[j] -= w
        R_vec = np.array([float(R_map.get(name, 1.0)) for name in node_list], dtype=float)
        if lambda_reg > 0:
            M = L + np.diag(lambda_reg * (1.0 - R_vec))
        else:
            deg_scale = float(np.mean(d)) if d.size else 1.0
            eps = max(1e-8, 1e-6 * deg_scale)
            M = L + np.eye(n) * eps
        x = np.linalg.lstsq(M, b, rcond=None)[0]
    ranks = np.asarray(x, dtype=float)
    sorted_pairs = [[node_list[i], float(ranks[i])] for i in range(n)]
    sorted_pairs.sort(reverse=True, key=lambda t: t[1])
    return ranks, sorted_pairs

def _load_manifest(path: str) -> Dict[str, Any]:
    """Load a caching manifest from JSON if present; otherwise return an empty structure."""
    try:
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data.setdefault('runs', {})
                    return data
    except Exception:
        pass
    return {"runs":{}}

def _save_manifest(path: str, manifest: Dict[str,Any]):
    """Persist the caching manifest to JSON safely."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path,'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return ''

def _file_signature(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{int(st.st_mtime)}:{st.st_size}"
    except FileNotFoundError:
        return ''

def _config_signature(cfg: Dict[str,Any]) -> str:
    # Use stable subset impacting ranking outputs
    subset = {
        'years': cfg.get('years'),
        'score_types': cfg.get('score_types'),
        'pitch_types': cfg.get('pitch_types'),
        'innings': cfg.get('innings'),
        'filters': cfg.get('filters'),
        'ranking': cfg.get('ranking'),
        'analysis': cfg.get('analysis',{}).get('mobility',{}).get('enabled', False)
    }
    data = json.dumps(subset, sort_keys=True)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()[:16]

def run_pipeline(cfg: Dict[str,Any]):
    years = cfg['years']
    force = cfg['scrape']['force']
    skip_scrape = bool(cfg['scrape'].get('skip', False)) if isinstance(cfg.get('scrape'), dict) else False
    force_edges = cfg.get('edges',{}).get('force', False)
    progress = cfg['logging']['progress']
    raw_data_dir = cfg['paths']['raw_data_dir']
    output_dir = cfg['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    dry_run = cfg.get('dry_run', False)
    formats = cfg['output']['formats'] if 'output' in cfg else ['csv']
    # Always disable caching to ensure ranks are regenerated and validation uses the correct data
    caching_enabled = False
    manifest_path = cfg.get('caching', {}).get('manifest', os.path.join(output_dir,'manifest.json'))
    manifest = _load_manifest(manifest_path) if caching_enabled else {"runs":{}}
    manifest.setdefault('signatures', {})
    cfg_sig = _config_signature(cfg)
    manifest['signatures']['last_config'] = cfg_sig
    # Track which baseline stats were actually used during this run
    baseline_metrics_used: Dict[str, set] = {'batter': set(), 'pitcher': set()}
    # Track orientation notes to emit into manifest
    orientation_notes: List[str] = []

    if progress: print(f"[pipeline] Years: {years} (dry_run={dry_run})")
    # If the user requests to run only the k-sweep, do it now and return early
    try:
        ks_cfg0 = (cfg.get('analysis', {}) or {}).get('k_sweep', {}) or {}
        if ks_cfg0.get('enabled') and ks_cfg0.get('only'):
            _run_k_sweep_section(cfg, output_dir, formats, progress)
            return
    except Exception:
        pass
    if not skip_scrape:
        ensure_scraped(years, raw_data_dir, force, progress)
    else:
        if progress: print("[scrape] skip requested in config -> will reuse existing raw at-bat data if present")

    score_types = cfg['score_types']
    pitch_types = cfg.get('pitch_types') or []
    innings = cfg.get('innings') or []
    stand_filter = cfg['filters'].get('stand') if 'filters' in cfg else None
    pthrows_filter = cfg['filters'].get('p_throws') if 'filters' in cfg else None
    if not innings:
        innings = list(range(1,10))

    # 1. Generate edge-only bipartite files if missing
    planned = []
    for y in years:
        for st in score_types:
            planned.append((y, st))
            if not dry_run:
                if st == 'aware':
                    # If skip_scrape, allow reuse of existing aware edges unless force_edges is set
                    b_edge_out = os.path.join('At Bats','batter_data','aware_scores', f"{y}_batter_edges.csv")
                    p_edge_out = os.path.join('At Bats','pitcher_data','aware_scores', f"{y}_pitcher_edges.csv")
                    if skip_scrape and (not force_edges) and os.path.isfile(b_edge_out) and os.path.isfile(p_edge_out):
                        if progress: print(f"[aware] reuse existing aware edges for {y} (skip_scrape=true, force_edges=false)")
                    else:
                        ensure_aware_edges(y, raw_data_dir, alpha_ridge=1.0, progress=progress, force=force_edges, use_shrink=bool(cfg.get('ranking',{}).get('aware_shrink', True)), shrink_mode=str(cfg.get('ranking',{}).get('aware_shrink_mode','se_based')), shrink_k=int(cfg.get('ranking',{}).get('aware_shrink_k',150)), use_covariates=bool(cfg.get('ranking',{}).get('aware_use_covariates', True)), shrink_k_batter=cfg.get('ranking',{}).get('aware_shrink_k_batter'), shrink_k_pitcher=cfg.get('ranking',{}).get('aware_shrink_k_pitcher'), basic_covariates=bool(cfg.get('ranking',{}).get('aware_covariates_basic', False)))
                else:
                    ensure_edge_only(y, st, raw_data_dir, progress, pitch_types=pitch_types, innings=innings, stand_filter=stand_filter, pthrows_filter=pthrows_filter, force=force_edges)
    if dry_run:
        print("[dry-run] Planned edge generation:")
        for y,st in planned:
            print(f"  - {y}:{st}")
        return True

    # 2. Convert bipartite edges to unipartite group edges for each score type
    results_summary = []
    levels_records = []
    metric = cfg.get('processing',{}).get('unipartite_metric','sum')
    for y in years:
        for st in score_types:
            if st == 'handmade':
                edge_file = os.path.join(raw_data_dir, 'handmade', f"{y}_edges_only.csv")
                intermediate_dir = None
                out_batter_dir = os.path.join('At Bats','batter_data','handmade_scores')
                out_pitcher_dir = os.path.join('At Bats','pitcher_data','handmade_scores')
                os.makedirs(out_batter_dir, exist_ok=True)
                os.makedirs(out_pitcher_dir, exist_ok=True)
                b_edge_out = os.path.join(out_batter_dir, f"{y}_batter_edges.csv")
                p_edge_out = os.path.join(out_pitcher_dir, f"{y}_pitcher_edges.csv")
                if os.path.isfile(edge_file) and (force_edges or not (os.path.isfile(b_edge_out) and os.path.isfile(p_edge_out))):
                    if progress: print(f"[unipartite] {y} {st}")
                    to_2_unipartite(edge_file, b_edge_out, p_edge_out, metric=metric, raw_data_dir=raw_data_dir)
            elif st == 'frequency':
                edge_file = os.path.join(raw_data_dir, 'frequency', f"{y}_edges_only.csv")
                inter_dir = os.path.join('At Bats','intermediate_results','frequency')
                os.makedirs(inter_dir, exist_ok=True)
                b_edge_out = os.path.join('At Bats','batter_data','frequency_scores', f"{y}_batter_edges.csv")
                p_edge_out = os.path.join('At Bats','pitcher_data','frequency_scores', f"{y}_pitcher_edges.csv")
                def _empty_edges(fp: str) -> bool:
                    try:
                        if not os.path.isfile(fp):
                            return True
                        dfc = pd.read_csv(fp)
                        return dfc is None or len(dfc) == 0
                    except Exception:
                        return True
                need_regen = force_edges or _empty_edges(b_edge_out) or _empty_edges(p_edge_out)
                if os.path.isfile(edge_file) and need_regen:
                    if progress: print(f"[unipartite] regenerating frequency edges for {y} (reason: {'force' if force_edges else 'empty outputs'})")
                    to_2_unipartite(edge_file, os.path.join(inter_dir,f"{y}_batter_edges.csv"), os.path.join(inter_dir,f"{y}_pitcher_edges.csv"), metric=metric, raw_data_dir=raw_data_dir)
                    to_2_unipartite(edge_file, b_edge_out, p_edge_out, metric=metric, raw_data_dir=raw_data_dir)  # reuse
            elif st == 'aware':
                # Aware already produced unipartite edges; ensure files exist
                b_edge_out = os.path.join('At Bats','batter_data','aware_scores', f"{y}_batter_edges.csv")
                p_edge_out = os.path.join('At Bats','pitcher_data','aware_scores', f"{y}_pitcher_edges.csv")
                if force_edges or not (os.path.isfile(b_edge_out) and os.path.isfile(p_edge_out)):
                    ensure_aware_edges(y, raw_data_dir, alpha_ridge=1.0, progress=progress, force=force_edges, use_shrink=bool(cfg.get('ranking',{}).get('aware_shrink', True)), shrink_mode=str(cfg.get('ranking',{}).get('aware_shrink_mode','se_based')), shrink_k=int(cfg.get('ranking',{}).get('aware_shrink_k',150)), use_covariates=bool(cfg.get('ranking',{}).get('aware_use_covariates', True)), shrink_k_batter=cfg.get('ranking',{}).get('aware_shrink_k_batter'), shrink_k_pitcher=cfg.get('ranking',{}).get('aware_shrink_k_pitcher'), basic_covariates=bool(cfg.get('ranking',{}).get('aware_covariates_basic', False)))
                # Optionally emit R_nk snapshot files for configured ks to aid inspection
                try:
                    rk_b = int(cfg.get('ranking',{}).get('aware_shrink_k_batter', cfg.get('ranking',{}).get('aware_shrink_k',150)))
                    rk_p = int(cfg.get('ranking',{}).get('aware_shrink_k_pitcher', cfg.get('ranking',{}).get('aware_shrink_k',150)))
                    raw_path = os.path.join(raw_data_dir, f"at_bat_data_{y}.csv")
                    if os.path.isfile(raw_path):
                        raw_df = pd.read_csv(raw_path)
                        path_b = os.path.join('At Bats','batter_data','aware_scores', f"{y}_R_nk_k{rk_b}.csv")
                        path_p = os.path.join('At Bats','pitcher_data','aware_scores', f"{y}_R_nk_k{rk_p}.csv")
                        if not os.path.isfile(path_b):
                            _write_R_nk(raw_df, 'batter', y, float(rk_b))
                        if not os.path.isfile(path_p):
                            _write_R_nk(raw_df, 'pitcher', y, float(rk_p))
                except Exception:
                    pass
            elif st == 'pitch_type':
                for pt in (pitch_types or ALLOWED_PITCH_TYPES):
                    edge_file = os.path.join(raw_data_dir, 'pitch_type', pt, f"{y}_edges_only.csv")
                    inter_dir = os.path.join('At Bats','intermediate_results','pitch_type', pt)
                    os.makedirs(inter_dir, exist_ok=True)
                    b_edge_out = os.path.join('At Bats','batter_data','pitchtype_scores', pt, f"{y}_batter_edges.csv")
                    p_edge_out = os.path.join('At Bats','pitcher_data','pitchtype_scores', pt, f"{y}_pitcher_edges.csv")
                    os.makedirs(os.path.dirname(b_edge_out), exist_ok=True)
                    os.makedirs(os.path.dirname(p_edge_out), exist_ok=True)
                    if os.path.isfile(edge_file) and not (os.path.isfile(b_edge_out) and os.path.isfile(p_edge_out)):
                        to_2_unipartite(edge_file, b_edge_out, p_edge_out)
            elif st == 'inning':
                for inn in innings:
                    edge_file = os.path.join(raw_data_dir, 'inning', str(inn), f"{y}_edges_only.csv")
                    b_edge_out = os.path.join('At Bats','batter_data','inning_scores', str(inn), f"{y}_batter_edges.csv")
                    p_edge_out = os.path.join('At Bats','pitcher_data','inning_scores', str(inn), f"{y}_pitcher_edges.csv")
                    os.makedirs(os.path.dirname(b_edge_out), exist_ok=True)
                    os.makedirs(os.path.dirname(p_edge_out), exist_ok=True)
                    if os.path.isfile(edge_file) and not (os.path.isfile(b_edge_out) and os.path.isfile(p_edge_out)):
                        to_2_unipartite(edge_file, b_edge_out, p_edge_out)

    # 3. Ranking computation with caching & validation
    top_n = cfg['ranking']['top_n']
    scale_req = cfg['ranking']['scale_ranks']
    validation_rows = []
    start_global = time.time()
    auc_rows = []  # ScoreType, Group, Condition, Year, Folds, ACC, AUC, TestEdges
    logloss_rows: List[List[Any]] = []  # ScoreType, Group, Condition, Year, Beta, LogLoss, Brier, TestEdges, Source
    baseline_auc_rows: List[List[Any]] = []  # ScoreType, Group, Year, Metric, AUC, Accuracy, TestEdges
    base_ranks: Dict[str, Dict[str, Dict[int, List[List[Any]]]]] = {}
    for st in score_types:
        for group in ['batter','pitcher']:
            for y in years:
                cache_prefix = f"{y}:{st}:{group}"
                if st == 'handmade':
                    edge_path = os.path.join('At Bats', f'{group}_data', 'handmade_scores', f"{y}_{group}_edges.csv")
                elif st == 'frequency':
                    edge_path = os.path.join('At Bats', f'{group}_data', 'frequency_scores', f"{y}_{group}_edges.csv")
                elif st == 'pitch_type':
                    for pt in (pitch_types or ALLOWED_PITCH_TYPES):
                        edge_path = os.path.join('At Bats', f'{group}_data', 'pitchtype_scores', pt, f"{y}_{group}_edges.csv")
                        if not os.path.isfile(edge_path):
                            continue
                        cache_key = cache_prefix+f":{pt}"
                        file_sig = _file_signature(edge_path)
                        if caching_enabled and cache_key in manifest['runs']:
                            prev = manifest['runs'][cache_key]
                            if prev.get('file_sig') == file_sig and prev.get('config_sig') == cfg_sig:
                                if progress: print(f"[cache] skip {cache_key}")
                                continue
                        raw_r, sorted_r = spring_rank(A, node_list)
                        rank_dir = os.path.join(output_dir, st, group, pt)
                        os.makedirs(rank_dir, exist_ok=True)
                        # Resolve MLBAM-coded names in ranks
                        try:
                            rdf = pd.DataFrame(sorted_r, columns=['Player','Rank'])
                            # Create a temporary loser column to use the existing resolver
                            tmp = rdf.rename(columns={'Player':'winner'}).copy()
                            tmp['loser'] = tmp['winner']
                            mapped = _resolve_names_in_edges_df(tmp)
                            rdf2 = mapped[['winner']].rename(columns={'winner':'Player'}).copy()
                            rdf2['Rank'] = rdf['Rank'].values
                        except Exception:
                            rdf2 = pd.DataFrame(sorted_r, columns=['Player','Rank'])
                        _write_multi(rdf2, os.path.join(rank_dir, f"{y}_springrank"), formats)
                        if scale_req:
                            scaled = scale_ranks(A, raw_r)
                            scaled_sorted = [[node_list[i], r] for i,r in enumerate(scaled)]
                            scaled_sorted.sort(reverse=True, key=lambda x: x[1])
                            try:
                                sdf = pd.DataFrame(scaled_sorted, columns=['Player','ScaledRank'])
                                tmp = sdf.rename(columns={'Player':'winner'}).copy(); tmp['loser'] = tmp['winner']
                                mapped = _resolve_names_in_edges_df(tmp)
                                sdf2 = mapped[['winner']].rename(columns={'winner':'Player'}).copy(); sdf2['ScaledRank'] = sdf['ScaledRank'].values
                            except Exception:
                                sdf2 = pd.DataFrame(scaled_sorted, columns=['Player','ScaledRank'])
                            _write_multi(sdf2, os.path.join(rank_dir,f"{y}_springrank_scaled"), formats)
                            levels_records.append([st, group, pt, y, max(scaled)-min(scaled)])
                        validation_rows.append([st, group, pt, y, len(node_list), A.count_nonzero(), float(A.count_nonzero())/(len(node_list)**2 if len(node_list)>0 else 1)])
                        # ACC/AUC on held-out edges (if any)
                        cv = cfg.get('validation_folds', 0)
                        if cv and test_edges is not None:
                            val_cfg = cfg.get('validation', {})
                            res = _compute_acc_auc(
                                sorted_r,
                                test_edges,
                                auc_mode=str(val_cfg.get('aucMode','balanced-negatives')),
                                k_neg=int(val_cfg.get('negatives_per_positive', 1)),
                                auto_flip=bool(val_cfg.get('auto_flip', False)),
                            )
                            if res:
                                acc, auc, used = res
                                auc_rows.append([st, group, pt, y, cv, acc, auc, used])
                        if caching_enabled:
                            manifest['runs'][cache_key] = {"time": time.time()-t0, "nodes": len(node_list), 'file_sig': file_sig, 'config_sig': cfg_sig}
                        results_summary.append(pd.DataFrame(sorted_r, columns=['Player','Rank']).head(top_n).assign(Year=y, Group=group, ScoreType=st, PitchType=pt))
                    continue
                elif st == 'inning':
                    for inn in innings:
                        edge_path = os.path.join('At Bats', f'{group}_data', 'inning_scores', str(inn), f"{y}_{group}_edges.csv")
                        if not os.path.isfile(edge_path):
                            continue
                        cache_key = cache_prefix+f":inn{inn}"
                        file_sig = _file_signature(edge_path)
                        if caching_enabled and cache_key in manifest['runs']:
                            prev = manifest['runs'][cache_key]
                            if prev.get('file_sig') == file_sig and prev.get('config_sig') == cfg_sig:
                                if progress: print(f"[cache] skip {cache_key}")
                                continue
                        t0 = time.time()
                        val_cfg = cfg.get('validation', {})
                        # Provide v_extra in this scope for later checks
                        try:
                            v_extra = val_cfg.get('extra', {}) if isinstance(val_cfg, dict) else {}
                        except Exception:
                            v_extra = {}
                        # Ensure v_extra is available for this branch
                        v_extra = {}
                        try:
                            if isinstance(val_cfg, dict):
                                v_extra = val_cfg.get('extra', {}) or {}
                        except Exception:
                            v_extra = {}
                        # Ensure v_extra exists in this scope
                        try:
                            v_extra = val_cfg.get('extra', {}) if isinstance(val_cfg, dict) else {}
                        except Exception:
                            v_extra = {}
                        # Ensure v_extra is always defined in this branch to avoid UnboundLocalError
                        try:
                            v_extra = val_cfg.get('extra', {}) if isinstance(val_cfg, dict) else {}
                        except Exception:
                            v_extra = {}
                        G,A,node_list,_,test_edges = make_graph_from_edge_csv(
                            edge_path,
                            validation_folds=cfg['validation_folds'],
                            seed=val_cfg.get('seed'),
                            sample_as_train=bool(val_cfg.get('sample_as_train', True)),
                            index_base=int(val_cfg.get('index_base', 0))
                        )
                        if not node_list:
                            if progress: print(f"[ranking] skip empty graph for {st}:{group}:inn{inn}:{y}")
                            continue
                        raw_r, sorted_r = spring_rank(A, node_list)
                        rank_dir = os.path.join(output_dir, st, group, str(inn))
                        os.makedirs(rank_dir, exist_ok=True)
                        try:
                            rdf = pd.DataFrame(sorted_r, columns=['Player','Rank'])
                            tmp = rdf.rename(columns={'Player':'winner'}).copy()
                            tmp['loser'] = tmp['winner']
                            mapped = _resolve_names_in_edges_df(tmp)
                            rdf2 = mapped[['winner']].rename(columns={'winner':'Player'}).copy()
                            rdf2['Rank'] = rdf['Rank'].values
                        except Exception:
                            rdf2 = pd.DataFrame(sorted_r, columns=['Player','Rank'])
                        _write_multi(rdf2, os.path.join(rank_dir, f"{y}_springrank"), formats)
                        if scale_req:
                            scaled = scale_ranks(A, raw_r)
                            scaled_sorted = [[node_list[i], r] for i,r in enumerate(scaled)]
                            scaled_sorted.sort(reverse=True, key=lambda x: x[1])
                            try:
                                sdf = pd.DataFrame(scaled_sorted, columns=['Player','ScaledRank'])
                                tmp = sdf.rename(columns={'Player':'winner'}).copy(); tmp['loser'] = tmp['winner']
                                mapped = _resolve_names_in_edges_df(tmp)
                                sdf2 = mapped[['winner']].rename(columns={'winner':'Player'}).copy(); sdf2['ScaledRank'] = sdf['ScaledRank'].values
                            except Exception:
                                sdf2 = pd.DataFrame(scaled_sorted, columns=['Player','ScaledRank'])
                            _write_multi(sdf2, os.path.join(rank_dir,f"{y}_springrank_scaled"), formats)
                            levels_records.append([st, group, inn, y, max(scaled)-min(scaled)])
                        validation_rows.append([st, group, f"inning_{inn}", y, len(node_list), A.count_nonzero(), float(A.count_nonzero())/(len(node_list)**2 if len(node_list)>0 else 1)])
                        cv = cfg.get('validation_folds', 0)
                        if cv and test_edges is not None:
                            val_cfg = cfg.get('validation', {})
                            res = _compute_acc_auc(
                                sorted_r,
                                test_edges,
                                auc_mode=str(val_cfg.get('aucMode','balanced-negatives')),
                                k_neg=int(val_cfg.get('negatives_per_positive', 1)),
                                auto_flip=bool(val_cfg.get('auto_flip', False)),
                            )
                            if res:
                                acc, auc, used = res
                                auc_rows.append([st, group, f"inning_{inn}", y, cv, acc, auc, used])
                        if caching_enabled:
                            manifest['runs'][cache_key] = {"time": time.time()-t0, "nodes": len(node_list), 'file_sig': file_sig, 'config_sig': cfg_sig}
                        results_summary.append(pd.DataFrame(sorted_r, columns=['Player','Rank']).head(top_n).assign(Year=y, Group=group, ScoreType=st, Inning=inn))
                    continue
                else:
                    if st == 'handmade':
                        edge_path = os.path.join('At Bats', f'{group}_data', 'handmade_scores', f"{y}_{group}_edges.csv")
                    elif st == 'frequency':
                        edge_path = os.path.join('At Bats', f'{group}_data', 'frequency_scores', f"{y}_{group}_edges.csv")
                    elif st == 'aware':
                        edge_path = os.path.join('At Bats', f'{group}_data', 'aware_scores', f"{y}_{group}_edges.csv")
                    else:
                        continue
                if not os.path.isfile(edge_path):
                    continue
                cache_key = cache_prefix
                file_sig = _file_signature(edge_path)
                if False and caching_enabled and cache_key in manifest['runs']:
                    prev = manifest['runs'][cache_key]
                    if prev.get('file_sig') == file_sig and prev.get('config_sig') == cfg_sig:
                        if progress: print(f"[cache] validate-only (using cached ranks) {cache_key}")
                        # Validate-only path: load graph/test edges and existing ranks, map names, compute metrics, and populate base_ranks
                        t0 = time.time()
                        val_cfg = cfg.get('validation', {})
                        # In cached mode we still compute standard 5-fold AUC/logloss so validation_auc includes both normal and opp-block
                        fast_cached_validate = False
                        G,A,node_list,_,test_edges = make_graph_from_edge_csv(
                            edge_path,
                            validation_folds=cfg['validation_folds'],
                            seed=val_cfg.get('seed'),
                            sample_as_train=bool(val_cfg.get('sample_as_train', True)),
                            index_base=int(val_cfg.get('index_base', 0))
                        )
                        # Load existing ranks
                        rank_dir = os.path.join(output_dir, st, group)
                        sorted_r: List[List[Any]] = []
                        loaded_df = None
                        for ext in ('csv','parquet','json'):
                            rp = os.path.join(rank_dir, f"{y}_springrank.{ext}")
                            if os.path.isfile(rp):
                                try:
                                    if ext=='csv':
                                        loaded_df = pd.read_csv(rp)
                                    elif ext=='parquet':
                                        loaded_df = pd.read_parquet(rp)
                                    else:
                                        loaded_df = pd.read_json(rp)
                                except Exception:
                                    loaded_df = None
                                if loaded_df is not None:
                                    break
                        if loaded_df is not None and {'Player','Rank'}.issubset(loaded_df.columns):
                            try:
                                tmp = loaded_df[['Player','Rank']].copy().rename(columns={'Player':'winner'})
                                tmp['loser'] = tmp['winner']
                                mapped = _resolve_names_in_edges_df(tmp)
                                rdf2 = mapped[['winner']].rename(columns={'winner':'Player'})
                                rdf2['Rank'] = pd.to_numeric(loaded_df['Rank'], errors='coerce')
                                # persist remapped ranks
                                try:
                                    os.makedirs(rank_dir, exist_ok=True)
                                    rdf2.to_csv(os.path.join(rank_dir, f"{y}_springrank.csv"), index=False)
                                except Exception:
                                    pass
                                sorted_r = [[str(row['Player']), float(row['Rank'])] for _,row in rdf2.iterrows() if not pd.isna(row['Rank'])]
                            except Exception:
                                sorted_r = [[str(row['Player']), float(row['Rank'])] for _,row in loaded_df.iterrows() if not pd.isna(row['Rank'])]
                        # If we still don't have ranks, compute raw_r and sorted_r quickly
                        if not sorted_r and node_list:
                            raw_r, sorted_r = spring_rank(A, node_list)
                            # write remapped ranks
                            try:
                                rdf = pd.DataFrame(sorted_r, columns=['Player','Rank'])
                                tmp = rdf.rename(columns={'Player':'winner'}).copy(); tmp['loser'] = tmp['winner']
                                mapped = _resolve_names_in_edges_df(tmp)
                                rdf2 = mapped[['winner']].rename(columns={'winner':'Player'}).copy(); rdf2['Rank'] = rdf['Rank'].values
                                os.makedirs(rank_dir, exist_ok=True)
                                rdf2.to_csv(os.path.join(rank_dir, f"{y}_springrank.csv"), index=False)
                            except Exception:
                                pass
                        # Report validation density
                        if node_list is not None:
                            try:
                                dens = float(A.count_nonzero())/(len(node_list)**2 if len(node_list)>0 else 1)
                            except Exception:
                                dens = 0.0
                            validation_rows.append([st, group, None, y, len(node_list), A.count_nonzero(), dens])
                        # ACC/AUC on held-out edges (if any)
                        cv = cfg.get('validation_folds', 0)
                        # Ensure v_extra exists before checking only_baseline_flag
                        try:
                            v_extra  # type: ignore[name-defined]
                        except NameError:
                            try:
                                v_extra = val_cfg.get('extra', {}) if isinstance(val_cfg, dict) else {}
                            except Exception:
                                v_extra = {}
                        only_baseline_flag = bool(v_extra.get('only_baseline', False)) if isinstance(v_extra, dict) else False
                        if (not only_baseline_flag) and cv and test_edges is not None and sorted_r:
                            res = _compute_acc_auc(
                                sorted_r,
                                test_edges,
                                auc_mode=str(val_cfg.get('aucMode','balanced-negatives')),
                                k_neg=int(val_cfg.get('negatives_per_positive', 1)),
                                auto_flip=bool(val_cfg.get('auto_flip', False)),
                            )
                            if res:
                                acc, auc, used = res
                                auc_rows.append([st, group, None, y, cv, acc, auc, used])
                            # Temperature logloss (rank): always compute for CV runs (unless only_baseline is active)
                            try:
                                rmap = {n: s for n,s in sorted_r}
                                diffs = []
                                labels = []
                                for (u,v,w) in test_edges:
                                    if (u in rmap) and (v in rmap):
                                        diffs.append(float(rmap[u] - rmap[v]))
                                        labels.append(1 if (w>0) else 0)
                                if diffs:
                                    d = np.array(diffs, dtype=float)
                                    l = np.array(labels, dtype=int)
                                    beta = _fit_temperature_beta(d, l)
                                    ll, br = _logloss_brier_from_diffs(d, l, beta)
                                    logloss_rows.append([st, group, None, y, beta, ll, br, len(diffs), 'rank'])
                            except Exception:
                                pass
                        # Flush pre-oppblock rows immediately in cached path to ensure they appear even if later steps are long
                        try:
                            if auc_rows:
                                _write_multi(pd.DataFrame(auc_rows, columns=['ScoreType','Group','Condition','Year','Folds','Accuracy','AUC','TestEdges']), os.path.join(output_dir,'validation_auc'), formats)
                            if logloss_rows:
                                _write_multi(pd.DataFrame(logloss_rows, columns=['ScoreType','Group','Condition','Year','Beta','LogLoss','Brier','TestEdges','Source']), os.path.join(output_dir,'validation_logloss'), formats)
                            if baseline_auc_rows:
                                _write_multi(pd.DataFrame(baseline_auc_rows, columns=['ScoreType','Group','Year','Metric','AUC','Accuracy','TestEdges']), os.path.join(output_dir,'validation_baseline_auc'), formats)
                        except Exception:
                            pass
                            # Baseline AUC/logloss if enabled (v_extra defined above)
                            if v_extra and (bool(v_extra.get('baseline_auc', False)) or bool(v_extra.get('statcast_logloss', False))):
                                try:
                                    import importlib as _il
                                    _pb = _il.import_module('pybaseball')
                                except Exception:
                                    _pb = None
                                if _pb is not None:
                                    try:
                                        _df_te = pd.DataFrame(test_edges, columns=['winner','loser','score'])
                                        _df_te = _resolve_names_in_edges_df(_df_te)
                                        test_edges_stats = list(_df_te[['winner','loser','score']].itertuples(index=False, name=None))
                                    except Exception:
                                        test_edges_stats = test_edges
                                    def _norm_name2(s: Any) -> str:
                                        import unicodedata as _ud
                                        t = str(s) if not pd.isna(s) else ''
                                        t = t.strip(); t = _ud.normalize('NFKD', t)
                                        t = ''.join(c for c in t if not _ud.combining(c))
                                        # Normalize "last, first [middle]" to "first [middle] last"
                                        if ',' in t:
                                            try:
                                                last, first = t.split(',', 1)
                                                t = f"{first.strip()} {last.strip()}"
                                            except Exception:
                                                t = t.replace(',', ' ')
                                        return ' '.join(t.split()).lower()
                                    stats_df = None; metrics = []
                                    try:
                                        if group=='batter':
                                            try:
                                                stats_df = _pb.batting_stats(y, qual=0)
                                            except TypeError:
                                                stats_df = _pb.batting_stats(y)
                                            for m, d in [('OPS', +1), ('xwOBA', +1), ('WAR', +1), ('xWAR', +1)]:
                                                if stats_df is not None and m in stats_df.columns: metrics.append((m, d))
                                        else:
                                            try:
                                                stats_df = _pb.pitching_stats(y, qual=0)
                                            except TypeError:
                                                stats_df = _pb.pitching_stats(y)
                                            if stats_df is not None:
                                                if 'WHIP' in stats_df.columns:
                                                    metrics.append(('WHIP', -1))
                                                else:
                                                    if all(c in stats_df.columns for c in ['BB','H','IP']):
                                                        try:
                                                            tmp = stats_df.copy(); tmp['WHIP'] = (tmp['BB'].astype(float)+tmp['H'].astype(float))/tmp['IP'].replace({0: np.nan}).astype(float)
                                                            if not tmp['WHIP'].isna().all(): stats_df = tmp; metrics.append(('WHIP', -1))
                                                        except Exception: pass
                                                # Include xFIP (lower is better) and ERA+ (higher is better) when available
                                                try:
                                                    if 'xFIP' in stats_df.columns:
                                                        metrics.append(('xFIP', -1))
                                                except Exception:
                                                    pass
                                                try:
                                                    if 'ERA+' in stats_df.columns:
                                                        # Coerce and drop infinities to avoid non-finite diffs
                                                        try:
                                                            stats_df['ERA+'] = pd.to_numeric(stats_df['ERA+'], errors='coerce').replace([np.inf, -np.inf], np.nan)
                                                        except Exception:
                                                            pass
                                                        metrics.append(('ERA+', +1))
                                                except Exception:
                                                    pass
                                                # Include FIP if present (lower is better)
                                                try:
                                                    if 'FIP' in stats_df.columns:
                                                        metrics.append(('FIP', -1))
                                                except Exception:
                                                    pass
                                                # Attempt to compute ERA+ if missing and ER/IP available
                                                try:
                                                    if 'ERA+' not in stats_df.columns and all(c in stats_df.columns for c in ['ER','IP']):
                                                        tmp_era = stats_df.copy()
                                                        ip = pd.to_numeric(tmp_era['IP'], errors='coerce')
                                                        er = pd.to_numeric(tmp_era['ER'], errors='coerce')
                                                        # Guard against division by zero
                                                        era_row = (9.0 * er) / ip.replace({0: np.nan})
                                                        lg_era = float((9.0 * er.sum()) / ip.replace({0: np.nan}).sum()) if ip.replace({0: np.nan}).sum() > 0 else np.nan
                                                        if np.isfinite(lg_era):
                                                            tmp_era['ERA+'] = 100.0 * (lg_era / era_row)
                                                            if not tmp_era['ERA+'].isna().all():
                                                                stats_df = tmp_era
                                                                metrics.append(('ERA+', +1))
                                                except Exception:
                                                    pass
                                                if 'xERA' in stats_df.columns:
                                                    metrics.append(('xERA', -1))
                                                elif 'ERA' in stats_df.columns:
                                                    metrics.append(('ERA', -1))
                                                k9_aliases = ['K/9','SO9','SO/9','K9']
                                                k9_found = False
                                                for alias in k9_aliases:
                                                    if alias in stats_df.columns:
                                                        if alias != 'K/9':
                                                            try: stats_df = stats_df.rename(columns={alias: 'K/9'})
                                                            except Exception: pass
                                                        metrics.append(('K/9', +1)); k9_found = True; break
                                                if not k9_found and all(c in stats_df.columns for c in ['SO','IP']):
                                                    try:
                                                        tmp2 = stats_df.copy(); tmp2['K/9'] = (9.0 * tmp2['SO'].astype(float)) / tmp2['IP'].replace({0: np.nan}).astype(float)
                                                        if not tmp2['K/9'].isna().all(): stats_df = tmp2; metrics.append(('K/9', +1))
                                                    except Exception: pass
                                    except Exception:
                                        stats_df = None
                                    if stats_df is not None and not stats_df.empty and metrics:
                                        try:
                                            for mcol, _d in metrics: baseline_metrics_used[group].add(mcol)
                                        except Exception: pass
                                        auc_rows_tmp, ll_rows_tmp = _collect_baseline_rows(stats_df, metrics, test_edges_stats, _norm_name2, st, group, y, v_extra)
                                        baseline_auc_rows.extend(auc_rows_tmp)
                                        logloss_rows.extend(ll_rows_tmp)
                        # Keep base ranks for next-year validation
                        if sorted_r:
                            base_ranks.setdefault(st, {}).setdefault(group, {})[y] = sorted_r
                        # Top summary
                        if sorted_r:
                            try:
                                results_summary.append(pd.DataFrame(sorted_r, columns=['Player','Rank']).head(top_n).assign(Year=y, Group=group, ScoreType=st))
                            except Exception:
                                pass
                        continue
                t0 = time.time()
                val_cfg = cfg.get('validation', {})
                G,A,node_list,_,test_edges = make_graph_from_edge_csv(
                    edge_path,
                    validation_folds=cfg['validation_folds'],
                    seed=val_cfg.get('seed'),
                    sample_as_train=bool(val_cfg.get('sample_as_train', True)),
                    index_base=int(val_cfg.get('index_base', 0))
                )
                if not node_list:
                    if progress: print(f"[ranking] skip empty graph for {st}:{group}:{y}")
                    continue
                # Use custom aware solver with tether and harmonic weights
                aware_lambda = float(cfg.get('ranking', {}).get('aware_lambda', 1.0))
                aware_harm = bool(cfg.get('ranking', {}).get('aware_harmonic', True))
                # Load shrink factors R for this group/year (default 1.0)
                R_map: Dict[str, float] = {}
                try:
                    R_path = os.path.join('At Bats', f'{group}_data','aware_scores', f"{y}_R.csv")
                    if os.path.isfile(R_path):
                        R_df = pd.read_csv(R_path)
                        # Accept either ['Player','R'] or ['winner'->'Player']
                        if 'Player' not in R_df.columns and 'winner' in R_df.columns:
                            R_df = R_df.rename(columns={'winner': 'Player'})
                        if {'Player','R'}.issubset(R_df.columns):
                            if bool(cfg.get('ranking',{}).get('aware_shrink', True)):
                                R_map = {str(n): float(r) for n, r in R_df[['Player','R']].dropna().itertuples(index=False, name=None)}
                            else:
                                R_map = {str(n): 1.0 for n, _r in R_df[['Player','R']].dropna().itertuples(index=False, name=None)}
                except Exception:
                    R_map = {}
                raw_r, sorted_r = aware_rank_with_tether(A, node_list, R_map, lambda_reg=aware_lambda, use_harmonic=aware_harm)
                rank_dir = os.path.join(output_dir, st, group)
                os.makedirs(rank_dir, exist_ok=True)
                try:
                    rdf = pd.DataFrame(sorted_r, columns=['Player','Rank'])
                    tmp = rdf.rename(columns={'Player':'winner'}).copy()
                    tmp['loser'] = tmp['winner']
                    mapped = _resolve_names_in_edges_df(tmp)
                    rdf2 = mapped[['winner']].rename(columns={'winner':'Player'}).copy()
                    rdf2['Rank'] = rdf['Rank'].values
                except Exception:
                    rdf2 = pd.DataFrame(sorted_r, columns=['Player','Rank'])
                _write_multi(rdf2, os.path.join(rank_dir, f"{y}_springrank"), formats)
                # If an old file existed with MLBAM codes, ensure we overwrite the CSV variant
                try:
                    pd.DataFrame(rdf2).to_csv(os.path.join(rank_dir, f"{y}_springrank.csv"), index=False)
                except Exception:
                    pass
                if scale_req:
                    scaled = scale_ranks(A, raw_r)
                    scaled_sorted = [[node_list[i], r] for i,r in enumerate(scaled)]
                    scaled_sorted.sort(reverse=True, key=lambda x: x[1])
                    try:
                        sdf = pd.DataFrame(scaled_sorted, columns=['Player','ScaledRank'])
                        tmp = sdf.rename(columns={'Player':'winner'}).copy(); tmp['loser'] = tmp['winner']
                        mapped = _resolve_names_in_edges_df(tmp)
                        sdf2 = mapped[['winner']].rename(columns={'winner':'Player'}).copy(); sdf2['ScaledRank'] = sdf['ScaledRank'].values
                    except Exception:
                        sdf2 = pd.DataFrame(scaled_sorted, columns=['Player','ScaledRank'])
                    _write_multi(sdf2, os.path.join(rank_dir,f"{y}_springrank_scaled"), formats)
                    try:
                        pd.DataFrame(sdf2).to_csv(os.path.join(rank_dir, f"{y}_springrank_scaled.csv"), index=False)
                    except Exception:
                        pass
                    levels_records.append([st, group, None, y, max(scaled)-min(scaled)])
                validation_rows.append([st, group, None, y, len(node_list), A.count_nonzero(), float(A.count_nonzero())/(len(node_list)**2 if len(node_list)>0 else 1)])
                # Keep base ranks for next-year validation
                base_ranks.setdefault(st, {}).setdefault(group, {})[y] = sorted_r
                cv = cfg.get('validation_folds', 0)
                val_cfg = cfg.get('validation', {})
                only_baseline_flag = bool(val_cfg.get('extra', {}).get('only_baseline', False)) if isinstance(val_cfg, dict) else False
                if (not only_baseline_flag) and cv and test_edges is not None:
                    # Refit ranks on the train folds for this CV evaluation, rather than using full-graph ranks
                    val_cfg = cfg.get('validation', {})
                    try:
                        # make_graph_from_edge_csv returned train_edges/test_edges; rebuild A_train and compute ranks
                        Gt,At,nodes_t,train_edges_cv,_ = make_graph_from_edge_csv(
                            edge_path,
                            validation_folds=cfg['validation_folds'],
                            seed=val_cfg.get('seed'),
                            sample_as_train=True,
                            index_base=int(val_cfg.get('index_base', 0))
                        )
                        if nodes_t:
                            try:
                                import scipy.sparse as sp
                                At = sp.csr_matrix(At)
                            except Exception:
                                pass
                            if st == 'aware':
                                aware_lambda = float(cfg.get('ranking', {}).get('aware_lambda', 1.0))
                                aware_harm = bool(cfg.get('ranking', {}).get('aware_harmonic', True))
                                # Load R for aware
                                R_map_cv = {}
                                try:
                                    R_path = os.path.join('At Bats', f'{group}_data','aware_scores', f"{y}_R.csv")
                                    if os.path.isfile(R_path):
                                        R_df = pd.read_csv(R_path)
                                        if 'Player' not in R_df.columns and 'winner' in R_df.columns:
                                            R_df = R_df.rename(columns={'winner': 'Player'})
                                        if {'Player','R'}.issubset(R_df.columns):
                                            R_map_cv = {str(n): float(r) for n, r in R_df[['Player','R']].dropna().itertuples(index=False, name=None)}
                                except Exception:
                                    R_map_cv = {}
                                _, sorted_r_cv = aware_rank_with_tether(At, nodes_t, R_map_cv, lambda_reg=aware_lambda, use_harmonic=aware_harm)
                            else:
                                _, sorted_r_cv = spring_rank(At, nodes_t)
                        else:
                            sorted_r_cv = sorted_r
                    except Exception:
                        sorted_r_cv = sorted_r
                    res = _compute_acc_auc(
                        sorted_r_cv,
                        test_edges,
                        auc_mode=str(val_cfg.get('aucMode','balanced-negatives')),
                        k_neg=int(val_cfg.get('negatives_per_positive', 1)),
                        auto_flip=bool(val_cfg.get('auto_flip', False)),
                        seed=val_cfg.get('seed'),
                    )
                    if res:
                        acc, auc, used = res
                        auc_rows.append([st, group, None, y, cv, acc, auc, used])
                        # Extra: temperature log-loss and Brier score on the same held-out pairs
                        v_extra = val_cfg.get('extra', {}) if isinstance(val_cfg, dict) else {}
                        # Orientation check on the same held-out edges
                        try:
                            frac, tot = _rank_orientation_fraction(sorted_r, list(test_edges))
                            ori = 'as-is' if frac >= 0.5 else 'flipped-at-eval'
                            orientation_notes.append(f"{st}:{group}:{y}: CV orientation={ori} (p={frac:.3f}, Npos={tot})")
                        except Exception:
                            pass
                        # Temperature logloss (rank): always compute for CV runs (unless only_baseline is active)
                        rmap = {n: s for n,s in sorted_r}
                        diffs = []
                        labels = []
                        for (u,v,w) in test_edges:
                            if (u in rmap) and (v in rmap):
                                diffs.append(float(rmap[u] - rmap[v]))
                                labels.append(1 if (w>0) else 0)
                        if diffs:
                            import numpy as _np
                            d = _np.array(diffs, dtype=float)
                            l = _np.array(labels, dtype=int)
                            beta = _fit_temperature_beta(d, l)
                            ll, br = _logloss_brier_from_diffs(d, l, beta)
                            logloss_rows.append([st, group, None, y, beta, ll, br, len(diffs), 'rank'])
                        if v_extra and (bool(v_extra.get('baseline_auc', False)) or bool(v_extra.get('statcast_logloss', False))):
                            # Attempt to load baseline stats via pybaseball
                            try:
                                import importlib as _il
                                _pb = _il.import_module('pybaseball')
                            except Exception:
                                _pb = None
                            if _pb is not None:
                                # Map MLBAM-coded names in held-out edges to real names for stats matching
                                try:
                                    _df_te = pd.DataFrame(test_edges, columns=['winner','loser','score'])
                                    _df_te = _resolve_names_in_edges_df(_df_te)
                                    test_edges_stats = list(_df_te[['winner','loser','score']].itertuples(index=False, name=None))
                                except Exception:
                                    test_edges_stats = test_edges
                                def _norm_name2(s: Any) -> str:
                                    import unicodedata as _ud
                                    t = str(s) if not pd.isna(s) else ''
                                    t = t.strip(); t = _ud.normalize('NFKD', t)
                                    t = ''.join(c for c in t if not _ud.combining(c))
                                    # Normalize "last, first [middle]" to "first [middle] last"
                                    if ',' in t:
                                        try:
                                            last, first = t.split(',', 1)
                                            t = f"{first.strip()} {last.strip()}"
                                        except Exception:
                                            t = t.replace(',', ' ')
                                    # Remove common suffixes and middle initials
                                    toks = [x for x in t.replace('.', ' ').split() if x]
                                    suffixes = {'jr','sr','ii','iii','iv','v'}
                                    toks = [x for x in toks if x.lower() not in suffixes and len(x) > 1]
                                    return ' '.join(toks).lower()
                                stats_df = None
                                metrics = []  # (col, direction)
                                try:
                                    if group=='batter':
                                        try:
                                            stats_df = _pb.batting_stats(y, qual=0)
                                        except TypeError:
                                            stats_df = _pb.batting_stats(y)
                                        for m, d in [('OPS', +1), ('xwOBA', +1), ('WAR', +1), ('xWAR', +1)]:
                                            if stats_df is not None and m in stats_df.columns:
                                                metrics.append((m, d))
                                    else:
                                        try:
                                            stats_df = _pb.pitching_stats(y, qual=0)
                                        except TypeError:
                                            stats_df = _pb.pitching_stats(y)
                                        # Build pitcher metrics list with robust alias handling
                                        # WHIP may be missing; compute from (BB + H) / IP if needed
                                        if stats_df is not None:
                                            if 'WHIP' in stats_df.columns:
                                                metrics.append(('WHIP', -1))
                                            else:
                                                # Attempt to compute WHIP if components exist
                                                if all(c in stats_df.columns for c in ['BB','H','IP']):
                                                    try:
                                                        tmp = stats_df.copy()
                                                        # Guard against division by zero
                                                        tmp['WHIP'] = (tmp['BB'].astype(float) + tmp['H'].astype(float)) / tmp['IP'].replace({0: np.nan}).astype(float)
                                                        if not tmp['WHIP'].isna().all():
                                                            stats_df = tmp
                                                            metrics.append(('WHIP', -1))
                                                    except Exception:
                                                        pass
                                            # Include xFIP (lower is better) and ERA+ (higher is better) when available
                                            try:
                                                if 'xFIP' in stats_df.columns:
                                                    metrics.append(('xFIP', -1))
                                            except Exception:
                                                pass
                                            try:
                                                if 'ERA+' in stats_df.columns:
                                                    try:
                                                        stats_df['ERA+'] = pd.to_numeric(stats_df['ERA+'], errors='coerce').replace([np.inf, -np.inf], np.nan)
                                                    except Exception:
                                                        pass
                                                    metrics.append(('ERA+', +1))
                                            except Exception:
                                                pass
                                            # Include FIP if present (lower is better)
                                            try:
                                                if 'FIP' in stats_df.columns:
                                                    metrics.append(('FIP', -1))
                                            except Exception:
                                                pass
                                            # Attempt to compute ERA+ if missing and ER/IP available
                                            try:
                                                if 'ERA+' not in stats_df.columns and all(c in stats_df.columns for c in ['ER','IP']):
                                                    tmp_era = stats_df.copy()
                                                    ip = pd.to_numeric(tmp_era['IP'], errors='coerce')
                                                    er = pd.to_numeric(tmp_era['ER'], errors='coerce')
                                                    era_row = (9.0 * er) / ip.replace({0: np.nan})
                                                    lg_era = float((9.0 * er.sum()) / ip.replace({0: np.nan}).sum()) if ip.replace({0: np.nan}).sum() > 0 else np.nan
                                                    if np.isfinite(lg_era):
                                                        tmp_era['ERA+'] = 100.0 * (lg_era / era_row)
                                                        try:
                                                            tmp_era['ERA+'] = pd.to_numeric(tmp_era['ERA+'], errors='coerce').replace([np.inf, -np.inf], np.nan)
                                                        except Exception:
                                                            pass
                                                        # Remove non-finite ERA+ values
                                                        try:
                                                            tmp_era['ERA+'] = pd.to_numeric(tmp_era['ERA+'], errors='coerce').replace([np.inf, -np.inf], np.nan)
                                                        except Exception:
                                                            pass
                                                        if not tmp_era['ERA+'].isna().all():
                                                            stats_df = tmp_era
                                                            metrics.append(('ERA+', +1))
                                            except Exception:
                                                pass
                                            # xERA is not always present in season summaries; include if available
                                            # If xERA is missing but ERA exists, include ERA as a fallback (and report as 'ERA')
                                            if 'xERA' in stats_df.columns:
                                                metrics.append(('xERA', -1))
                                            elif 'ERA' in stats_df.columns:
                                                metrics.append(('ERA', -1))
                                            # K/9 may appear under different names across data sources
                                            k9_aliases = ['K/9','SO9','SO/9','K9']
                                            k9_found = False
                                            for alias in k9_aliases:
                                                if alias in stats_df.columns:
                                                    # Normalize reported metric name to 'K/9' in outputs for consistency
                                                    if alias != 'K/9':
                                                        try:
                                                            stats_df = stats_df.rename(columns={alias: 'K/9'})
                                                        except Exception:
                                                            pass
                                                    metrics.append(('K/9', +1))
                                                    k9_found = True
                                                    break
                                            # If K/9 still not present but we have SO and IP, compute it
                                            if not k9_found and all(c in stats_df.columns for c in ['SO','IP']):
                                                try:
                                                    tmp2 = stats_df.copy()
                                                    tmp2['K/9'] = (9.0 * tmp2['SO'].astype(float)) / tmp2['IP'].replace({0: np.nan}).astype(float)
                                                    if not tmp2['K/9'].isna().all():
                                                        stats_df = tmp2
                                                        metrics.append(('K/9', +1))
                                                except Exception:
                                                    pass
                                except Exception:
                                    stats_df = None
                                if stats_df is not None and not stats_df.empty and metrics:
                                    # Record which metrics were actually used
                                    try:
                                        for mcol, _d in metrics:
                                            baseline_metrics_used[group].add(mcol)
                                    except Exception:
                                        pass
                                    auc_rows_tmp, ll_rows_tmp = _collect_baseline_rows(stats_df, metrics, test_edges_stats, _norm_name2, st, group, y, v_extra)
                                    baseline_auc_rows.extend(auc_rows_tmp)
                                    logloss_rows.extend(ll_rows_tmp)
                if caching_enabled:
                    manifest['runs'][cache_key] = {"time": time.time()-t0, "nodes": len(node_list), 'file_sig': file_sig, 'config_sig': cfg_sig}
                results_summary.append(pd.DataFrame(sorted_r, columns=['Player','Rank']).head(top_n).assign(Year=y, Group=group, ScoreType=st))

    # Flush core validation outputs early so normal rows exist even if later steps are long/interrupted
    try:
        # Optionally suppress standard AUC/logloss when only_baseline is requested
        val_cfg_tmp = cfg.get('validation', {})
        extra_tmp = val_cfg_tmp.get('extra', {}) if isinstance(val_cfg_tmp, dict) else {}
        only_baseline = bool(extra_tmp.get('only_baseline', False))
        if (not only_baseline) and auc_rows:
            auc_df = pd.DataFrame(auc_rows, columns=['ScoreType','Group','Condition','Year','Folds','Accuracy','AUC','TestEdges'])
            _write_multi(auc_df, os.path.join(output_dir,'validation_auc'), formats)
            if progress: print('[pipeline] validation_auc (pre-oppblock) written')
        if (not only_baseline) and logloss_rows:
            ll_df = pd.DataFrame(logloss_rows, columns=['ScoreType','Group','Condition','Year','Beta','LogLoss','Brier','TestEdges','Source'])
            _write_multi(ll_df, os.path.join(output_dir,'validation_logloss'), formats)
            if progress: print('[pipeline] validation_logloss (pre-oppblock) written')
        if baseline_auc_rows:
            ba_df = pd.DataFrame(baseline_auc_rows, columns=['ScoreType','Group','Year','Metric','AUC','Accuracy','TestEdges'])
            _write_multi(ba_df, os.path.join(output_dir,'validation_baseline_auc'), formats)
            if progress: print('[pipeline] validation_baseline_auc (pre-oppblock) written')
    except Exception:
        pass

    # Opponent-blockout cross-validation: split by opponents during unipartite creation
    oppblock_accums: Dict[Tuple[str,str,int], List[Tuple[float,float,int]]] = {}
    try:
        val_cfg_top = cfg.get('validation', {})
        extra_val = val_cfg_top.get('extra', {}) if isinstance(val_cfg_top, dict) else {}
        if extra_val.get('only_baseline', False):
            raise Exception('skip_oppblock_due_to_only_baseline')
        if extra_val and bool(extra_val.get('opponent_blockout', False)):
            cv = cfg.get('validation_folds', 0)
            if cv and cv > 0:
                rng = np.random.RandomState(val_cfg_top.get('seed')) if val_cfg_top.get('seed') is not None else np.random
                metric = cfg.get('processing',{}).get('unipartite_metric','sum')
                for st in [s for s in score_types if s in ('handmade','frequency','aware')]:
                    for y in years:
                        # Load edges for opp-block CV; for aware use precomputed unipartite edges to avoid heavy recompute
                        if st in ('handmade','frequency'):
                            edge_only_path = os.path.join(raw_data_dir, st if st!='frequency' else 'frequency', f"{y}_edges_only.csv")
                            if not os.path.isfile(edge_only_path):
                                if progress: print(f"[oppblock] missing {st} edge-only path: {edge_only_path}")
                                continue
                            bpdf = pd.read_csv(edge_only_path)
                            if 'who_won' not in bpdf.columns:
                                if progress: print(f"[oppblock] {st}:{y} missing who_won column")
                                continue
                        else:  # aware
                            try:
                                # Use unipartite aware edges and treat them as 'who_won' per group
                                # We'll iterate group in the next loop and select the proper file
                                bpdf = None  # placeholder; per-group load below
                            except Exception:
                                bpdf = None
                        for group in ['batter','pitcher']:
                            if st == 'aware':
                                edge_path = os.path.join('At Bats', f'{group}_data','aware_scores', f"{y}_{group}_edges.csv")
                                if not os.path.isfile(edge_path):
                                    if progress: print(f"[oppblock] missing aware unipartite edges for {group}:{y}: {edge_path}")
                                    continue
                                sub = pd.read_csv(edge_path)
                                needed = {'winner','loser','score'}
                                if not needed.issubset(sub.columns):
                                    if progress: print(f"[oppblock] aware edges for {group}:{y} missing columns {needed}")
                                    continue
                                # Normalize columns
                                sub = sub[['winner','loser','score']].copy()
                            else:
                                sel = 'batter' if group=='batter' else 'pitcher'
                                sub = bpdf[bpdf['who_won']==sel][['winner','loser','score']].copy()
                            if sub.empty:
                                if progress: print(f"[oppblock] {st}:{group}:{y}: no rows after selection")
                                continue
                            # Opponents are in loser column for this group's wins
                            opps = sorted(sub['loser'].dropna().unique().tolist())
                            if len(opps) < cv:
                                if progress: print(f"[oppblock] {st}:{group}:{y}: insufficient unique opponents (n={len(opps)}) for cv={cv}")
                                continue
                            # Create folds over opponents
                            rng.shuffle(opps)
                            fold_sizes = [len(opps)//cv + (1 if i < (len(opps) % cv) else 0) for i in range(cv)]
                            idx = 0
                            folds = []
                            for fs in fold_sizes:
                                folds.append(opps[idx:idx+fs])
                                idx += fs
                            if progress: print(f"[oppblock] {st}:{group}:{y}: opps={len(opps)} fold_sizes={fold_sizes}")
                            for k in range(cv):
                                held = set(folds[k])
                                train_sub = sub[~sub['loser'].isin(held)]
                                test_sub = sub[sub['loser'].isin(held)]
                                if train_sub.empty or test_sub.empty:
                                    if progress: print(f"[oppblock] {st}:{group}:{y}: fold {k} skipped (train_empty={train_sub.empty}, test_empty={test_sub.empty})")
                                    continue
                                # Train ranks
                                Gt = nx.DiGraph()
                                try:
                                    Gt.add_weighted_edges_from(train_sub[['winner','loser','score']].itertuples(index=False, name=None))
                                except Exception:
                                    continue
                                node_list_t = list(Gt.nodes())
                                try:
                                    At = nx.to_scipy_sparse_matrix(Gt, dtype=float, nodelist=node_list_t)
                                except AttributeError:
                                    At = nx.to_scipy_sparse_array(Gt, dtype=float, nodelist=node_list_t)
                                try:
                                    import scipy.sparse as sp
                                    At = sp.csr_matrix(At)
                                except Exception:
                                    pass
                                # Use aware solver for aware score_type; else spring_rank
                                if st == 'aware':
                                    aware_lambda = float(cfg.get('ranking', {}).get('aware_lambda', 1.0))
                                    aware_harm = bool(cfg.get('ranking', {}).get('aware_harmonic', True))
                                    # Load R map for this year/group
                                    R_map_cv: Dict[str, float] = {}
                                    try:
                                        R_path = os.path.join('At Bats', f'{group}_data','aware_scores', f"{y}_R.csv")
                                        if os.path.isfile(R_path):
                                            R_df = pd.read_csv(R_path)
                                            if 'Player' not in R_df.columns and 'winner' in R_df.columns:
                                                R_df = R_df.rename(columns={'winner': 'Player'})
                                            if {'Player','R'}.issubset(R_df.columns):
                                                R_map_cv = {str(n): float(r) for n, r in R_df[['Player','R']].dropna().itertuples(index=False, name=None)}
                                    except Exception:
                                        R_map_cv = {}
                                    _, sorted_rt = aware_rank_with_tether(At, node_list_t, R_map_cv, lambda_reg=aware_lambda, use_harmonic=aware_harm)
                                else:
                                    _, sorted_rt = spring_rank(At, node_list_t)
                                # Build test edges
                                test_edges = list(test_sub[['winner','loser','score']].itertuples(index=False, name=None))
                                res = _compute_acc_auc(
                                    sorted_rt,
                                    test_edges,
                                    auc_mode=str(val_cfg_top.get('aucMode','balanced-negatives')),
                                    k_neg=int(val_cfg_top.get('negatives_per_positive', 1)),
                                    auto_flip=bool(val_cfg_top.get('auto_flip', False)),
                                )
                                if res:
                                    acc, auc, used = res
                                    oppblock_accums.setdefault((st, group, y), []).append((acc, auc, used))
                                    # Orientation on held-out opponents fold
                                    try:
                                        frac, tot = _rank_orientation_fraction(sorted_rt, list(test_edges))
                                        ori = 'as-is' if frac >= 0.5 else 'flipped-at-eval'
                                        orientation_notes.append(f"{st}:{group}:{y}: OppBlock fold orientation={ori} (p={frac:.3f}, Npos={tot})")
                                    except Exception:
                                        pass
    except Exception:
        pass

    # After collecting per-fold results, aggregate to a single row per (scoreType, group, year)
    try:
        if oppblock_accums:
            cv = cfg.get('validation_folds', 0)
            for (st, group, y), vals in oppblock_accums.items():
                try:
                    acc_mean = float(np.mean([a for (a, _b, _u) in vals]))
                    auc_mean = float(np.mean([b for (_a, b, _u) in vals]))
                    used_sum = int(np.sum([u for (_a, _b, u) in vals]))
                except Exception:
                    # Fallbacks if numpy unavailable for some reason
                    accs = [a for (a, _b, _u) in vals]
                    aucs = [b for (_a, b, _u) in vals]
                    useds = [u for (_a, _b, u) in vals]
                    acc_mean = sum(accs)/len(accs) if accs else 0.0
                    auc_mean = sum(aucs)/len(aucs) if aucs else 0.5
                    used_sum = sum(useds)
                auc_rows.append([st, group, 'oppblock', y, cv, acc_mean, auc_mean, used_sum])
            try:
                if progress:
                    tot_folds = sum(len(v) for v in oppblock_accums.values())
                    print(f"[pipeline] opponent-blockout aggregated rows: {len(oppblock_accums)} groups; folds={tot_folds}")
            except Exception:
                pass
    except Exception:
        pass

    if results_summary:
        summary_df = pd.concat(results_summary, ignore_index=True)
        _write_multi(summary_df, os.path.join(output_dir, 'summary_top_players'), formats)
        if progress: print("[pipeline] summary_top_players written")
    if cfg['ranking']['output_levels'] and levels_records:
        levels_df = pd.DataFrame(levels_records, columns=['ScoreType','Group','Condition','Year','LevelsRange'])
        _write_multi(levels_df, os.path.join(output_dir,'levels_by_year'), formats)
        if progress: print("[pipeline] levels_by_year written")
    if 'validation_rows' in locals() and validation_rows:
        val_df = pd.DataFrame(validation_rows, columns=['ScoreType','Group','Condition','Year','Nodes','Edges','Density'])
        _write_multi(val_df, os.path.join(output_dir,'validation_report'), formats)
        if progress: print("[pipeline] validation_report written")
    # Cross-validated ACC/AUC report (suppressed if only_baseline)
    _valcfg = cfg.get('validation', {})
    _extra = _valcfg.get('extra', {}) if isinstance(_valcfg, dict) else {}
    _only_baseline = bool(_extra.get('only_baseline', False))
    if (not _only_baseline) and auc_rows:
        auc_df = pd.DataFrame(auc_rows, columns=['ScoreType','Group','Condition','Year','Folds','Accuracy','AUC','TestEdges'])
        _write_multi(auc_df, os.path.join(output_dir,'validation_auc'), formats)
        if progress: print('[pipeline] validation_auc written')
    if (not _only_baseline) and logloss_rows:
        ll_df = pd.DataFrame(logloss_rows, columns=['ScoreType','Group','Condition','Year','Beta','LogLoss','Brier','TestEdges','Source'])
        _write_multi(ll_df, os.path.join(output_dir,'validation_logloss'), formats)
        if progress: print('[pipeline] validation_logloss written')
    if baseline_auc_rows:
        ba_df = pd.DataFrame(baseline_auc_rows, columns=['ScoreType','Group','Year','Metric','AUC','Accuracy','TestEdges'])
        _write_multi(ba_df, os.path.join(output_dir,'validation_baseline_auc'), formats)
        if progress: print('[pipeline] validation_baseline_auc written')
    # Historical accuracy tracker: append a one-line summary per (st,group,year,condition)
    try:
        # Build a compact summary from auc_rows
        if auc_rows:
            hist_cols = ['Timestamp','ConfigSig','ScoreType','Group','Year','Condition','Folds','Accuracy','AUC','TestEdges']
            # Aggregate by last entry for each key
            try:
                auc_df2 = pd.DataFrame(auc_rows, columns=['ScoreType','Group','Condition','Year','Folds','Accuracy','AUC','TestEdges'])
                auc_df2['Timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
                auc_df2['ConfigSig'] = cfg_sig
                # Reorder columns
                hist_df = auc_df2[['Timestamp','ConfigSig','ScoreType','Group','Year','Condition','Folds','Accuracy','AUC','TestEdges']]
                # Prefer outputs_aware_full for historical accuracy when 'aware' is part of the run
                try:
                    hist_dir = output_dir
                    if 'aware' in score_types:
                        base_dir = os.path.dirname(output_dir) or '.'
                        hist_dir = os.path.join(base_dir, 'outputs_aware_full')
                        os.makedirs(hist_dir, exist_ok=True)
                except Exception:
                    hist_dir = output_dir
                hist_path = os.path.join(hist_dir, 'historical_accuracy.csv')
                # Append or create
                if os.path.isfile(hist_path):
                    try:
                        prev = pd.read_csv(hist_path)
                        hist_df = pd.concat([prev, hist_df], ignore_index=True)
                    except Exception:
                        pass
                # Deduplicate identical rows to prevent runaway growth
                try:
                    hist_df = hist_df.drop_duplicates()
                except Exception:
                    pass
                hist_df.to_csv(hist_path, index=False)
                if progress: print('[pipeline] historical_accuracy appended')
            except Exception:
                pass
    except Exception:
        pass
    # Year-over-year rank correlation analysis (optional)
    if cfg.get('analysis',{}).get('rank_correlation',{}).get('enabled'):
        rc_rows = []
        def load_ranks_df(base_dir: str, years: List[int]) -> Dict[int,pd.DataFrame]:
            out = {}
            for y in years:
                for ext in ('csv','parquet','json'):
                    path = os.path.join(base_dir, f"{y}_springrank.{ext}")
                    if os.path.isfile(path):
                        try:
                            if ext=='csv':
                                out[y] = pd.read_csv(path)
                            elif ext=='parquet':
                                out[y] = pd.read_parquet(path)
                            else:
                                out[y] = pd.read_json(path)
                        except Exception:
                            pass
                        break
            return out
        pairs_mode = cfg.get('analysis',{}).get('rank_correlation',{}).get('pairs','adjacent')
        for st in score_types:
            if st in ('handmade','frequency'):
                groups = ['batter','pitcher']
                conditions = [None]
                base_tpl = lambda g: os.path.join(output_dir, st, g)
            elif st == 'pitch_type':
                groups = ['batter','pitcher']
                conditions = list(pitch_types or ALLOWED_PITCH_TYPES)
                base_tpl = lambda g, c=None: os.path.join(output_dir, st, g, c or '')
            elif st == 'inning':
                groups = ['batter','pitcher']
                conditions = innings
                base_tpl = lambda g, c=None: os.path.join(output_dir, st, g, str(c))
            else:
                continue
            for g in groups:
                for cond in conditions:
                    base_dir = base_tpl(g) if st in ('handmade','frequency') else base_tpl(g, cond)
                    yr_map = load_ranks_df(base_dir, years)
                    ylist = sorted(yr_map.keys())
                    if len(ylist) < 2:
                        continue
                    candidate_pairs = []
                    if pairs_mode == 'adjacent':
                        candidate_pairs = list(zip(ylist[:-1], ylist[1:]))
                    else:  # all pairs
                        for i in range(len(ylist)):
                            for j in range(i+1, len(ylist)):
                                candidate_pairs.append((ylist[i], ylist[j]))
                    for y0, y1 in candidate_pairs:
                        df0 = yr_map[y0][['Player','Rank']].rename(columns={'Rank':'Rank0'})
                        df1 = yr_map[y1][['Player','Rank']].rename(columns={'Rank':'Rank1'})
                        m = df0.merge(df1, on='Player')
                        if m.empty:
                            continue
                        try:
                            # Spearman and Pearson
                            rho = m['Rank0'].rank(method='average').corr(m['Rank1'].rank(method='average'))
                            pear = m['Rank0'].corr(m['Rank1'])
                            rc_rows.append([st, g, cond, y0, y1, len(m), float(rho) if rho is not None else None, float(pear) if pear is not None else None])
                        except Exception:
                            continue
        if rc_rows:
            rc_df = pd.DataFrame(rc_rows, columns=['ScoreType','Group','Condition','Year0','Year1','Players','Spearman','Pearson'])
            _write_multi(rc_df, os.path.join(output_dir,'rank_correlation'), formats)
            if progress: print('[pipeline] rank_correlation written')
        # OPS correlation (optional, batters only)
        if cfg.get('analysis',{}).get('ops_correlation',{}).get('enabled'):
            ops_rows = []
            def _norm_name(s: Any) -> str:
                import unicodedata as _ud
                t = str(s) if not pd.isna(s) else ''
                t = t.strip()
                t = _ud.normalize('NFKD', t)
                t = ''.join(c for c in t if not _ud.combining(c))
                return ' '.join(t.split()).lower()
            for st in score_types:
                # Allow all score_types including 'aware'
                base_dir = os.path.join(output_dir, st, 'batter')
                # Load ranks per year
                ranks_map: Dict[int,pd.DataFrame] = {}
                for y in years:
                    for ext in ('csv','parquet','json'):
                        rf = os.path.join(base_dir, f"{y}_springrank.{ext}")
                        if os.path.isfile(rf):
                            try:
                                if ext=='csv':
                                    ranks_map[y] = pd.read_csv(rf)
                                elif ext=='parquet':
                                    ranks_map[y] = pd.read_parquet(rf)
                                else:
                                    ranks_map[y] = pd.read_json(rf)
                            except Exception:
                                pass
                            break
                for y, rdf in ranks_map.items():
                    try:
                        # Try pybaseball for OPS
                        try:
                            import importlib as _il
                            _pb = _il.import_module('pybaseball')
                            stats = _pb.batting_stats(y)
                        except Exception:
                            stats = None
                        if stats is None or stats.empty:
                            continue
                        # Harmonize columns
                        name_col = None
                        for c in ('Name','name','player_name','Player','player'):
                            if c in stats.columns:
                                name_col = c; break
                        if name_col is None or 'OPS' not in stats.columns:
                            continue
                        r = rdf[['Player','Rank']].copy()
                        r['k'] = r['Player'].apply(_norm_name)
                        s = stats[[name_col,'OPS']].copy()
                        s['k'] = s[name_col].apply(_norm_name)
                        m = r.merge(s[['k','OPS']], on='k')
                        if m.empty:
                            continue
                        try:
                            spear = m['Rank'].rank(method='average').corr(m['OPS'].rank(method='average'))
                            pear = m['Rank'].corr(m['OPS'])
                            ops_rows.append([st, y, len(m), float(spear) if spear is not None else None, float(pear) if pear is not None else None])
                            # Persist joined for inspection
                            try:
                                os.makedirs(os.path.join(output_dir, 'ops_joined'), exist_ok=True)
                            except Exception:
                                pass
                            _write_multi(m[['Player','Rank','OPS']].sort_values('Rank', ascending=False), os.path.join(output_dir, 'ops_joined', f"{st}_{y}"), formats)
                        except Exception:
                            continue
                    except Exception:
                        continue
            if ops_rows:
                ops_df = pd.DataFrame(ops_rows, columns=['ScoreType','Year','Players','Spearman','Pearson'])
                _write_multi(ops_df, os.path.join(output_dir,'ops_correlation'), formats)
                if progress: print('[pipeline] ops_correlation written')
        # Baseline stat correlation with ranks (optional; computed when baseline AUC extra is enabled)
        try:
            val_cfg_top = cfg.get('validation', {})
            extra_val = val_cfg_top.get('extra', {}) if isinstance(val_cfg_top, dict) else {}
            if extra_val.get('baseline_auc'):
                bc_rows: List[List[Any]] = []
                def _norm_name2(s: Any) -> str:
                    import unicodedata as _ud
                    t = str(s) if not pd.isna(s) else ''
                    t = t.strip(); t = _ud.normalize('NFKD', t)
                    t = ''.join(c for c in t if not _ud.combining(c))
                    return ' '.join(t.split()).lower()
                for st in score_types:
                    for group in ['batter','pitcher']:
                        base_dir = os.path.join(output_dir, st, group)
                        ranks_map: Dict[int,pd.DataFrame] = {}
                        for y in years:
                            for ext in ('csv','parquet','json'):
                                rf = os.path.join(base_dir, f"{y}_springrank.{ext}")
                                if os.path.isfile(rf):
                                    try:
                                        if ext=='csv':
                                            ranks_map[y] = pd.read_csv(rf)
                                        elif ext=='parquet':
                                            ranks_map[y] = pd.read_parquet(rf)
                                        else:
                                            ranks_map[y] = pd.read_json(rf)
                                    except Exception:
                                        pass
                                    break
                        for y, rdf in ranks_map.items():
                            try:
                                # Fetch season stats
                                try:
                                    import importlib as _il
                                    _pb = _il.import_module('pybaseball')
                                except Exception:
                                    _pb = None
                                if _pb is None:
                                    continue
                                stats = _pb.batting_stats(y) if group=='batter' else _pb.pitching_stats(y)
                                if stats is None or stats.empty:
                                    continue
                                name_col = None
                                for c in ('Name','name','player_name','Player','player'):
                                    if c in stats.columns:
                                        name_col = c; break
                                if name_col is None:
                                    continue
                                metrics: List[Tuple[str,int]] = []
                                if group=='batter':
                                    for m, d in [('OPS', +1), ('xwOBA', +1), ('WAR', +1), ('xWAR', +1)]:
                                        if m in stats.columns: metrics.append((m, d))
                                else:
                                    tmp_stats = stats.copy()
                                    if 'WHIP' in tmp_stats.columns:
                                        metrics.append(('WHIP', -1))
                                    else:
                                        if all(c in tmp_stats.columns for c in ['BB','H','IP']):
                                            try:
                                                tmp_stats['WHIP'] = (tmp_stats['BB'].astype(float) + tmp_stats['H'].astype(float)) / tmp_stats['IP'].replace({0: np.nan}).astype(float)
                                                if not tmp_stats['WHIP'].isna().all():
                                                    metrics.append(('WHIP', -1))
                                            except Exception:
                                                pass
                                    if 'xERA' in tmp_stats.columns:
                                        metrics.append(('xERA', -1))
                                    elif 'ERA' in tmp_stats.columns:
                                        metrics.append(('ERA', -1))
                                    k9_aliases = ['K/9','SO9','SO/9','K9']
                                    k9_col = None
                                    for alias in k9_aliases:
                                        if alias in tmp_stats.columns:
                                            if alias != 'K/9':
                                                try:
                                                    tmp_stats = tmp_stats.rename(columns={alias: 'K/9'})
                                                except Exception:
                                                    pass
                                            k9_col = 'K/9'
                                            break
                                    if k9_col is None and all(c in tmp_stats.columns for c in ['SO','IP']):
                                        try:
                                            tmp_stats['K/9'] = (9.0 * tmp_stats['SO'].astype(float)) / tmp_stats['IP'].replace({0: np.nan}).astype(float)
                                            if not tmp_stats['K/9'].isna().all():
                                                k9_col = 'K/9'
                                        except Exception:
                                            pass
                                    if k9_col is not None:
                                        metrics.append(('K/9', +1))
                                    stats = tmp_stats
                                if not metrics:
                                    continue
                                r = rdf[['Player','Rank']].copy(); r['k'] = r['Player'].apply(_norm_name2)
                                s = stats.copy(); s['k'] = s[name_col].apply(_norm_name2)
                                for mcol, direction in metrics:
                                    if mcol not in s.columns: continue
                                    m = r.merge(s[['k', mcol]], on='k')
                                    if m.empty: continue
                                    vals = m[mcol].astype(float).values
                                    vals2 = vals if direction > 0 else -vals
                                    try:
                                        spear = pd.Series(m['Rank']).rank(method='average').corr(pd.Series(vals2).rank(method='average'))
                                        pear = pd.Series(m['Rank']).corr(pd.Series(vals2))
                                        bc_rows.append([st, group, y, mcol, len(m), float(spear) if spear is not None else None, float(pear) if pear is not None else None])
                                    except Exception:
                                        continue
                            except Exception:
                                continue
                if bc_rows:
                    bc_df = pd.DataFrame(bc_rows, columns=['ScoreType','Group','Year','Metric','Players','Spearman','Pearson'])
                    _write_multi(bc_df, os.path.join(output_dir, 'baseline_correlation'), formats)
                    if progress: print('[pipeline] baseline_correlation written')
        except Exception:
            pass
        # Iterative rank-weighted opponent re-pass (disabled)
        ir_cfg = cfg.get('analysis', {}).get('iterative_rank_weighted', {}) or {}
        if False and ir_cfg.get('enabled'):
            passes = int(ir_cfg.get('passes', 1) or 1)
            if passes > 0:
                if progress: print(f"[pipeline] iterative rank-weighted re-pass: passes={passes}")
                for st in [s for s in score_types if s in ('handmade','frequency')]:
                    for y in years:
                        for p in range(1, passes+1):
                            for group in ['batter','pitcher']:
                                opp_group = 'pitcher' if group=='batter' else 'batter'
                                # Load opponent ranks: prefer previous-pass iterative, else base
                                opp_rank_df = None
                                ir_opp_path = os.path.join(output_dir, 'iterative', st, opp_group, f"{y}_springrank_iter{p-1}.csv") if p>1 else None
                                base_opp_dir = os.path.join(output_dir, st, opp_group)
                                base_opp_path = os.path.join(base_opp_dir, f"{y}_springrank.csv")
                                try:
                                    if ir_opp_path and os.path.isfile(ir_opp_path):
                                        opp_rank_df = pd.read_csv(ir_opp_path)
                                    elif os.path.isfile(base_opp_path):
                                        opp_rank_df = pd.read_csv(base_opp_path)
                                    else:
                                        continue
                                except Exception:
                                    continue
                                opp_weight = {str(n): float(s) for n, s in opp_rank_df[['Player','Rank']].itertuples(index=False, name=None)}
                                # Load bipartite edges-only and select this group's rows to recompute unipartite with opponent weighting
                                try:
                                    if st == 'handmade':
                                        edge_only_path = os.path.join(cfg['paths']['raw_data_dir'], 'handmade', f"{y}_edges_only.csv")
                                    else:
                                        edge_only_path = os.path.join(cfg['paths']['raw_data_dir'], 'frequency', f"{y}_edges_only.csv")
                                    if not os.path.isfile(edge_only_path):
                                        continue
                                    bpdf = pd.read_csv(edge_only_path)
                                    if 'who_won' not in bpdf.columns:
                                        continue
                                    sel = 'batter' if group=='batter' else 'pitcher'
                                    sub = bpdf[bpdf['who_won']==sel]
                                    if sub.empty:
                                        continue
                                    gdf = sub[['winner','loser','score']].copy()
                                except Exception:
                                    continue
                                # Recompute edges weighting opponents by ranks
                                try:
                                    imetric = str(ir_cfg.get('metric','sum'))
                                    edf = _unipartite_vectorized(gdf, metric=imetric, year=y, raw_data_dir=cfg['paths']['raw_data_dir'], winners_role=group, opponent_weights=opp_weight)
                                except Exception:
                                    edf = None
                                if edf is None or edf.empty:
                                    continue
                                # Compute ranks from reweighted edges
                                G = nx.DiGraph()
                                try:
                                    G.add_weighted_edges_from(edf[['winner','loser','score']].itertuples(index=False, name=None))
                                except Exception:
                                    continue
                                node_list = list(G.nodes())
                                try:
                                    A = nx.to_scipy_sparse_matrix(G, dtype=float, nodelist=node_list)
                                except AttributeError:
                                    A = nx.to_scipy_sparse_array(G, dtype=float, nodelist=node_list)
                                try:
                                    import scipy.sparse as sp
                                    A = sp.csr_matrix(A)
                                except Exception:
                                    pass
                                raw_rp, sorted_rp = spring_rank(A, node_list)
                                # Persist ranks for this pass
                                ir_dir = os.path.join(output_dir, 'iterative', st, group)
                                os.makedirs(ir_dir, exist_ok=True)
                                _write_multi(pd.DataFrame(sorted_rp, columns=['Player','Rank']), os.path.join(ir_dir, f"{y}_springrank_iter{p}"), formats)
                                # Evaluate AUC on held-out pairs if requested
                                if bool(ir_cfg.get('evaluate_auc', True)):
                                    if st == 'handmade':
                                        edge_path = os.path.join('At Bats', f'{group}_data', 'handmade_scores', f"{y}_{group}_edges.csv")
                                    else:
                                        edge_path = os.path.join('At Bats', f'{group}_data', 'frequency_scores', f"{y}_{group}_edges.csv")
                                    if os.path.isfile(edge_path):
                                        val_cfg = cfg.get('validation', {})
                                        _G,_A,_nodes,_train,test_edges2 = make_graph_from_edge_csv(
                                            edge_path,
                                            validation_folds=cfg['validation_folds'],
                                            seed=val_cfg.get('seed'),
                                            sample_as_train=bool(val_cfg.get('sample_as_train', True)),
                                            index_base=int(val_cfg.get('index_base', 0))
                                        )
                                        if test_edges2:
                                            # Inline balanced-negatives AUC to avoid legacy branch bug
                                            try:
                                                rmap = {n: float(s) for n, s in sorted_rp}
                                                k_neg = int(val_cfg.get('negatives_per_positive', 1))
                                                # Build positive diffs and labels, and sample negatives per source
                                                pos_preds = []
                                                pos_labels = []
                                                auc_scores = []
                                                auc_labels = []
                                                used_cnt = 0
                                                from collections import defaultdict
                                                pos_by_u = defaultdict(set)
                                                losers = []
                                                for (u,v,w) in test_edges2:
                                                    try:
                                                        u = str(u); v = str(v)
                                                        pos_by_u[u].add(v)
                                                        losers.append(v)
                                                    except Exception:
                                                        continue
                                                unique_losers = sorted(set(losers))
                                                rng = np.random.RandomState(42)
                                                for (u,v,w) in test_edges2:
                                                    try:
                                                        u = str(u); v = str(v)
                                                        if (u not in rmap) or (v not in rmap):
                                                            continue
                                                        duv = float(rmap[u] - rmap[v])
                                                        pos_preds.append(1 if duv > 0 else 0)
                                                        pos_labels.append(1)
                                                        auc_scores.append(duv)
                                                        auc_labels.append(1)
                                                        if k_neg > 0 and len(unique_losers) > 1:
                                                            ban = pos_by_u.get(u, set())
                                                            cand = [x for x in unique_losers if (x != v and x not in ban and x in rmap)]
                                                            if cand:
                                                                m = min(k_neg, len(cand))
                                                                for vneg in rng.choice(cand, size=m, replace=False):
                                                                    auc_scores.append(float(rmap[u] - rmap[str(vneg)]))
                                                                    auc_labels.append(0)
                                                        used_cnt += 1
                                                    except Exception:
                                                        continue
                                                # Compute ACC/AUC
                                                acc2 = float(np.mean(np.array(pos_preds) == np.array(pos_labels))) if pos_preds else 0.0
                                                try:
                                                    from sklearn.metrics import roc_auc_score  # type: ignore
                                                    if len(set(auc_labels)) < 2:
                                                        auc2 = 0.5
                                                    else:
                                                        auc2 = float(roc_auc_score(auc_labels, auc_scores))
                                                except Exception:
                                                    auc2 = 0.5
                                                auc_rows.append([st, group, f"iter{p}", y, cfg.get('validation_folds',0), acc2, auc2, used_cnt])
                                            except Exception:
                                                pass
                                # Compare base vs iter on final pass
                                if p == passes:
                                    try:
                                        base_path = os.path.join(output_dir, st, group, f"{y}_springrank.csv")
                                        if os.path.isfile(base_path):
                                            bdf = pd.read_csv(base_path)
                                            idf = pd.DataFrame(sorted_rp, columns=['Player','Rank']).rename(columns={'Rank':'IterRank'})
                                            m = bdf.merge(idf, on='Player')
                                            if not m.empty:
                                                spear = m['Rank'].rank(method='average').corr(m['IterRank'].rank(method='average'))
                                                pear = m['Rank'].corr(m['IterRank'])
                                                m['Delta'] = m['IterRank'] - m['Rank']
                                                cmp_dir = os.path.join(output_dir, 'iterative', st, group)
                                                os.makedirs(cmp_dir, exist_ok=True)
                                                _write_multi(m.sort_values('Delta', ascending=False), os.path.join(cmp_dir, f"{y}_iter_compare"), formats)
                                                # append summary
                                                try:
                                                    cmp_summary_path = os.path.join(output_dir, 'iterative', 'compare_summary.csv')
                                                    # Write header if file does not exist or is empty
                                                    need_header = (not os.path.isfile(cmp_summary_path)) or (os.path.getsize(cmp_summary_path) == 0)
                                                    with open(cmp_summary_path, 'a', encoding='utf-8') as fh:
                                                        if need_header:
                                                            fh.write("ScoreType,Group,Year,N,Spearman,Pearson\n")
                                                        fh.write(f"{st},{group},{y},{len(m)},{spear},{pear}\n")
                                                except Exception:
                                                    pass
                                    except Exception:
                                        pass
                if progress: print('[pipeline] iterative rank-weighted re-pass written')
    # Next-year validation: train ranks on y, validate on y+1 test edges
    if cfg.get('analysis',{}).get('next_year_validation',{}).get('enabled'):
        ny_rows: List[List[Any]] = []
        for st in score_types:
            for group in ['batter','pitcher']:
                ys = sorted(base_ranks.get(st, {}).get(group, {}).keys())
                for y in ys:
                    y2 = y + 1
                    # Locate next year edge file
                    if st == 'handmade':
                        edge_path = os.path.join('At Bats', f'{group}_data', 'handmade_scores', f"{y2}_{group}_edges.csv")
                    elif st == 'frequency':
                        edge_path = os.path.join('At Bats', f'{group}_data', 'frequency_scores', f"{y2}_{group}_edges.csv")
                    elif st == 'aware':
                        edge_path = os.path.join('At Bats', f'{group}_data', 'aware_scores', f"{y2}_{group}_edges.csv")
                        # If next year's aware edges are missing, attempt to generate them from raw PAs
                        if not os.path.isfile(edge_path):
                            try:
                                ensure_aware_edges(y2, raw_data_dir, alpha_ridge=1.0, progress=progress)
                            except Exception:
                                pass
                    else:
                        # Unsupported for next-year at the moment
                        continue
                    if not os.path.isfile(edge_path):
                        continue
                    val_cfg = cfg.get('validation', {})
                    _G,_A,_nodes,_train,test_edges = make_graph_from_edge_csv(
                        edge_path,
                        validation_folds=cfg['validation_folds'],
                        seed=val_cfg.get('seed'),
                        sample_as_train=bool(val_cfg.get('sample_as_train', True)),
                        index_base=int(val_cfg.get('index_base', 0))
                    )
                    if not test_edges:
                        continue
                    res = _compute_acc_auc(
                        base_ranks[st][group][y],
                        test_edges,
                        auc_mode=str(val_cfg.get('aucMode','balanced-negatives')),
                        k_neg=int(val_cfg.get('negatives_per_positive', 1)),
                        auto_flip=bool(val_cfg.get('auto_flip', False)),
                    )
                    if res:
                        acc, auc, used = res
                        ny_rows.append([st, group, y, y2, cfg.get('validation_folds',0), acc, auc, used])
                        try:
                            frac, tot = _rank_orientation_fraction(base_ranks[st][group][y], list(test_edges))
                            ori = 'as-is' if frac >= 0.5 else 'flipped-at-eval'
                            orientation_notes.append(f"{st}:{group}:{y}->{y2}: NextYear orientation={ori} (p={frac:.3f}, Npos={tot})")
                        except Exception:
                            pass
        if ny_rows:
            ny_df = pd.DataFrame(ny_rows, columns=['ScoreType','Group','TrainYear','TestYear','Folds','Accuracy','AUC','TestEdges'])
            _write_multi(ny_df, os.path.join(output_dir, 'next_year_auc'), formats)
            if progress: print('[pipeline] next_year_auc written')
    # Write a Markdown output manifest summarizing what was computed
    try:
        now_ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        val_cfg_top = cfg.get('validation', {})
        extra_val = val_cfg_top.get('extra', {}) if isinstance(val_cfg_top, dict) else {}
        proc_metric = cfg.get('processing', {}).get('unipartite_metric', 'sum')
        ir_cfg = (cfg.get('analysis', {}) or {}).get('iterative_rank_weighted', {}) or {}
        ny_cfg = (cfg.get('analysis', {}) or {}).get('next_year_validation', {}) or {}
        methods = []
        methods.append(f"- Pairwise-negative AUC (original): folds={cfg.get('validation_folds', 0)}; auto_flip={bool(val_cfg_top.get('auto_flip', False))}; negatives_per_positive={int(val_cfg_top.get('negatives_per_positive', 1))}")
        if extra_val.get('opponent_blockout'):
            cvn = cfg.get('validation_folds', 0)
            held_frac = (1.0/float(cvn)) if cvn else 0.0
            methods.append(f"- Opponent-blockout: folds={cvn}; heldout_fraction≈{held_frac:.2f}")
        if extra_val.get('temperature_logloss'):
            methods.append("- Temperature log-loss: enabled")
        if extra_val.get('statcast_logloss'):
            methods.append("- Statcast log-loss: enabled")
        # Rank correlation and OPS correlation notes when enabled
        rankcorr_cfg = (cfg.get('analysis', {}) or {}).get('rank_correlation', {}) or {}
        if rankcorr_cfg.get('enabled'):
            methods.append(f"- Rank correlation: enabled; pairs={rankcorr_cfg.get('pairs','adjacent')}")
        ops_cfg = (cfg.get('analysis', {}) or {}).get('ops_correlation', {}) or {}
        if ops_cfg.get('enabled'):
            methods.append("- OPS correlation: enabled")
        if extra_val.get('baseline_auc'):
            # List the baseline stats actually used if available
            b_used = sorted(list(baseline_metrics_used.get('batter', set())))
            p_used = sorted(list(baseline_metrics_used.get('pitcher', set())))
            b_stats_str = ', '.join(b_used) if b_used else '(none found)'
            p_stats_str = ', '.join(p_used) if p_used else '(none found)'
            methods.append(f"- Other Baseline AUC: batters={b_stats_str}; pitchers={p_stats_str}")
            if extra_val.get('statcast_logloss'):
                methods.append(f"- Statcast log-loss stats: batters={b_stats_str}; pitchers={p_stats_str}")
        if (cfg.get('analysis', {}) or {}).get('iterative_rank_weighted', {}).get('enabled'):
            methods.append(f"- Iterative ranks: passes={int(ir_cfg.get('passes',0))}; metric={ir_cfg.get('metric','sum')}; evaluate_auc={bool(ir_cfg.get('evaluate_auc', False))}")
        if (cfg.get('analysis', {}) or {}).get('next_year_validation', {}).get('enabled'):
            methods.append("- Year-to-year validation: enabled")
        years_str = ', '.join(str(y) for y in cfg.get('years', []))
        score_types_str = ', '.join(cfg.get('score_types', []))
        groups_str = 'batter, pitcher'
        # Prepare output filepaths section
        potential_outputs = [
            ("validation_auc", os.path.join(output_dir, 'validation_auc.csv')),
            ("validation_logloss", os.path.join(output_dir, 'validation_logloss.csv')),
            ("validation_baseline_auc", os.path.join(output_dir, 'validation_baseline_auc.csv')),
            ("baseline_correlation", os.path.join(output_dir, 'baseline_correlation.csv')),
            ("next_year_auc", os.path.join(output_dir, 'next_year_auc.csv')),
            ("iterative_compare_summary", os.path.join(output_dir, 'iterative', 'compare_summary.csv')),
            ("rank_correlation", os.path.join(output_dir, 'rank_correlation.csv')),
            ("ops_correlation", os.path.join(output_dir, 'ops_correlation.csv')),
            ("levels_by_year", os.path.join(output_dir, 'levels_by_year.csv')),
            ("summary_top_players", os.path.join(output_dir, 'summary_top_players.csv')),
        ]
        out_files_lines = ["", "## Output files"]
        for label, p in potential_outputs:
            exists_tag = " (exists)" if os.path.isfile(p) else ""
            out_files_lines.append(f"- {label}: {p}{exists_tag}")

        lines = [
            f"# Output Manifest",
            f"Timestamp: {now_ts}",
            f"Training Years: {years_str}",
            f"Score types: {score_types_str}",
            f"Groups: {groups_str}",
            f"Unipartite accumulation: {proc_metric}",
            "",
            "## Methods",
            *methods,
            *out_files_lines,
        ]
        # Append orientation section
        if orientation_notes:
            lines.extend(["", "## Orientation checks", *[f"- {x}" for x in orientation_notes]])
        out_manifest = os.path.join(output_dir, 'output_manifest.md')
        with open(out_manifest, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        if progress:
            print('[pipeline] output_manifest.md written')
    except Exception as e:
        print(f"[warn] failed to write output_manifest.md: {e}")
    # Optional: quick k-sweep for n/(n+k) shrinkage to inspect sensitivity (does not change main outputs)
    try:
        ks_cfg = (cfg.get('analysis', {}) or {}).get('k_sweep', {}) or {}
        if ks_cfg.get('enabled') and ks_cfg.get('only'):
            _run_k_sweep_section(cfg, output_dir, formats, progress)
        elif ks_cfg.get('enabled'):
            bats = ks_cfg.get('batter_k', [100,125,175,200,225])
            pits = ks_cfg.get('pitcher_k', [300,350,450])
            years = cfg.get('years', [])
            out_rows = []
            for y in years:
                # Load raw PAs once and compute R_nk snapshots
                try:
                    raw_path = os.path.join(cfg['paths']['raw_data_dir'], f"at_bat_data_{y}.csv")
                    raw = pd.read_csv(raw_path) if os.path.isfile(raw_path) else None
                except Exception:
                    raw = None
                if raw is not None:
                    for kb in bats:
                        _write_R_nk(raw, 'batter', y, float(kb))
                    for kp in pits:
                        _write_R_nk(raw, 'pitcher', y, float(kp))
                for kb in bats:
                    for kp in pits:
                        for group in ['batter','pitcher']:
                            try:
                                k_val = float(kb) if group=='batter' else float(kp)
                                Rnk_path = os.path.join('At Bats', f'{group}_data','aware_scores', f"{y}_R_nk_k{int(k_val)}.csv")
                                R_df = pd.read_csv(Rnk_path) if os.path.isfile(Rnk_path) else None
                                R_map = {str(n): float(r) for n, r in (R_df[['Player','R_nk']].itertuples(index=False, name=None) if R_df is not None and not R_df.empty else [])}
                                edge_path = os.path.join('At Bats', f'{group}_data','aware_scores', f"{y}_{group}_edges.csv")
                                if not os.path.isfile(edge_path):
                                    continue
                                val_cfg = cfg.get('validation', {})
                                G,A,node_list,train_edges,test_edges = make_graph_from_edge_csv(
                                    edge_path,
                                    validation_folds=cfg['validation_folds'],
                                    seed=val_cfg.get('seed'),
                                    sample_as_train=bool(val_cfg.get('sample_as_train', True)),
                                    index_base=int(val_cfg.get('index_base', 0))
                                )
                                if not node_list or not test_edges:
                                    continue
                                try:
                                    import scipy.sparse as sp
                                    A = sp.csr_matrix(A)
                                except Exception:
                                    pass
                                aware_lambda = float(cfg.get('ranking', {}).get('aware_lambda', 1.0))
                                aware_harm = bool(cfg.get('ranking', {}).get('aware_harmonic', True))
                                _, sorted_r = aware_rank_with_tether(A, node_list, R_map, lambda_reg=aware_lambda, use_harmonic=aware_harm)
                                # Ensure at least 1 negative per positive unless explicitly set to 0
                                negs = int(val_cfg.get('negatives_per_positive', 1))
                                if negs is None:
                                    negs = 1
                                res = _compute_acc_auc(
                                    sorted_r,
                                    list(test_edges),
                                    auc_mode=str(val_cfg.get('aucMode','balanced-negatives')),
                                    k_neg=int(negs),
                                    auto_flip=bool(val_cfg.get('auto_flip', False)),
                                )
                                if res:
                                    acc, auc, used = res
                                    out_rows.append([y, int(kb), int(kp), group, float(acc), float(auc), int(used)])
                                    # Debug: compute positive-edge correctness rate over a small sample
                                    try:
                                        if progress:
                                            rmap = {n: s for n,s in sorted_r}
                                            cnt = 0; ok = 0
                                            for (u,v,_w) in list(test_edges)[:1000]:
                                                if (u in rmap) and (v in rmap):
                                                    cnt += 1; ok += (1 if (float(rmap[u]-rmap[v]) > 0) else 0)
                                            if cnt > 0:
                                                print(f"[k_sweep] {y}:{group} Kb={kb} Kp={kp} pos-correct={ok}/{cnt}={ok/max(cnt,1):.3f} acc={acc:.3f} auc={auc:.3f} used={used}")
                                    except Exception:
                                        pass
                            except Exception as _ks_e:
                                try:
                                    if progress:
                                        print(f"[k_sweep] skip {y}:{group} Kb={kb} Kp={kp} due to: {_ks_e}")
                                except Exception:
                                    pass
                                continue
            if out_rows:
                kdf = pd.DataFrame(out_rows, columns=['Year','Kb','Kp','Group','Accuracy','AUC','TestEdges'])
                _write_multi(kdf, os.path.join(output_dir, 'k_sweep_summary'), formats)
                try:
                    if progress:
                        print(f"[pipeline] k_sweep_summary written ({len(out_rows)} rows)")
                except Exception:
                    pass
            else:
                try:
                    if progress:
                        print("[pipeline] k_sweep produced 0 rows (check R_nk and edges availability)")
                except Exception:
                    pass
    except Exception:
        pass
    # Append validation snapshot with timestamp and config signature
    try:
        import time as _time
        snap_path = os.path.join(output_dir, 'validation_snapshot.json')
        ts = _time.strftime('%Y-%m-%dT%H:%M:%SZ', _time.gmtime())
        # Collect latest AUC/Logloss tables if available
        snap = {
            'timestamp': ts,
            'config_sig': cfg_sig,
            'years': years,
            'validation_folds': cfg.get('validation_folds', 0),
            'score_types': score_types,
            'files': {}
        }
        try:
            auc_path = os.path.join(output_dir, 'validation_auc.csv')
            if os.path.isfile(auc_path):
                snap['files']['validation_auc'] = auc_path
        except Exception:
            pass
        try:
            ll_path = os.path.join(output_dir, 'validation_logloss.csv')
            if os.path.isfile(ll_path):
                snap['files']['validation_logloss'] = ll_path
        except Exception:
            pass
        # For quick reference, also embed the last AUC rows (small sample: last 10 lines)
        try:
            import csv as _csv
            last_auc = []
            auc_path = os.path.join(output_dir, 'validation_auc.csv')
            if os.path.isfile(auc_path):
                import pandas as _pd
                df_auc = _pd.read_csv(auc_path)
                last_auc = df_auc.tail(10).to_dict(orient='records')
            snap['last_auc'] = last_auc
        except Exception:
            pass
        # Append as JSON line
        with open(snap_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(snap) + '\n')
        if progress:
            print('[pipeline] validation_snapshot.json appended')
    except Exception:
        pass
    # Mobility Metrics (quartile transitions) if enabled and scaled ranks produced
    if cfg.get('analysis',{}).get('mobility',{}).get('enabled') and cfg['ranking']['scale_ranks']:
        mobility_rows = []
        # For each score_type/group/condition track scaled rank files across years
        def load_scaled(base_dir: str, years: List[int]) -> Dict[int,pd.DataFrame]:
            out = {}
            for y in years:
                path_csv = base_dir + f"/{y}_springrank_scaled.csv"
                path_parquet = base_dir + f"/{y}_springrank_scaled.parquet"
                path_json = base_dir + f"/{y}_springrank_scaled.json"
                if os.path.isfile(path_csv):
                    out[y] = pd.read_csv(path_csv)
                elif os.path.isfile(path_parquet):
                    try:
                        out[y] = pd.read_parquet(path_parquet)
                    except Exception:
                        continue
                elif os.path.isfile(path_json):
                    out[y] = pd.read_json(path_json)
            return out
        def compute_mobility(df_prev: pd.DataFrame, df_curr: pd.DataFrame, y_prev: int, y_curr: int, score_type: str, group: str, condition: Any):
            if df_prev is None or df_curr is None: return
            # Merge on Player
            mprev = df_prev[['Player','ScaledRank']].rename(columns={'ScaledRank':'ScaledRankPrev'})
            mcurr = df_curr[['Player','ScaledRank']].rename(columns={'ScaledRank':'ScaledRankCurr'})
            merged = mprev.merge(mcurr, on='Player')
            if merged.empty: return
            # Assign quartiles (Q1 = top 25%) based on rank ordering (higher is better)
            merged['QuartilePrev'] = pd.qcut(merged['ScaledRankPrev'].rank(method='first', ascending=False), 4, labels=[1,2,3,4])
            merged['QuartileCurr'] = pd.qcut(merged['ScaledRankCurr'].rank(method='first', ascending=False), 4, labels=[1,2,3,4])
            # Mobility event counts
            up_any = (merged['QuartileCurr'] < merged['QuartilePrev']).sum()
            down_any = (merged['QuartileCurr'] > merged['QuartilePrev']).sum()
            same = (merged['QuartileCurr'] == merged['QuartilePrev']).sum()
            total = len(merged)
            moved_2_or_more = ( (merged['QuartilePrev'] - merged['QuartileCurr']).abs() >= 2 ).sum()
            mobility_rows.append([
                score_type, group, condition, y_prev, y_curr,
                total, up_any, down_any, same, moved_2_or_more,
                round(up_any/total if total else 0,4),
                round(down_any/total if total else 0,4),
                round(moved_2_or_more/total if total else 0,4)
            ])
        for st in score_types:
            if st in ('handmade','frequency'):
                for group in ['batter','pitcher']:
                    base_dir = os.path.join(output_dir, st, group)
                    scaled_years = load_scaled(base_dir, years)
                    sorted_years = sorted(scaled_years.keys())
                    for i in range(1, len(sorted_years)):
                        compute_mobility(scaled_years[sorted_years[i-1]], scaled_years[sorted_years[i]], sorted_years[i-1], sorted_years[i], st, group, None)
            elif st == 'pitch_type':
                for group in ['batter','pitcher']:
                    for pt in (pitch_types or ALLOWED_PITCH_TYPES):
                        base_dir = os.path.join(output_dir, st, group, pt)
                        scaled_years = load_scaled(base_dir, years)
                        sorted_years = sorted(scaled_years.keys())
                        for i in range(1, len(sorted_years)):
                            compute_mobility(scaled_years[sorted_years[i-1]], scaled_years[sorted_years[i]], sorted_years[i-1], sorted_years[i], st, group, pt)
            elif st == 'inning':
                for group in ['batter','pitcher']:
                    for inn in innings:
                        base_dir = os.path.join(output_dir, st, group, str(inn))
                        scaled_years = load_scaled(base_dir, years)
                        sorted_years = sorted(scaled_years.keys())
                        for i in range(1, len(sorted_years)):
                            compute_mobility(scaled_years[sorted_years[i-1]], scaled_years[sorted_years[i]], sorted_years[i-1], sorted_years[i], st, group, inn)
        if mobility_rows:
            mob_cols = ['ScoreType','Group','Condition','YearPrev','YearCurr','Players','Up','Down','Same','Moved2Plus','FracUp','FracDown','FracMoved2Plus']
            mobility_df = pd.DataFrame(mobility_rows, columns=mob_cols)
            _write_multi(mobility_df, os.path.join(output_dir,'mobility_report'), formats)
            if progress: print('[pipeline] mobility_report written')
    if caching_enabled:
        manifest['last_runtime_seconds'] = time.time() - start_global
        _save_manifest(manifest_path, manifest)
    # Anomaly detection (large year-over-year scaled rank deltas) after mobility
    if cfg.get('analysis',{}).get('anomalies',{}).get('enabled') and cfg['ranking']['scale_ranks']:
        an_cfg = cfg['analysis']['anomalies']
        method = an_cfg.get('method','quantile')
        q = an_cfg.get('quantile',0.95)
        abs_thr = an_cfg.get('abs_threshold',0.2)
        min_players = an_cfg.get('min_players',20)
        anomaly_rows = []
        def load_scaled_generic(base_dir: str, years: List[int]) -> Dict[int,pd.DataFrame]:
            out = {}
            for y in years:
                for ext in ('csv','parquet','json'):
                    path = os.path.join(base_dir, f"{y}_springrank_scaled.{ext}")
                    if os.path.isfile(path):
                        try:
                            if ext=='csv':
                                out[y] = pd.read_csv(path)
                            elif ext=='parquet':
                                out[y] = pd.read_parquet(path)
                            else:
                                out[y] = pd.read_json(path)
                        except Exception:
                            continue
                        break
            return out
        def process_anomalies(st: str, group: str, condition: Any, base_dir: str):
            scaled_map = load_scaled_generic(base_dir, years)
            sorted_years = sorted(scaled_map.keys())
            if len(sorted_years) < 2:
                return
            deltas_all = []  # collect abs deltas to compute quantile threshold
            pair_deltas = []  # store (meta, list_of_rows) until threshold known
            for i in range(1, len(sorted_years)):
                y_prev = sorted_years[i-1]; y_curr = sorted_years[i]
                df_prev = scaled_map[y_prev][['Player','ScaledRank']].rename(columns={'ScaledRank':'Prev'})
                df_curr = scaled_map[y_curr][['Player','ScaledRank']].rename(columns={'ScaledRank':'Curr'})
                merged = df_prev.merge(df_curr, on='Player')
                if len(merged) < min_players:
                    continue
                merged['Delta'] = merged['Curr'] - merged['Prev']
                merged['AbsDelta'] = merged['Delta'].abs()
                deltas_all.extend(merged['AbsDelta'].tolist())
                pair_deltas.append((y_prev,y_curr,merged))
            if not deltas_all:
                return
            if method == 'quantile':
                threshold = float(pd.Series(deltas_all).quantile(q))
            else:
                threshold = abs_thr
            for (y_prev,y_curr,merged) in pair_deltas:
                sel = merged[merged['AbsDelta'] >= threshold]
                if sel.empty: continue
                for row in sel.itertuples(index=False):
                    anomaly_rows.append([
                        st, group, condition, y_prev, y_curr,
                        row.Player, row.Prev, row.Curr, row.Delta,
                        'up' if row.Delta>0 else 'down', row.AbsDelta, threshold, method
                    ])
        for st in score_types:
            if st in ('handmade','frequency'):
                for group in ['batter','pitcher']:
                    base_dir = os.path.join(output_dir, st, group)
                    process_anomalies(st, group, None, base_dir)
            elif st == 'pitch_type':
                for group in ['batter','pitcher']:
                    for pt in (pitch_types or ALLOWED_PITCH_TYPES):
                        base_dir = os.path.join(output_dir, st, group, pt)
                        process_anomalies(st, group, pt, base_dir)
            elif st == 'inning':
                for group in ['batter','pitcher']:
                    for inn in innings:
                        base_dir = os.path.join(output_dir, st, group, str(inn))
                        process_anomalies(st, group, inn, base_dir)
        if anomaly_rows:
            an_cols = ['ScoreType','Group','Condition','YearPrev','YearCurr','Player','ScaledRankPrev','ScaledRankCurr','Delta','Direction','AbsDelta','Threshold','Method']
            anomalies_df = pd.DataFrame(anomaly_rows, columns=an_cols)
            _write_multi(anomalies_df, os.path.join(output_dir,'anomalies_report'), formats)
            if progress: print('[pipeline] anomalies_report written')
    # Rolling window rankings (aggregate edges across last N seasons) - computed only for base score types (handmade/frequency) for simplicity
    if cfg.get('analysis',{}).get('rolling',{}).get('enabled'):
        windows = cfg['analysis']['rolling']['windows']
        rolling_records = []
        for st in score_types:
            if st not in ('handmade','frequency'):
                continue
            for group in ['batter','pitcher']:
                # Collect per-year edge paths
                year_edge_paths = {}
                for y in years:
                    if st == 'handmade':
                        ep = os.path.join('At Bats', f'{group}_data','handmade_scores', f"{y}_{group}_edges.csv")
                    else:
                        ep = os.path.join('At Bats', f'{group}_data','frequency_scores', f"{y}_{group}_edges.csv")
                    if os.path.isfile(ep):
                        year_edge_paths[y] = ep
                sorted_years = sorted(year_edge_paths.keys())
                for win in windows:
                    if len(sorted_years) < win: continue
                    for i in range(win-1, len(sorted_years)):
                        span_years = sorted_years[i-win+1:i+1]
                        # Build aggregate graph by summing adjacency matrices
                        combined_edges = []
                        for sy in span_years:
                            df_e = pd.read_csv(year_edge_paths[sy])
                            combined_edges.append(df_e)
                        if not combined_edges: continue
                        agg = pd.concat(combined_edges, ignore_index=True)
                        # assume columns winner,loser,score
                        agg_grouped = agg.groupby(['winner','loser']).sum().reset_index()
                        # Build graph
                        G = nx.DiGraph()
                        G.add_weighted_edges_from(agg_grouped[['winner','loser','score']].itertuples(index=False, name=None))
                        node_list = list(G.nodes())
                        try:
                            A = nx.to_scipy_sparse_matrix(G, dtype=float, nodelist=node_list)
                        except AttributeError:
                            A = nx.to_scipy_sparse_array(G, dtype=float, nodelist=node_list)
                        try:
                            import scipy.sparse as sp
                            A = sp.csr_matrix(A)
                        except Exception:
                            pass
                        raw_r, sorted_r = spring_rank(A, node_list)
                        rank_dir = os.path.join(output_dir, 'rolling', st, group, f"win{win}")
                        os.makedirs(rank_dir, exist_ok=True)
                        label = f"{span_years[0]}_{span_years[-1]}"
                        _write_multi(pd.DataFrame(sorted_r, columns=['Player','Rank']), os.path.join(rank_dir, f"{label}_springrank"), formats)
                        if scale_req:
                            scaled = scale_ranks(A, raw_r)
                            scaled_sorted = [[node_list[i], r] for i,r in enumerate(scaled)]
                            scaled_sorted.sort(reverse=True, key=lambda x: x[1])
                            _write_multi(pd.DataFrame(scaled_sorted, columns=['Player','ScaledRank']), os.path.join(rank_dir, f"{label}_springrank_scaled"), formats)
                        rolling_records.append([st, group, win, span_years[0], span_years[-1], len(node_list), A.count_nonzero()])
        if rolling_records:
            roll_df = pd.DataFrame(rolling_records, columns=['ScoreType','Group','Window','StartYear','EndYear','Nodes','Edges'])
            _write_multi(roll_df, os.path.join(output_dir,'rolling_summary'), formats)
            if progress: print('[pipeline] rolling_summary written')
    return True

# ------------------------- Helper partial stages -------------------------- #

def generate_edges(cfg: Dict[str,Any]) -> bool:
    """Generate edges and unipartite conversions only (no rankings)."""
    years = cfg['years']
    raw_data_dir = cfg['paths']['raw_data_dir']
    progress = cfg['logging']['progress']
    force_scrape = cfg['scrape']['force']
    force_edges = cfg.get('edges',{}).get('force', False)
    ensure_scraped(years, raw_data_dir, force_scrape, progress)
    score_types = cfg['score_types']
    pitch_types = cfg.get('pitch_types') or []
    innings = cfg.get('innings') or list(range(1,10))
    stand_filter = cfg['filters'].get('stand') if 'filters' in cfg else None
    pthrows_filter = cfg['filters'].get('p_throws') if 'filters' in cfg else None
    for y in years:
        for st in score_types:
            ensure_edge_only(y, st, raw_data_dir, progress, pitch_types=pitch_types, innings=innings, stand_filter=stand_filter, pthrows_filter=pthrows_filter, force=force_edges)
    # Unipartite conversion (copied from run_pipeline)
    metric = cfg.get('processing',{}).get('unipartite_metric','sum')
    for y in years:
        for st in score_types:
            if st == 'handmade':
                edge_file = os.path.join(raw_data_dir, 'handmade', f"{y}_edges_only.csv")
                out_batter_dir = os.path.join('At Bats','batter_data','handmade_scores')
                out_pitcher_dir = os.path.join('At Bats','pitcher_data','handmade_scores')
                os.makedirs(out_batter_dir, exist_ok=True); os.makedirs(out_pitcher_dir, exist_ok=True)
                b_edge_out = os.path.join(out_batter_dir, f"{y}_batter_edges.csv")
                p_edge_out = os.path.join(out_pitcher_dir, f"{y}_pitcher_edges.csv")
                if os.path.isfile(edge_file) and (force_edges or not (os.path.isfile(b_edge_out) and os.path.isfile(p_edge_out))):
                    to_2_unipartite(edge_file, b_edge_out, p_edge_out, metric=metric, raw_data_dir=raw_data_dir)
            elif st == 'frequency':
                edge_file = os.path.join(raw_data_dir, 'frequency', f"{y}_edges_only.csv")
                b_edge_out = os.path.join('At Bats','batter_data','frequency_scores', f"{y}_batter_edges.csv")
                p_edge_out = os.path.join('At Bats','pitcher_data','frequency_scores', f"{y}_pitcher_edges.csv")
                if os.path.isfile(edge_file) and (force_edges or not (os.path.isfile(b_edge_out) and os.path.isfile(p_edge_out))):
                    to_2_unipartite(edge_file, b_edge_out, p_edge_out, metric=metric, raw_data_dir=raw_data_dir)
            elif st == 'pitch_type':
                for pt in (pitch_types or ALLOWED_PITCH_TYPES):
                    edge_file = os.path.join(raw_data_dir, 'pitch_type', pt, f"{y}_edges_only.csv")
                    b_edge_out = os.path.join('At Bats','batter_data','pitchtype_scores', pt, f"{y}_batter_edges.csv")
                    p_edge_out = os.path.join('At Bats','pitcher_data','pitchtype_scores', pt, f"{y}_pitcher_edges.csv")
                    os.makedirs(os.path.dirname(b_edge_out), exist_ok=True); os.makedirs(os.path.dirname(p_edge_out), exist_ok=True)
                    if os.path.isfile(edge_file) and (force_edges or not (os.path.isfile(b_edge_out) and os.path.isfile(p_edge_out))):
                        to_2_unipartite(edge_file, b_edge_out, p_edge_out, metric=metric, raw_data_dir=raw_data_dir)
            elif st == 'inning':
                for inn in innings:
                    edge_file = os.path.join(raw_data_dir, 'inning', str(inn), f"{y}_edges_only.csv")
                    b_edge_out = os.path.join('At Bats','batter_data','inning_scores', str(inn), f"{y}_batter_edges.csv")
                    p_edge_out = os.path.join('At Bats','pitcher_data','inning_scores', str(inn), f"{y}_pitcher_edges.csv")
                    os.makedirs(os.path.dirname(b_edge_out), exist_ok=True); os.makedirs(os.path.dirname(p_edge_out), exist_ok=True)
                    if os.path.isfile(edge_file) and (force_edges or not (os.path.isfile(b_edge_out) and os.path.isfile(p_edge_out))):
                        to_2_unipartite(edge_file, b_edge_out, p_edge_out, metric=metric, raw_data_dir=raw_data_dir)
    if progress: print('[edges] generation complete')
    return True

def compute_rankings(cfg: Dict[str,Any]) -> bool:
    """Compute rankings & analyses assuming edges already converted."""
    output_dir = cfg['paths']['output_dir']; os.makedirs(output_dir, exist_ok=True)
    score_types = cfg['score_types']; years = cfg['years']
    pitch_types = cfg.get('pitch_types') or []
    innings = cfg.get('innings') or list(range(1,10))
    formats = cfg['output']['formats']
    progress = cfg['logging']['progress']
    caching_enabled = cfg.get('caching',{}).get('enabled', False)
    manifest_path = cfg.get('caching',{}).get('manifest', os.path.join(output_dir,'manifest.json'))
    manifest = _load_manifest(manifest_path) if caching_enabled else {'runs':{}}
    manifest.setdefault('signatures', {})
    cfg_sig = _config_signature(cfg)
    manifest['signatures']['last_config'] = cfg_sig
    top_n = cfg['ranking']['top_n']; scale_req = cfg['ranking']['scale_ranks']
    validation_rows = []; levels_records = []; results_summary = []
    did_any = False
    start_global = time.time()
    for st in score_types:
        for group in ['batter','pitcher']:
            for y in years:
                cache_prefix = f"{y}:{st}:{group}"
                if st == 'pitch_type':
                    for pt in (pitch_types or ALLOWED_PITCH_TYPES):
                        edge_path = os.path.join('At Bats', f'{group}_data','pitchtype_scores', pt, f"{y}_{group}_edges.csv")
                        if not os.path.isfile(edge_path): continue
                        cache_key = cache_prefix+f":{pt}"; file_sig = _file_signature(edge_path)
                        if False and caching_enabled and cache_key in manifest['runs']:
                            prev = manifest['runs'][cache_key]
                            if prev.get('file_sig') == file_sig and prev.get('config_sig') == cfg_sig:
                                if progress: print(f"[cache] skip {cache_key}"); continue
                        t0 = time.time()
                        G,A,node_list,_,_ = make_graph_from_edge_csv(edge_path, validation_folds=cfg['validation_folds'])
                        raw_r, sorted_r = spring_rank(A, node_list)
                        rank_dir = os.path.join(output_dir, st, group, pt); os.makedirs(rank_dir, exist_ok=True)
                        _write_multi(pd.DataFrame(sorted_r, columns=['Player','Rank']), os.path.join(rank_dir,f"{y}_springrank"), formats)
                        if scale_req:
                            scaled = scale_ranks(A, raw_r)
                            scaled_sorted = [[node_list[i], r] for i,r in enumerate(scaled)]; scaled_sorted.sort(reverse=True, key=lambda x: x[1])
                            _write_multi(pd.DataFrame(scaled_sorted, columns=['Player','ScaledRank']), os.path.join(rank_dir,f"{y}_springrank_scaled"), formats)
                            levels_records.append([st, group, pt, y, max(scaled)-min(scaled)])
                        validation_rows.append([st, group, pt, y, len(node_list), A.count_nonzero(), float(A.count_nonzero())/(len(node_list)**2 if len(node_list)>0 else 1)])
                        if caching_enabled:
                            manifest['runs'][cache_key] = {"time": time.time()-t0, "nodes": len(node_list), 'file_sig': file_sig, 'config_sig': cfg_sig}
                        results_summary.append(pd.DataFrame(sorted_r, columns=['Player','Rank']).head(top_n).assign(Year=y, Group=group, ScoreType=st, PitchType=pt))
                        did_any = True
                    continue
                if st == 'inning':
                    for inn in innings:
                        edge_path = os.path.join('At Bats', f'{group}_data','inning_scores', str(inn), f"{y}_{group}_edges.csv")
                        if not os.path.isfile(edge_path): continue
                        cache_key = cache_prefix+f":inn{inn}"; file_sig = _file_signature(edge_path)
                        if caching_enabled and cache_key in manifest['runs']:
                            prev = manifest['runs'][cache_key]
                            if prev.get('file_sig') == file_sig and prev.get('config_sig') == cfg_sig:
                                if progress: print(f"[cache] skip {cache_key}"); continue
                        t0 = time.time()
                        G,A,node_list,_,_ = make_graph_from_edge_csv(edge_path, validation_folds=cfg['validation_folds'])
                        raw_r, sorted_r = spring_rank(A, node_list)
                        rank_dir = os.path.join(output_dir, st, group, str(inn)); os.makedirs(rank_dir, exist_ok=True)
                        _write_multi(pd.DataFrame(sorted_r, columns=['Player','Rank']), os.path.join(rank_dir,f"{y}_springrank"), formats)
                        if scale_req:
                            scaled = scale_ranks(A, raw_r)
                            scaled_sorted = [[node_list[i], r] for i,r in enumerate(scaled)]; scaled_sorted.sort(reverse=True, key=lambda x: x[1])
                            _write_multi(pd.DataFrame(scaled_sorted, columns=['Player','ScaledRank']), os.path.join(rank_dir,f"{y}_springrank_scaled"), formats)
                            levels_records.append([st, group, inn, y, max(scaled)-min(scaled)])
                        validation_rows.append([st, group, f"inning_{inn}", y, len(node_list), A.count_nonzero(), float(A.count_nonzero())/(len(node_list)**2 if len(node_list)>0 else 1)])
                        if caching_enabled:
                            manifest['runs'][cache_key] = {"time": time.time()-t0, "nodes": len(node_list), 'file_sig': file_sig, 'config_sig': cfg_sig}
                        results_summary.append(pd.DataFrame(sorted_r, columns=['Player','Rank']).head(top_n).assign(Year=y, Group=group, ScoreType=st, Inning=inn))
                        did_any = True
                    continue
                # base handmade/frequency/aware
                if st == 'handmade':
                    edge_path = os.path.join('At Bats', f'{group}_data','handmade_scores', f"{y}_{group}_edges.csv")
                elif st == 'frequency':
                    edge_path = os.path.join('At Bats', f'{group}_data','frequency_scores', f"{y}_{group}_edges.csv")
                elif st == 'aware':
                    edge_path = os.path.join('At Bats', f'{group}_data','aware_scores', f"{y}_{group}_edges.csv")
                else:
                    continue
                if not os.path.isfile(edge_path): continue
                cache_key = cache_prefix; file_sig = _file_signature(edge_path)
                if caching_enabled and cache_key in manifest['runs']:
                    prev = manifest['runs'][cache_key]
                    if prev.get('file_sig') == file_sig and prev.get('config_sig') == cfg_sig:
                        if progress: print(f"[cache] skip {cache_key}"); continue
                t0 = time.time()
                G,A,node_list,_,_ = make_graph_from_edge_csv(edge_path, validation_folds=cfg['validation_folds'])
                if st == 'aware':
                    aware_lambda = float(cfg.get('ranking', {}).get('aware_lambda', 1.0))
                    aware_harm = bool(cfg.get('ranking', {}).get('aware_harmonic', True))
                    R_map: Dict[str, float] = {}
                    try:
                        R_path = os.path.join('At Bats', f'{group}_data','aware_scores', f"{y}_R.csv")
                        if os.path.isfile(R_path):
                            R_df = pd.read_csv(R_path)
                            if 'Player' not in R_df.columns and 'winner' in R_df.columns:
                                R_df = R_df.rename(columns={'winner': 'Player'})
                            if {'Player','R'}.issubset(R_df.columns):
                                R_map = {str(n): float(r) for n, r in R_df[['Player','R']].dropna().itertuples(index=False, name=None)}
                    except Exception:
                        R_map = {}
                    raw_r, sorted_r = aware_rank_with_tether(A, node_list, R_map, lambda_reg=aware_lambda, use_harmonic=aware_harm)
                else:
                    raw_r, sorted_r = spring_rank(A, node_list)
                rank_dir = os.path.join(output_dir, st, group); os.makedirs(rank_dir, exist_ok=True)
                _write_multi(pd.DataFrame(sorted_r, columns=['Player','Rank']), os.path.join(rank_dir,f"{y}_springrank"), formats)
                if scale_req:
                    scaled = scale_ranks(A, raw_r)
                    scaled_sorted = [[node_list[i], r] for i,r in enumerate(scaled)]; scaled_sorted.sort(reverse=True, key=lambda x: x[1])
                    _write_multi(pd.DataFrame(scaled_sorted, columns=['Player','ScaledRank']), os.path.join(rank_dir,f"{y}_springrank_scaled"), formats)
                    levels_records.append([st, group, None, y, max(scaled)-min(scaled)])
                validation_rows.append([st, group, None, y, len(node_list), A.count_nonzero(), float(A.count_nonzero())/(len(node_list)**2 if len(node_list)>0 else 1)])
                if caching_enabled:
                    manifest['runs'][cache_key] = {"time": time.time()-t0, "nodes": len(node_list), 'file_sig': file_sig, 'config_sig': cfg_sig}
                results_summary.append(pd.DataFrame(sorted_r, columns=['Player','Rank']).head(top_n).assign(Year=y, Group=group, ScoreType=st))
                did_any = True
    # Outputs
    if results_summary:
        summary_df = pd.concat(results_summary, ignore_index=True)
        _write_multi(summary_df, os.path.join(output_dir,'summary_top_players'), formats)
    elif progress:
        print('[rank] No rank outputs produced (missing edge files?). Try running the edges or full pipeline first.')
    if cfg['ranking']['output_levels'] and levels_records:
        levels_df = pd.DataFrame(levels_records, columns=['ScoreType','Group','Condition','Year','LevelsRange'])
        _write_multi(levels_df, os.path.join(output_dir,'levels_by_year'), formats)
    if validation_rows:
        val_df = pd.DataFrame(validation_rows, columns=['ScoreType','Group','Condition','Year','Nodes','Edges','Density'])
        _write_multi(val_df, os.path.join(output_dir,'validation_report'), formats)
    # Reuse analysis sections from run_pipeline by calling mobility/anomaly/rolling if needed
    # Simplest route: call run_pipeline analysis part by constructing minimal structure; to avoid duplication we lightly invoke those blocks.
    # For simplicity we skip re-running mobility/anomalies here; user can run full pipeline or rely on full command.
    if caching_enabled:
        manifest['last_runtime_seconds'] = time.time() - start_global
        _save_manifest(manifest_path, manifest)
    if progress: print('[rank] compute_rankings complete')
    return True

if __name__ == '__main__':
    import argparse
    from config.loader import load_config
    ap = argparse.ArgumentParser(description='Run MLB pipeline with JSON config')
    ap.add_argument('--config','-c', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_pipeline(cfg)
