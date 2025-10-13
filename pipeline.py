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

# Reuse logic from Rankings.py minimally (avoid import side-effects)
# (Potential improvement: refactor Rankings.py into functions and import.)

ALLOWED_PITCH_TYPES = ["CH","CU","FC","FF","FS","FT","SI","SL"]

# Base handcrafted scoring (from add_edgeinfo.py) and base pitcher scoring
BASE_BATTER_SCORING = {'hit_by_pitch':1,'walk':2,'single':3,'double':6,'triple':9,'home_run':12}
BASE_PITCHER_SCORING = {'fielders_choice':1,'fielders_choice_out':1,'other_out':1,'field_out':1,'force_out':2,'grounded_into_double_play':2,'strikeout':6}

# ------------------------- Scraping ---------------------------------------- #

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

    Safe no-op if none found or lookups fail.
    """
    try:
        cols = ['winner','loser']
        masks = [edf[c].astype(str).apply(lambda x: _extract_mlbam_id_edges(x) is not None) for c in cols]
        ids = []
        for m,c in zip(masks, cols):
            if m.any():
                ids.extend([ _extract_mlbam_id_edges(x) for x in edf.loc[m,c].astype(str) if _extract_mlbam_id_edges(x) is not None ])
        ids = sorted(set([i for i in ids if i is not None]))
        if not ids:
            return edf
        name_map: dict[int,str] = {}
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
            return edf
        def _map_val(v: str) -> str:
            pid = _extract_mlbam_id_edges(v)
            if pid is not None:
                return name_map.get(pid, v)
            return v
        out = edf.copy()
        for c in cols:
            out[c] = out[c].map(_map_val)
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


def make_graph_from_edge_csv(path: str, weights: bool=True, validation_folds: int=0):
    df = pd.read_csv(path)
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
            sel_inds = np.random.choice(np.arange(m, dtype=int), int(m*(1-(1/validation_folds))), replace=False)
            not_sel = np.setdiff1d(np.arange(m, dtype=int), sel_inds)
            # Sample by index from the list
            train_edges = [edge_list[i] for i in sel_inds]
            test_edges = [edge_list[i] for i in not_sel]
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
) -> Optional[Tuple[float, float, int]]:
    """Compute ACC/AUC against held-out directed edges using rank differences.

    Modes:
    - balanced-negatives (default): For each held-out positive (u->v), score = r[u]-r[v] with label 1,
      and add k_neg negatives for the same source u by sampling v' not in the positive set, label 0.
      This tests whether rank differences discriminate winners from random non-winners.
    - legacy: Use only the held-out positives. Accuracy is fraction of dv = r[u]-r[v] > 0.
      For AUC, treat label = 1 if prediction is correct (dv>0) else 0, and score = |dv| as a
      confidence measure. This mirrors the earlier "confidence-vs-correctness" style AUC.

    Returns (accuracy, auc, used_test_edges) or None if no edges usable.
    """
    if test_edges is None or len(test_edges) == 0:
        return None
    # Map player -> rank score
    rank_map = {name: float(score) for name, score in sorted_ranks}
    # Legacy mode: accuracy on positives; AUC as confidence (|dv|) vs correctness label
    if str(auc_mode).lower() == "legacy":
        acc_preds: List[int] = []
        acc_obs: List[int] = []
        auc_scores: List[float] = []
        auc_labels: List[int] = []
        used = 0
        for edge in test_edges:
            try:
                u, v, _w = str(edge[0]), str(edge[1]), edge[2]
                si = rank_map.get(u)
                sj = rank_map.get(v)
                if si is None or sj is None:
                    continue
                dv = float(si - sj)
                correct = 1 if dv > 0 else 0
                acc_preds.append(correct)
                acc_obs.append(1)  # held-out edge is a positive (u beat v)
                # Confidence-style AUC: magnitude of separation should be larger when prediction is correct
                auc_scores.append(abs(dv))
                auc_labels.append(correct)
                used += 1
            except Exception:
                continue
        if used == 0:
            return None
        acc = float(np.mean(np.array(acc_preds) == np.array(acc_obs)))
        # If only one class (all correct or all incorrect), fall back to 0.5
        try:
            from sklearn.metrics import roc_auc_score  # type: ignore
            if len(set(auc_labels)) < 2:
                auc = 0.5
            else:
                auc = float(roc_auc_score(auc_labels, auc_scores))
        except Exception:
            auc = 0.5
        return acc, auc, used

    # Default (balanced-negatives) mode
    acc_preds2: List[int] = []
    acc_obs2: List[int] = []
    auc_scores2: List[float] = []
    auc_labels2: List[int] = []
    used2 = 0
    # Build quick lookup structures from test set for negative sampling per source
    test_pos_by_u: Dict[str, set] = {}
    losers_pool: List[str] = []
    for edge in test_edges:
        try:
            u, v = str(edge[0]), str(edge[1])
        except Exception:
            continue
        test_pos_by_u.setdefault(u, set()).add(v)
        losers_pool.append(v)
    unique_losers = np.array(sorted(set(losers_pool)))
    rng = np.random.RandomState(42)  # reproducible sampling
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
            if k_neg > 0 and unique_losers.size > 1:
                ban = test_pos_by_u.get(u, set())
                # Fast mask for candidates
                mask = np.vectorize(lambda x: (x != v) and (x not in ban))(unique_losers)
                cand = unique_losers[mask]
                if cand.size > 0:
                    sample_idx = rng.choice(np.arange(cand.size), size=min(k_neg, cand.size), replace=False)
                    for idx in np.atleast_1d(sample_idx):
                        v_neg = str(cand[int(idx)])
                        sjn = rank_map.get(v_neg)
                        if sjn is None:
                            continue
                        auc_scores2.append(float(si - sjn))
                        auc_labels2.append(0)
            used2 += 1
        except Exception:
            continue
    if used2 == 0:
        return None
    acc = float(np.mean(np.array(acc_preds2) == np.array(acc_obs2)))
    obs_arr = np.array(auc_labels2)
    if len(np.unique(obs_arr)) < 2:
        auc = 0.5
    else:
        try:
            from sklearn.metrics import roc_auc_score  # type: ignore
            auc = float(roc_auc_score(auc_labels2, auc_scores2))
        except Exception:
            # Fallback to a rough approximation using zero threshold on dv
            scores = np.array(auc_scores2)
            pos = obs_arr == 1
            neg = ~pos
            tpr = (scores[pos] > 0).mean() if pos.any() else 0.5
            fpr = (scores[neg] > 0).mean() if neg.any() else 0.5
            auc = 0.5 * (tpr + (1 - fpr))
    return acc, auc, used2

# ------------------------- High-level run ---------------------------------- #

def _write_multi(df: pd.DataFrame, base_path: str, formats: List[str]):
    if 'csv' in formats:
        df.to_csv(base_path + '.csv', index=False)
    if 'parquet' in formats:
        try:
            df.to_parquet(base_path + '.parquet', index=False)
        except Exception as e:
            print(f"[warn] parquet write failed: {e}")
    if 'json' in formats:
        df.to_json(base_path + '.json', orient='records')

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
    force_edges = cfg.get('edges',{}).get('force', False)
    progress = cfg['logging']['progress']
    raw_data_dir = cfg['paths']['raw_data_dir']
    output_dir = cfg['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    dry_run = cfg.get('dry_run', False)
    formats = cfg['output']['formats'] if 'output' in cfg else ['csv']
    caching_enabled = cfg.get('caching', {}).get('enabled', False)
    manifest_path = cfg.get('caching', {}).get('manifest', os.path.join(output_dir,'manifest.json'))
    manifest = _load_manifest(manifest_path) if caching_enabled else {"runs":{}}
    manifest.setdefault('signatures', {})
    cfg_sig = _config_signature(cfg)
    manifest['signatures']['last_config'] = cfg_sig

    if progress: print(f"[pipeline] Years: {years} (dry_run={dry_run})")
    ensure_scraped(years, raw_data_dir, force, progress)

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
                        _write_multi(pd.DataFrame(sorted_r, columns=['Player','Rank']), os.path.join(rank_dir, f"{y}_springrank"), formats)
                        if scale_req:
                            scaled = scale_ranks(A, raw_r)
                            scaled_sorted = [[node_list[i], r] for i,r in enumerate(scaled)]
                            scaled_sorted.sort(reverse=True, key=lambda x: x[1])
                            _write_multi(pd.DataFrame(scaled_sorted, columns=['Player','ScaledRank']), os.path.join(rank_dir,f"{y}_springrank_scaled"), formats)
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
                        G,A,node_list,_,test_edges = make_graph_from_edge_csv(edge_path, validation_folds=cfg['validation_folds'])
                        if not node_list:
                            if progress: print(f"[ranking] skip empty graph for {st}:{group}:inn{inn}:{y}")
                            continue
                        raw_r, sorted_r = spring_rank(A, node_list)
                        rank_dir = os.path.join(output_dir, st, group, str(inn))
                        os.makedirs(rank_dir, exist_ok=True)
                        _write_multi(pd.DataFrame(sorted_r, columns=['Player','Rank']), os.path.join(rank_dir, f"{y}_springrank"), formats)
                        if scale_req:
                            scaled = scale_ranks(A, raw_r)
                            scaled_sorted = [[node_list[i], r] for i,r in enumerate(scaled)]
                            scaled_sorted.sort(reverse=True, key=lambda x: x[1])
                            _write_multi(pd.DataFrame(scaled_sorted, columns=['Player','ScaledRank']), os.path.join(rank_dir,f"{y}_springrank_scaled"), formats)
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
                    else:
                        continue
                if not os.path.isfile(edge_path):
                    continue
                cache_key = cache_prefix
                file_sig = _file_signature(edge_path)
                if caching_enabled and cache_key in manifest['runs']:
                    prev = manifest['runs'][cache_key]
                    if prev.get('file_sig') == file_sig and prev.get('config_sig') == cfg_sig:
                        if progress: print(f"[cache] skip {cache_key}")
                        continue
                t0 = time.time()
                G,A,node_list,_,test_edges = make_graph_from_edge_csv(edge_path, validation_folds=cfg['validation_folds'])
                if not node_list:
                    if progress: print(f"[ranking] skip empty graph for {st}:{group}:{y}")
                    continue
                raw_r, sorted_r = spring_rank(A, node_list)
                rank_dir = os.path.join(output_dir, st, group)
                os.makedirs(rank_dir, exist_ok=True)
                _write_multi(pd.DataFrame(sorted_r, columns=['Player','Rank']), os.path.join(rank_dir, f"{y}_springrank"), formats)
                if scale_req:
                    scaled = scale_ranks(A, raw_r)
                    scaled_sorted = [[node_list[i], r] for i,r in enumerate(scaled)]
                    scaled_sorted.sort(reverse=True, key=lambda x: x[1])
                    _write_multi(pd.DataFrame(scaled_sorted, columns=['Player','ScaledRank']), os.path.join(rank_dir,f"{y}_springrank_scaled"), formats)
                    levels_records.append([st, group, None, y, max(scaled)-min(scaled)])
                validation_rows.append([st, group, None, y, len(node_list), A.count_nonzero(), float(A.count_nonzero())/(len(node_list)**2 if len(node_list)>0 else 1)])
                cv = cfg.get('validation_folds', 0)
                if cv and test_edges is not None:
                    val_cfg = cfg.get('validation', {})
                    res = _compute_acc_auc(
                        sorted_r,
                        test_edges,
                        auc_mode=str(val_cfg.get('aucMode','balanced-negatives')),
                        k_neg=int(val_cfg.get('negatives_per_positive', 1)),
                    )
                    if res:
                        acc, auc, used = res
                        auc_rows.append([st, group, None, y, cv, acc, auc, used])
                if caching_enabled:
                    manifest['runs'][cache_key] = {"time": time.time()-t0, "nodes": len(node_list), 'file_sig': file_sig, 'config_sig': cfg_sig}
                results_summary.append(pd.DataFrame(sorted_r, columns=['Player','Rank']).head(top_n).assign(Year=y, Group=group, ScoreType=st))

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
    # Cross-validated ACC/AUC report
    if auc_rows:
        auc_df = pd.DataFrame(auc_rows, columns=['ScoreType','Group','Condition','Year','Folds','Accuracy','AUC','TestEdges'])
        _write_multi(auc_df, os.path.join(output_dir,'validation_auc'), formats)
        if progress: print('[pipeline] validation_auc written')
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
                        if caching_enabled and cache_key in manifest['runs']:
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
                # base handmade/frequency
                if st == 'handmade':
                    edge_path = os.path.join('At Bats', f'{group}_data','handmade_scores', f"{y}_{group}_edges.csv")
                elif st == 'frequency':
                    edge_path = os.path.join('At Bats', f'{group}_data','frequency_scores', f"{y}_{group}_edges.csv")
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
