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
from SpringRank import SpringRank as sr  # type: ignore

# Reuse logic from Rankings.py minimally (avoid import side-effects)
# (Potential improvement: refactor Rankings.py into functions and import.)

ALLOWED_PITCH_TYPES = ["CH","CU","FC","FF","FS","FT","SI","SL"]

# Base handcrafted scoring (from add_edgeinfo.py) and base pitcher scoring
BASE_BATTER_SCORING = {'hit_by_pitch':1,'walk':2,'single':3,'double':6,'triple':9,'home_run':12}
BASE_PITCHER_SCORING = {'fielders_choice':1,'fielders_choice_out':1,'other_out':1,'field_out':1,'force_out':2,'grounded_into_double_play':2,'strikeout':6}

# ------------------------- Scraping ---------------------------------------- #

def season_date_range(year: int) -> Tuple[str,str]:
    # Using existing hard-coded ranges as in original at_bat_scraper.py
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
    if year not in ranges:
        raise ValueError(f"No season date range for {year}")
    return ranges[year]

SCRAPE_COLUMNS = ['pitch_type','player_name','batter','events','description',
                  'home_team','away_team','inning','stand','p_throws',
                  'home_score','away_score']


def scrape_year(year: int, out_dir: str, force: bool=False, progress: bool=True) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"at_bat_data_{year}.csv")
    if os.path.isfile(out_path) and not force:
        if progress: print(f"[scrape] {year} exists -> skip")
        return out_path
    start, end = season_date_range(year)
    if progress: print(f"[scrape] Fetching Statcast {year} {start}..{end}")
    data = pyb.statcast(start, end)  # can be large
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


# ------------------------- Edge Generation -------------------------------- #

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

def ensure_edge_only(year: int, score_type: str, raw_data_dir: str, progress: bool, pitch_types=None, innings=None, stand_filter=None, pthrows_filter=None):
    """Create edge-only files under general_data/<type>/... if missing.
    score_type in {handmade, frequency, pitch_type, inning}
    """
    base_dir = raw_data_dir  # expected 'At Bats/general_data'
    raw_file = os.path.join(base_dir, f"at_bat_data_{year}.csv")
    if not os.path.isfile(raw_file):
        if progress: print(f"[edges] raw file missing {raw_file}")
        return []
    df = pd.read_csv(raw_file)
    # Apply handedness filters if provided
    if stand_filter:
        df = df[df['stand'].isin(stand_filter)]
    if pthrows_filter:
        df = df[df['p_throws'].isin(pthrows_filter)]
    created = []
    if score_type == 'handmade':
        out_dir = os.path.join(base_dir, 'handmade')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{year}_edges_only.csv")
        if not os.path.isfile(out_path):
            b_dict = BASE_BATTER_SCORING
            p_dict = BASE_PITCHER_SCORING
            rows = []
            for batter_name, pitcher_name, event in df[['batter_name','player_name','events']].itertuples(index=False):
                r = _score_event(batter_name, pitcher_name, event, b_dict, p_dict)
                if r: rows.append(r)
            if rows:
                edf = pd.DataFrame(rows, columns=['winner','loser','score','who_won'])
                edf = edf.groupby(['winner','loser','who_won']).sum()
                edf.to_csv(out_path)
                created.append(out_path)
                if progress: print(f"[edges] wrote {out_path}")
        return created
    if score_type == 'frequency':
        out_dir = os.path.join(base_dir, 'frequency')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{year}_edges_only.csv")
        if not os.path.isfile(out_path):
            b_freq = _frequency_scaling(df, BASE_BATTER_SCORING)
            p_freq = _frequency_scaling(df, BASE_PITCHER_SCORING)
            rows = []
            for batter_name, pitcher_name, event in df[['batter_name','player_name','events']].itertuples(index=False):
                r = _score_event(batter_name, pitcher_name, event, b_freq, p_freq)
                if r: rows.append(r)
            if rows:
                edf = pd.DataFrame(rows, columns=['winner','loser','score','who_won'])
                edf = edf.groupby(['winner','loser','who_won']).sum()
                edf.to_csv(out_path)
                created.append(out_path)
                if progress: print(f"[edges] wrote {out_path}")
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
            if not os.path.isfile(out_path):
                rows = []
                pt_df = df[df['pitch_type']==pt]
                if pt_df.empty: continue
                for batter_name, pitcher_name, event in pt_df[['batter_name','player_name','events']].itertuples(index=False):
                    r = _score_event(batter_name, pitcher_name, event, BASE_BATTER_SCORING, BASE_PITCHER_SCORING)
                    if r: rows.append(r)
                if rows:
                    edf = pd.DataFrame(rows, columns=['winner','loser','score','who_won'])
                    edf = edf.groupby(['winner','loser','who_won']).sum()
                    edf.to_csv(out_path)
                    created.append(out_path)
                    if progress: print(f"[edges] wrote {out_path}")
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
            if not os.path.isfile(out_path):
                rows = []
                inn_df = df[df['inning']==inn]
                if inn_df.empty: continue
                for batter_name, pitcher_name, event in inn_df[['batter_name','player_name','events']].itertuples(index=False):
                    r = _score_event(batter_name, pitcher_name, event, BASE_BATTER_SCORING, BASE_PITCHER_SCORING)
                    if r: rows.append(r)
                if rows:
                    edf = pd.DataFrame(rows, columns=['winner','loser','score','who_won'])
                    edf = edf.groupby(['winner','loser','who_won']).sum()
                    edf.to_csv(out_path)
                    created.append(out_path)
                    if progress: print(f"[edges] wrote {out_path}")
        return created
    return []

def to_2_unipartite(edge_only_path: str, save_batter: str, save_pitcher: str):
    # Re-implement minimal variant of BipartiteTo2Unipartite.to2Unipartite for internal use
    df = pd.read_csv(edge_only_path)
    bwe = df[df.who_won=='batter'][['winner','loser','score']].sort_values(['winner','loser'])
    pwe = df[df.who_won=='pitcher'][['winner','loser','score']].sort_values(['winner','loser'])
    def group_edges(gwe_df, out_path):
        gwe_arr = gwe_df.to_numpy()
        group_players = np.unique(gwe_arr[:,0])
        # include players who only appeared as losers
        group_players = np.unique(np.hstack((group_players, np.unique(gwe_arr[:,1]))))
        player_edgelist = []
        # Quadratic complexity; acceptable for moderate node counts
        for i, p1 in enumerate(group_players):
            p1_arr_base = gwe_arr[gwe_arr[:,0]==p1]
            for j, p2 in enumerate(group_players):
                if i==j: continue
                p1_arr = p1_arr_base.copy()
                p2_arr = gwe_arr[gwe_arr[:,0]==p2]
                p2_arr = p2_arr[[k for k,x in enumerate(p2_arr[:,1]) if x in p1_arr[:,1]]]
                p1_arr = p1_arr[[k for k,x in enumerate(p1_arr[:,1]) if x in p2_arr[:,1]]]
                p2_arr[:,2] = p2_arr[:,2]*(-1)
                score_diffs = np.add(p1_arr[:,2], p2_arr[:,2])
                relu = np.where(score_diffs>=0, score_diffs, 0)
                total = np.sum(relu)
                player_edgelist.append([p1,p2,total])
        edf = pd.DataFrame(player_edgelist, columns=['winner','loser','score'])
        edf.to_csv(out_path, index=False)
    group_edges(bwe, save_batter)
    group_edges(pwe, save_pitcher)


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
    edge_list = df.to_numpy()[:,1:]
    G = nx.DiGraph()
    if validation_folds>0:
        m = len(edge_list)
        sel_inds = np.random.choice(list(range(m)), int(m*(1-(1/validation_folds))), replace=False)
        not_sel = np.setdiff1d(list(range(m)), sel_inds)
        train_edges = edge_list[sel_inds]
        test_edges = edge_list[not_sel]
    else:
        train_edges = edge_list
        test_edges = None
    if weights:
        G.add_weighted_edges_from(train_edges)
    else:
        G.add_edges_from(train_edges[:,:2])
    node_list = list(G.nodes())
    A = nx.to_scipy_sparse_matrix(G, dtype=float, nodelist=node_list)
    return G, A, node_list, train_edges, test_edges


def spring_rank(A, node_list: List[str]):
    sr_rank = sr.SpringRank(A, alpha=0)
    sr_sorted = [[node_list[i], r] for i,r in enumerate(sr_rank)]
    sr_sorted.sort(reverse=True, key=lambda x: x[1])
    return sr_rank, sr_sorted


def scale_ranks(A, raw_ranks, a=0.01, b=20, scale=0.75):
    from scipy.optimize import brentq
    from SpringRank import SpringRank as _sr
    inverse_temperature = brentq(_sr.eqs39, a, b, args=(raw_ranks, A))
    scaling_factor = 1 / (np.log(scale / (1 - scale)) / (2 * inverse_temperature))
    return _sr.scale_ranks(raw_ranks, scaling_factor)

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
    if not os.path.isfile(path):
        return {"runs":{}}
    try:
        with open(path,'r',encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {"runs":{}}

def _save_manifest(path: str, manifest: Dict[str,Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
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
                ensure_edge_only(y, st, raw_data_dir, progress, pitch_types=pitch_types, innings=innings, stand_filter=stand_filter, pthrows_filter=pthrows_filter)
    if dry_run:
        print("[dry-run] Planned edge generation:")
        for y,st in planned:
            print(f"  - {y}:{st}")
        return True

    # 2. Convert bipartite edges to unipartite group edges for each score type
    results_summary = []
    levels_records = []
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
                if os.path.isfile(edge_file) and not (os.path.isfile(b_edge_out) and os.path.isfile(p_edge_out)):
                    if progress: print(f"[unipartite] {y} {st}")
                    to_2_unipartite(edge_file, b_edge_out, p_edge_out)
            elif st == 'frequency':
                edge_file = os.path.join(raw_data_dir, 'frequency', f"{y}_edges_only.csv")
                inter_dir = os.path.join('At Bats','intermediate_results','frequency')
                os.makedirs(inter_dir, exist_ok=True)
                b_edge_out = os.path.join('At Bats','batter_data','frequency_scores', f"{y}_batter_edges.csv")
                p_edge_out = os.path.join('At Bats','pitcher_data','frequency_scores', f"{y}_pitcher_edges.csv")
                if os.path.isfile(edge_file) and not (os.path.isfile(b_edge_out) and os.path.isfile(p_edge_out)):
                    to_2_unipartite(edge_file, os.path.join(inter_dir,f"{y}_batter_edges.csv"), os.path.join(inter_dir,f"{y}_pitcher_edges.csv"))
                    to_2_unipartite(edge_file, b_edge_out, p_edge_out)  # reuse
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
                        t0 = time.time()
                        G,A,node_list,_,_ = make_graph_from_edge_csv(edge_path, validation_folds=cfg['validation_folds'])
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
                        G,A,node_list,_,_ = make_graph_from_edge_csv(edge_path, validation_folds=cfg['validation_folds'])
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
                G,A,node_list,_,_ = make_graph_from_edge_csv(edge_path, validation_folds=cfg['validation_folds'])
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
    return True

if __name__ == '__main__':
    import argparse
    from config.loader import load_config
    ap = argparse.ArgumentParser(description='Run MLB pipeline with JSON config')
    ap.add_argument('--config','-c', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_pipeline(cfg)
