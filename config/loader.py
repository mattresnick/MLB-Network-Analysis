import json
import os
from typing import Any, Dict, List, Union, Tuple

ALLOWED_SCORE_TYPES = {"handmade", "frequency", "pitch_type", "inning"}
ALLOWED_PITCH_TYPES = {"CH","CU","FC","FF","FS","FT","SI","SL"}
ALLOWED_INNINGS = set(range(1,10))

class ConfigError(Exception):
    pass

def _years_list(years_val: Union[List[int], Dict[str,int]]) -> List[int]:
    if isinstance(years_val, list):
        if not years_val:
            raise ConfigError("years list cannot be empty")
        for y in years_val:
            if not isinstance(y,int):
                raise ConfigError(f"year {y} is not int")
        return sorted(set(years_val))
    if isinstance(years_val, dict):
        try:
            start = int(years_val["start"]); end = int(years_val["end"])
        except (KeyError, ValueError):
            raise ConfigError("years object must have integer start and end")
        if end < start:
            raise ConfigError("years.end must be >= years.start")
        return list(range(start, end+1))
    raise ConfigError("years must be list[int] or {start:int,end:int}")

def _bool(x: Any, default: bool=False) -> bool:
    if x is None: return default
    if isinstance(x,bool): return x
    raise ConfigError(f"expected bool got {type(x)}")

def load_config(path: str) -> Dict[str,Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path,'r',encoding='utf-8') as f:
        raw = json.load(f)
    pipeline = raw.get('pipeline')
    if pipeline is None:
        raise ConfigError("missing 'pipeline' root key")

    years = _years_list(pipeline.get('years',[2019]))
    # Exclude 2020 by default unless explicitly allowed.
    allow_2020 = bool(pipeline.get('allow_2020', False))
    if not allow_2020 and 2020 in years:
        years = [y for y in years if y != 2020]
    score_types = pipeline.get('score_types',["handmade"]) or ["handmade"]
    score_types = list(dict.fromkeys(score_types))  # preserve order unique
    for st in score_types:
        if st not in ALLOWED_SCORE_TYPES:
            raise ConfigError(f"invalid score_type '{st}'")

    pitch_types = pipeline.get('pitch_types',[]) if 'pitch_type' in score_types else []
    for pt in pitch_types:
        if pt not in ALLOWED_PITCH_TYPES:
            raise ConfigError(f"invalid pitch_type '{pt}'")

    innings = pipeline.get('innings',[]) if 'inning' in score_types else []
    for inn in innings:
        if inn not in ALLOWED_INNINGS:
            raise ConfigError(f"invalid inning '{inn}' (1-9)")

    filters = pipeline.get('filters',{}) or {}
    stand = filters.get('stand')
    if stand is not None:
        if not isinstance(stand,list) or any(s not in {'L','R'} for s in stand):
            raise ConfigError("filters.stand must be [\"L\",\"R\"] subset")
    p_throws = filters.get('p_throws')
    if p_throws is not None:
        if not isinstance(p_throws,list) or any(s not in {'L','R'} for s in p_throws):
            raise ConfigError("filters.p_throws must be [\"L\",\"R\"] subset")

    # Algorithms field kept for backwards compatibility but only SpringRank is used now.
    algorithms = pipeline.get('algorithms', {'springrank': True}) or {'springrank': True}
    # Force to only springrank to avoid confusion.
    algorithms = {'springrank': True}

    validation_folds = int(pipeline.get('validation_folds',0) or 0)
    if validation_folds < 0:
        raise ConfigError("validation_folds must be >=0")

    ranking = pipeline.get('ranking',{}) or {}
    top_n = int(ranking.get('top_n',25) or 25)
    scale_ranks = _bool(ranking.get('scale_ranks'), True)
    output_levels = _bool(ranking.get('output_levels'), True)

    scrape = pipeline.get('scrape',{}) or {}
    force_scrape = _bool(scrape.get('force'), False)
    dry_run = _bool(pipeline.get('dry_run'), False)

    paths = pipeline.get('paths',{}) or {}
    # Provide defaults relative to repo root (caller's cwd expected root).
    raw_data_dir = paths.get('raw_data_dir','At Bats/general_data')
    batter_dir = paths.get('batter_dir','At Bats/batter_data')
    pitcher_dir = paths.get('pitcher_dir','At Bats/pitcher_data')
    intermediate_dir = paths.get('intermediate_dir','At Bats/intermediate_results')
    output_dir = paths.get('output_dir','outputs')

    output = pipeline.get('output',{}) or {}
    formats = output.get('formats',["csv"]) or ["csv"]
    include_edge_lists = _bool(output.get('include_edge_lists'), True)
    include_rank_tables = _bool(output.get('include_rank_tables'), True)
    include_scaled_rank_tables = _bool(output.get('include_scaled_rank_tables'), True)

    # Caching settings
    caching_cfg = pipeline.get('caching', {}) or {}
    caching_enabled = _bool(caching_cfg.get('enabled'), True)
    caching_manifest = caching_cfg.get('manifest', 'outputs/manifest.json')

    logging_cfg = raw.get('logging',{}) or {}
    log_level = logging_cfg.get('level','INFO')
    progress = _bool(logging_cfg.get('progress'), True)

    # Analysis block (mobility/anomalies placeholders)
    analysis_cfg = pipeline.get('analysis', {}) or {}
    mobility_cfg = analysis_cfg.get('mobility', {}) or {}
    mobility_enabled = _bool(mobility_cfg.get('enabled'), False)
    # Additional analysis toggles (future): anomalies, rolling windows etc.
    anomalies_cfg = analysis_cfg.get('anomalies', {}) or {}
    anomalies_enabled = _bool(anomalies_cfg.get('enabled'), False)

    cfg = {
        'years': years,
        'score_types': score_types,
        'pitch_types': pitch_types,
        'innings': innings,
        'filters': {'stand': stand, 'p_throws': p_throws},
        'algorithms': algorithms,
        'validation_folds': validation_folds,
        'ranking': {'top_n': top_n, 'scale_ranks': scale_ranks, 'output_levels': output_levels},
    'scrape': {'force': force_scrape},
    'dry_run': dry_run,
        'paths': {
            'raw_data_dir': raw_data_dir,
            'batter_dir': batter_dir,
            'pitcher_dir': pitcher_dir,
            'intermediate_dir': intermediate_dir,
            'output_dir': output_dir
        },
        'output': {
            'formats': formats,
            'include_edge_lists': include_edge_lists,
            'include_rank_tables': include_rank_tables,
            'include_scaled_rank_tables': include_scaled_rank_tables
        },
        'caching': {
            'enabled': caching_enabled,
            'manifest': caching_manifest
        },
        'logging': {'level': log_level, 'progress': progress},
        'allow_2020': allow_2020,
        'analysis': {
            'mobility': {'enabled': mobility_enabled},
            'anomalies': {'enabled': anomalies_enabled}
        }
    }
    return cfg

if __name__ == '__main__':
    import argparse, pprint
    ap = argparse.ArgumentParser(description='Load and validate MLB Network Analysis config')
    ap.add_argument('--config','-c', required=True)
    args = ap.parse_args()
    pprint.pprint(load_config(args.config))
