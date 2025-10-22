import json
import os
from typing import Any, Dict, List, Union, Tuple

ALLOWED_SCORE_TYPES = {"handmade", "frequency", "pitch_type", "inning", "aware"}
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
    # Default to aware-based scoring as primary workflow (strict aware-only unless user opts out)
    score_types = pipeline.get('score_types',["aware"]) or ["aware"]
    score_types = list(dict.fromkeys(score_types))  # preserve order unique
    for st in score_types:
        if st not in ALLOWED_SCORE_TYPES:
            raise ConfigError(f"invalid score_type '{st}'")
    # Optional strict flag: when true and 'aware' requested, ignore other score types
    strict_aware_only = bool(pipeline.get('enforce_aware_only', True))
    if strict_aware_only and ('aware' in score_types):
        score_types = ['aware']

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

    # Use 5-fold cross validation by default to mirror original scripts
    validation_folds = int(pipeline.get('validation_folds',5) or 5)
    if validation_folds < 0:
        raise ConfigError("validation_folds must be >=0")
    # Validation/AUC settings
    validation_cfg = pipeline.get('validation', {}) or {}
    # Accept any of the implemented modes; default to balanced-negatives going forward
    _requested_auc_mode = str(validation_cfg.get('aucMode', 'balanced-negatives') or 'balanced-negatives')
    if _requested_auc_mode not in {"legacy","balanced-negatives","pairwise-reversal"}:
        raise ConfigError("validation.aucMode must be 'legacy', 'balanced-negatives', or 'pairwise-reversal'")
    auc_mode = _requested_auc_mode
    negatives_per_positive = int(validation_cfg.get('negatives_per_positive', 1) or 1)
    if negatives_per_positive < 0:
        raise ConfigError('validation.negatives_per_positive must be >= 0')
    auto_flip = _bool(validation_cfg.get('auto_flip'), True)
    seed = validation_cfg.get('seed')
    if seed is not None:
        try:
            seed = int(seed)
        except Exception:
            raise ConfigError('validation.seed must be an integer')
    sample_as_train = _bool(validation_cfg.get('sample_as_train'), True)
    index_base = validation_cfg.get('index_base', 0)
    try:
        index_base = int(index_base)
    except Exception:
        raise ConfigError('validation.index_base must be 0 or 1')
    if index_base not in (0,1):
        raise ConfigError('validation.index_base must be 0 or 1')

    # Extended validation methods toggles
    extra_val = validation_cfg.get('extra', {}) or {}
    val_opponent_blockout = _bool(extra_val.get('opponent_blockout'), False)
    val_temperature_logloss = _bool(extra_val.get('temperature_logloss'), False)
    val_statcast_logloss = _bool(extra_val.get('statcast_logloss'), False)
    # Leak-free validation modes (edge_block, pa_block, loeo)
    modes_raw = validation_cfg.get('modes', {}) or {}
    if modes_raw and not isinstance(modes_raw, dict):
        raise ConfigError('validation.modes must be an object when provided')
    allowed_modes = {'edge_block','pa_block','loeo'}
    val_modes: Dict[str,bool] = {}
    for k, v in modes_raw.items():
        if k not in allowed_modes:
            raise ConfigError(f"validation.modes contains unknown key '{k}'")
        val_modes[k] = _bool(v, False)
    # Explicitly gate legacy/baseline metrics; default off
    val_baseline_auc = _bool(extra_val.get('baseline_auc'), False)
    val_only_baseline = _bool(extra_val.get('only_baseline'), False)
    # Optional switch to disable all legacy AUC except specified modes
    val_allow_legacy_auc = _bool(extra_val.get('allow_legacy_auc'), False)

    ranking = pipeline.get('ranking',{}) or {}
    top_n = int(ranking.get('top_n',25) or 25)
    scale_ranks = _bool(ranking.get('scale_ranks'), True)
    output_levels = _bool(ranking.get('output_levels'), True)
    # Optional: aware-specific tether lambda for ranking
    try:
        aware_lambda = float(ranking.get('aware_lambda', 1.0))
    except Exception:
        raise ConfigError('ranking.aware_lambda must be a number')
    # Optional: aware-specific harmonic scaling toggle (Step 2)
    aware_harmonic = _bool(ranking.get('aware_harmonic'), True)
    # Optional: aware-specific shrinkage toggle (Step 1)
    aware_shrink = _bool(ranking.get('aware_shrink'), True)
    # Whether to include covariates (chi term) in the aware ridge fit
    aware_use_covariates = _bool(ranking.get('aware_use_covariates'), True)
    # Optional: use only basic covariates (stand, p_throws); drop home/park and pitch count
    aware_covariates_basic = _bool(ranking.get('aware_covariates_basic'), False)
    # Optional: aware shrink mode and parameter k for n/(n+k)
    aware_shrink_mode = ranking.get('aware_shrink_mode', 'se_based') or 'se_based'
    if aware_shrink_mode not in {'se_based','n_over_n_plus_k'}:
        raise ConfigError("ranking.aware_shrink_mode must be 'se_based' or 'n_over_n_plus_k'")
    try:
        aware_shrink_k = int(ranking.get('aware_shrink_k', 150) or 150)
    except Exception:
        raise ConfigError('ranking.aware_shrink_k must be an integer')
    # Optional per-group ks
    try:
        aware_shrink_k_batter = ranking.get('aware_shrink_k_batter')
        aware_shrink_k_batter = int(aware_shrink_k_batter) if aware_shrink_k_batter is not None else aware_shrink_k
    except Exception:
        raise ConfigError('ranking.aware_shrink_k_batter must be an integer if provided')
    try:
        aware_shrink_k_pitcher = ranking.get('aware_shrink_k_pitcher')
        aware_shrink_k_pitcher = int(aware_shrink_k_pitcher) if aware_shrink_k_pitcher is not None else aware_shrink_k
    except Exception:
        raise ConfigError('ranking.aware_shrink_k_pitcher must be an integer if provided')

    scrape = pipeline.get('scrape',{}) or {}
    force_scrape = _bool(scrape.get('force'), False)
    skip_scrape = _bool(scrape.get('skip'), False)
    # Edge generation settings
    edges_cfg = pipeline.get('edges', {}) or {}
    force_edges = _bool(edges_cfg.get('force'), False)
    aware_topk_per_node = edges_cfg.get('aware_topk_per_node')
    if aware_topk_per_node is not None:
        try:
            aware_topk_per_node = int(aware_topk_per_node)
        except Exception:
            raise ConfigError('edges.aware_topk_per_node must be an integer if provided')
        if aware_topk_per_node <= 0:
            aware_topk_per_node = None
    aware_min_quantile = edges_cfg.get('aware_min_quantile')
    if aware_min_quantile is not None:
        try:
            aware_min_quantile = float(aware_min_quantile)
        except Exception:
            raise ConfigError('edges.aware_min_quantile must be a float in [0,1) if provided')
        if not (0.0 <= aware_min_quantile < 1.0):
            raise ConfigError('edges.aware_min_quantile must be in [0,1)')
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
    # Scenario A/B controls and MLB-only leaderboard display
    scenarios_cfg = pipeline.get('scenarios', {}) or {}
    scenarioA_include_milb = _bool(scenarios_cfg.get('A_include_milb'), True)
    scenarioB_exclude_milb = _bool(scenarios_cfg.get('B_exclude_milb'), True)
    mlb_only_leaderboard = _bool(scenarios_cfg.get('mlb_only_leaderboard'), True)

    # Caching settings
    caching_cfg = pipeline.get('caching', {}) or {}
    caching_enabled = _bool(caching_cfg.get('enabled'), True)
    caching_manifest = caching_cfg.get('manifest', 'outputs/manifest.json')

    logging_cfg = raw.get('logging',{}) or {}
    log_level = logging_cfg.get('level','INFO')
    progress = _bool(logging_cfg.get('progress'), True)

    # Processing settings (performance-oriented, no information loss)
    processing_cfg = pipeline.get('processing', {}) or {}
    vectorized = _bool(processing_cfg.get('vectorized'), True)
    unipartite_metric = processing_cfg.get('unipartite_metric', 'sum') or 'sum'
    if unipartite_metric not in {'sum','rate'}:
        raise ConfigError("processing.unipartite_metric must be 'sum' or 'rate'")

    # Analysis block (mobility/anomalies placeholders)
    analysis_cfg = pipeline.get('analysis', {}) or {}
    mobility_cfg = analysis_cfg.get('mobility', {}) or {}
    mobility_enabled = _bool(mobility_cfg.get('enabled'), False)
    rolling_cfg = analysis_cfg.get('rolling', {}) or {}
    rolling_enabled = _bool(rolling_cfg.get('enabled'), False)
    rolling_window_sizes = rolling_cfg.get('windows',[3]) if rolling_enabled else []
    if rolling_enabled:
        if (not isinstance(rolling_window_sizes,list) or any((not isinstance(x,int) or x<2) for x in rolling_window_sizes)):
            raise ConfigError('analysis.rolling.windows must be list[int>=2]')
    # Additional analysis toggles (future): anomalies, rolling windows etc.
    anomalies_cfg = analysis_cfg.get('anomalies', {}) or {}
    anomalies_enabled = _bool(anomalies_cfg.get('enabled'), False)
    anomalies_method = anomalies_cfg.get('method','quantile') or 'quantile'
    if anomalies_method not in {'quantile','absolute'}:
        raise ConfigError("analysis.anomalies.method must be 'quantile' or 'absolute'")
    anomalies_quantile = float(anomalies_cfg.get('quantile',0.95) or 0.95)
    if not (0 < anomalies_quantile < 1):
        raise ConfigError('analysis.anomalies.quantile must be between 0 and 1')
    anomalies_abs_threshold = float(anomalies_cfg.get('abs_threshold',0.2) or 0.2)
    anomalies_min_players = int(anomalies_cfg.get('min_players',20) or 20)
    if anomalies_min_players < 5:
        raise ConfigError('analysis.anomalies.min_players must be >=5')

    # Rank correlation (year-over-year) analysis
    rankcorr_cfg = analysis_cfg.get('rank_correlation', {}) or {}
    rankcorr_enabled = _bool(rankcorr_cfg.get('enabled'), False)
    rankcorr_pairs = rankcorr_cfg.get('pairs', 'adjacent') or 'adjacent'
    if rankcorr_pairs not in {'adjacent','all'}:
        raise ConfigError("analysis.rank_correlation.pairs must be 'adjacent' or 'all'")
    # OPS correlation analysis (ranks vs OPS)
    ops_cfg = analysis_cfg.get('ops_correlation', {}) or {}
    ops_enabled = _bool(ops_cfg.get('enabled'), False)

    # Iterative rank-weighted re-pass analysis
    ir_cfg = analysis_cfg.get('iterative_rank_weighted', {}) or {}
    ir_enabled = _bool(ir_cfg.get('enabled'), False)
    ir_passes = int(ir_cfg.get('passes', 1) or 1)
    if ir_passes < 0:
        raise ConfigError('analysis.iterative_rank_weighted.passes must be >= 0')
    ir_metric = ir_cfg.get('metric', 'sum') or 'sum'
    if ir_metric not in {'sum','rate'}:
        raise ConfigError("analysis.iterative_rank_weighted.metric must be 'sum' or 'rate'")
    ir_eval_auc = _bool(ir_cfg.get('evaluate_auc'), True)

    # Next-year validation (train ranks on year y, validate predictions on y+1 edges)
    ny_cfg = analysis_cfg.get('next_year_validation', {}) or {}
    ny_enabled = _bool(ny_cfg.get('enabled'), False)

    # k-sweep analysis configuration (pass-through with light validation)
    ks_cfg = analysis_cfg.get('k_sweep', {}) or {}
    try:
        ks_enabled = _bool(ks_cfg.get('enabled'), False)
        ks_only = _bool(ks_cfg.get('only'), False)
    except Exception:
        ks_enabled = False
        ks_only = False
    ks_batter_k = ks_cfg.get('batter_k', []) or []
    ks_pitcher_k = ks_cfg.get('pitcher_k', []) or []
    ks_max_test_edges = ks_cfg.get('max_test_edges')
    if ks_max_test_edges is not None:
        try:
            ks_max_test_edges = int(ks_max_test_edges)
        except Exception:
            raise ConfigError('analysis.k_sweep.max_test_edges must be an integer if provided')
    # Validate lists if provided
    if ks_batter_k and (not isinstance(ks_batter_k, list) or any((not isinstance(x, (int,float))) for x in ks_batter_k)):
        raise ConfigError('analysis.k_sweep.batter_k must be a list of numbers (ints)')
    if ks_pitcher_k and (not isinstance(ks_pitcher_k, list) or any((not isinstance(x, (int,float))) for x in ks_pitcher_k)):
        raise ConfigError('analysis.k_sweep.pitcher_k must be a list of numbers (ints)')
    # extra_R optional
    ks_extra_R = ks_cfg.get('extra_R', {}) or {}
    if ks_extra_R and not isinstance(ks_extra_R, dict):
        raise ConfigError('analysis.k_sweep.extra_R must be an object when provided')
    # Normalize to ints
    ks_batter_k = [int(x) for x in ks_batter_k] if ks_batter_k else []
    ks_pitcher_k = [int(x) for x in ks_pitcher_k] if ks_pitcher_k else []

    cfg = {
        'years': years,
        'score_types': score_types,
        'pitch_types': pitch_types,
        'innings': innings,
        'filters': {'stand': stand, 'p_throws': p_throws},
        'algorithms': algorithms,
        'validation_folds': validation_folds,
    'ranking': {'top_n': top_n, 'scale_ranks': scale_ranks, 'output_levels': output_levels, 'aware_lambda': aware_lambda, 'aware_harmonic': aware_harmonic, 'aware_shrink': aware_shrink, 'aware_shrink_mode': aware_shrink_mode, 'aware_shrink_k': aware_shrink_k, 'aware_use_covariates': aware_use_covariates},
    'ranking': {'top_n': top_n, 'scale_ranks': scale_ranks, 'output_levels': output_levels, 'aware_lambda': aware_lambda, 'aware_harmonic': aware_harmonic, 'aware_shrink': aware_shrink, 'aware_shrink_mode': aware_shrink_mode, 'aware_shrink_k': aware_shrink_k, 'aware_shrink_k_batter': aware_shrink_k_batter, 'aware_shrink_k_pitcher': aware_shrink_k_pitcher, 'aware_use_covariates': aware_use_covariates, 'aware_covariates_basic': aware_covariates_basic},
        'validation': {
            'aucMode': auc_mode,
            'negatives_per_positive': negatives_per_positive,
            'auto_flip': auto_flip,
            'seed': seed,
            'sample_as_train': sample_as_train,
            'index_base': index_base,
            'modes': val_modes,
            'extra': {
                'opponent_blockout': val_opponent_blockout,
                'temperature_logloss': val_temperature_logloss,
                'statcast_logloss': val_statcast_logloss,
                'baseline_auc': val_baseline_auc,
                'only_baseline': val_only_baseline,
                'allow_legacy_auc': val_allow_legacy_auc
            }
        },
    'scrape': {'force': force_scrape, 'skip': skip_scrape},
    'edges': {'force': force_edges, 'aware_topk_per_node': aware_topk_per_node, 'aware_min_quantile': aware_min_quantile},
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
        'scenarios': {
            'A_include_milb': scenarioA_include_milb,
            'B_exclude_milb': scenarioB_exclude_milb,
            'mlb_only_leaderboard': mlb_only_leaderboard
        },
        'caching': {
            'enabled': caching_enabled,
            'manifest': caching_manifest
        },
        'logging': {'level': log_level, 'progress': progress},
    'processing': {'vectorized': vectorized, 'unipartite_metric': unipartite_metric},
        'allow_2020': allow_2020,
    'enforce_aware_only': strict_aware_only,
        'analysis': {
            'mobility': {'enabled': mobility_enabled},
            'anomalies': {
                'enabled': anomalies_enabled,
                'method': anomalies_method,
                'quantile': anomalies_quantile,
                'abs_threshold': anomalies_abs_threshold,
                'min_players': anomalies_min_players
            },
            'rolling': {
                'enabled': rolling_enabled,
                'windows': rolling_window_sizes
            },
            'rank_correlation': {
                'enabled': rankcorr_enabled,
                'pairs': rankcorr_pairs
            },
            'ops_correlation': {
                'enabled': ops_enabled
            },
            'iterative_rank_weighted': {
                'enabled': ir_enabled,
                'passes': ir_passes,
                'metric': ir_metric,
                'evaluate_auc': ir_eval_auc
            },
            'next_year_validation': {
                'enabled': ny_enabled
            },
            'k_sweep': {
                'enabled': ks_enabled,
                'only': ks_only,
                'batter_k': ks_batter_k,
                'pitcher_k': ks_pitcher_k,
                'extra_R': ks_extra_R,
                'max_test_edges': ks_max_test_edges
            }
        }
    }
    return cfg

if __name__ == '__main__':
    import argparse, pprint
    ap = argparse.ArgumentParser(description='Load and validate MLB Network Analysis config')
    ap.add_argument('--config','-c', required=True)
    args = ap.parse_args()
    pprint.pprint(load_config(args.config))
