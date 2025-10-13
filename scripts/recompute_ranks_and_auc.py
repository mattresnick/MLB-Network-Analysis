import os
import sys
import argparse
import pandas as pd
from typing import Tuple

# Load pipeline explicitly by file path to avoid import path issues
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PIPELINE_PATH = os.path.join(ROOT, 'pipeline.py')
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('pipeline', PIPELINE_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Unable to load pipeline module from {PIPELINE_PATH}")
pl = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(pl)  # type: ignore

def ensure_dirs(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def recompute_group(
    year: int,
    group: str,
    score_type: str = 'frequency',
    folds: int = 5,
    auc_mode: str = 'legacy',
    k_neg: int = 1,
) -> Tuple[float, float, int]:
    edge_path = os.path.join('At Bats', f'{group}_data', f'{score_type}_scores', f"{year}_{group}_edges.csv")
    if not os.path.isfile(edge_path):
        raise FileNotFoundError(edge_path)
    # Build graph with validation folds
    G, A, node_list, _, test_edges = pl.make_graph_from_edge_csv(edge_path, validation_folds=folds)
    if not node_list:
        print(f"[recompute] empty graph for {group} {year}")
        return (float('nan'), 0.5, 0)
    raw_r, sorted_r = pl.spring_rank(A, node_list)
    # Write ranks to outputs
    out_dir = os.path.join('outputs', score_type, group)
    ensure_dirs(out_dir)
    out_csv = os.path.join(out_dir, f"{year}_springrank.csv")
    pd.DataFrame(sorted_r, columns=['Player','Rank']).to_csv(out_csv, index=False)
    # Optional scaled ranks for consistency
    scaled = pl.scale_ranks(A, raw_r)
    pd.DataFrame([[node_list[i], float(scaled[i])] for i in range(len(node_list))], columns=['Player','ScaledRank'])\
        .sort_values('ScaledRank', ascending=False)\
        .to_csv(os.path.join(out_dir, f"{year}_springrank_scaled.csv"), index=False)
    # Compute ACC/AUC on held-out edges
    acc, auc, used = (float('nan'), 0.5, 0)
    if folds and test_edges is not None:
        res = pl._compute_acc_auc(sorted_r, test_edges, auc_mode=auc_mode, k_neg=k_neg)  # type: ignore
        if res:
            acc, auc, used = res
    print(f"[recompute] {score_type} {group} {year} (auc={auc_mode}, k_neg={k_neg}): ACC={acc:.4f} AUC={auc:.4f} TestEdges={used}")
    # Persist a record for later inspection
    try:
        import csv, time
        ensure_dirs(os.path.join('outputs','validation_auc_recompute.csv'))
        out_csv = os.path.join('outputs','validation_auc_recompute.csv')
        exists = os.path.isfile(out_csv)
        with open(out_csv, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(['Timestamp','ScoreType','Group','Year','AUCMode','KNeg','ACC','AUC','TestEdges'])
            w.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), score_type, group, year, auc_mode, k_neg, f"{acc:.6f}", f"{auc:.6f}", used])
    except Exception:
        pass
    return acc, auc, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--score_type', default='frequency')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--auc-mode', choices=['legacy','balanced-negatives'], default='legacy')
    ap.add_argument('--k-neg', type=int, default=1, help='negatives per positive for balanced-negatives mode')
    args = ap.parse_args()
    for group in ['batter','pitcher']:
        recompute_group(args.year, group, score_type=args.score_type, folds=args.folds, auc_mode=args.auc_mode, k_neg=args.k_neg)

if __name__ == '__main__':
    main()
