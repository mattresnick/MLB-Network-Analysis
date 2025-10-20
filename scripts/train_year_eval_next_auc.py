import os
import argparse
import pandas as pd
import numpy as np
from typing import Tuple

# Load pipeline explicitly to reuse SpringRank and evaluators
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
PIPELINE_PATH = os.path.join(ROOT, 'pipeline.py')
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('pipeline', PIPELINE_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Unable to load pipeline module from {PIPELINE_PATH}")
pl = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(pl)  # type: ignore


def load_edges(year: int, group: str, score_type: str = 'frequency') -> pd.DataFrame:
    path = os.path.join('At Bats', f'{group}_data', f'{score_type}_scores', f"{year}_{group}_edges.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)[['winner','loser','score']]


def fit_ranks_on_year(year: int, group: str, score_type: str = 'frequency'):
    df = load_edges(year, group, score_type)
    import networkx as nx
    G = nx.DiGraph()
    # Only train on positives (as usual)
    G.add_weighted_edges_from(df[['winner','loser','score']].itertuples(index=False, name=None))
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
    raw_r, sorted_r = pl.spring_rank(A, node_list)
    return raw_r, sorted_r


def eval_on_next_year(year_next: int, group: str, sorted_r) -> Tuple[float, float, int]:
    # Use balanced-negatives to get a robust AUC
    df_next = load_edges(year_next, group)
    test_edges = df_next[['winner','loser','score']].to_numpy()
    res = pl._compute_acc_auc(sorted_r, test_edges, auc_mode='balanced-negatives', k_neg=1, auto_flip=False)
    if not res:
        return (float('nan'), 0.5, 0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, required=True, help='train year')
    ap.add_argument('--score_type', default='frequency')
    args = ap.parse_args()
    rows = []
    for group in ['batter','pitcher']:
        _, sorted_r = fit_ranks_on_year(args.year, group, args.score_type)
        acc, auc, used = eval_on_next_year(args.year + 1, group, sorted_r)
        rows.append([args.score_type, group, args.year, args.year+1, acc, auc, used])
        print(f"[train-eval] {args.score_type} {group}: train {args.year} -> eval {args.year+1}: ACC={acc:.4f} AUC={auc:.4f} TestEdges={used}")
    out = os.path.join('outputs', f"train_eval_{args.year}_{args.year+1}.csv")
    os.makedirs('outputs', exist_ok=True)
    pd.DataFrame(rows, columns=['ScoreType','Group','TrainYear','EvalYear','Accuracy','AUC','TestEdges']).to_csv(out, index=False)


if __name__ == '__main__':
    main()
