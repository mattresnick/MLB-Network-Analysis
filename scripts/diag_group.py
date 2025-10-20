import argparse
import os
import pandas as pd
import numpy as np
from pipeline import make_graph_from_edge_csv, aware_rank_with_tether, _rank_orientation_fraction, _compute_acc_auc

def diag(group: str, year: int, k: int, seed: int = 123, folds: int = 5, aware_lambda: float = 1.0, auc_mode: str = 'balanced-negatives'):
    edge_path = os.path.join('At Bats', f'{group}_data', 'aware_scores', f'{year}_{group}_edges.csv')
    R_path = os.path.join('At Bats', f'{group}_data', 'aware_scores', f'{year}_R_nk_k{k}.csv')
    if not os.path.isfile(edge_path):
        print(f"missing edges: {edge_path}")
        return
    if not os.path.isfile(R_path):
        print(f"missing R: {R_path}")
        return
    R = pd.read_csv(R_path)
    R_map = dict(R[['Player','R_nk']].itertuples(index=False, name=None))
    G,A,node_list,train_edges,test_edges = make_graph_from_edge_csv(edge_path, validation_folds=folds, seed=seed, sample_as_train=True, index_base=0, fast=True)
    n_nodes = len(node_list)
    n_train = len(train_edges) if train_edges is not None else 0
    n_test = len(test_edges) if test_edges is not None else 0
    if not node_list or not test_edges:
        print(f"{group}: no test edges")
        return
    _, sorted_r = aware_rank_with_tether(A, node_list, R_map, lambda_reg=aware_lambda, use_harmonic=True)
    # Diagnostics
    orient_frac, orient_tot = _rank_orientation_fraction(sorted_r, list(test_edges))
    rank_std = float(np.std([s for _, s in sorted_r])) if sorted_r else 0.0
    res_bal = _compute_acc_auc(sorted_r, list(test_edges), auc_mode=auc_mode, k_neg=1, auto_flip=True, seed=seed, acc_mode='balanced')
    res_pos = _compute_acc_auc(sorted_r, list(test_edges), auc_mode=auc_mode, k_neg=1, auto_flip=True, seed=seed, acc_mode='positive-only')
    acc_bal = res_bal[0] if res_bal else float('nan')
    auc = res_bal[1] if res_bal else float('nan')
    acc_pos = res_pos[0] if res_pos else float('nan')
    print(f"group={group} year={year} K={k} mode={auc_mode} nodes={n_nodes} train={n_train} test={n_test} orient_frac={orient_frac:.4f} rank_std={rank_std:.5f} acc_pos={acc_pos:.5f} acc_bal={acc_bal:.5f} auc={auc:.5f}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', choices=['batter','pitcher'], required=True)
    ap.add_argument('--year', type=int, default=2024)
    ap.add_argument('--k', type=int)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--lambda', dest='lam', type=float, default=1.0)
    ap.add_argument('--aucMode', dest='auc_mode', type=str, default='balanced-negatives')
    args = ap.parse_args()
    if args.k is None:
        args.k = 150 if args.group=='batter' else 400
    diag(args.group, args.year, args.k, seed=args.seed, folds=args.folds, aware_lambda=args.lam, auc_mode=args.auc_mode)
