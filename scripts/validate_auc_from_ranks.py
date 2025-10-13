import os
import sys
import argparse
import pandas as pd
import numpy as np
from typing import List, Tuple

# This script computes a quick AUC using existing SpringRank outputs and the
# unipartite edge CSVs by treating all edges as positives and sampling negatives
# per source node (u). It is optimistic (ranks were learned on these edges).

def load_ranks(year: int, group: str, score_type: str = 'frequency') -> pd.DataFrame:
    path = os.path.join('outputs', score_type, group, f"{year}_springrank.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # Expect columns: Player, Rank
    return df

def load_edges(year: int, group: str, score_type: str = 'frequency') -> pd.DataFrame:
    path = os.path.join('At Bats', f'{group}_data', f'{score_type}_scores', f"{year}_{group}_edges.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # Expect columns: winner, loser, score
    return df[['winner','loser','score']]


def compute_auc_from_ranks(ranks_df: pd.DataFrame, edges_df: pd.DataFrame, k_neg: int = 1, seed: int = 42) -> Tuple[float, float, int]:
    # Build rank map
    rank_map = {str(n): float(r) for n, r in ranks_df[['Player','Rank']].itertuples(index=False, name=None)}
    # Prepare test_edges as numpy array tuples
    test_edges = edges_df[['winner','loser','score']].to_numpy()
    # Accuracy on positives; AUC using neg sampling
    acc_preds: List[int] = []
    acc_obs: List[int] = []
    auc_scores: List[float] = []
    auc_labels: List[int] = []
    used = 0
    # Build helpers for negatives per source
    test_pos_by_u = {}
    losers_pool = []
    for u, v, _ in test_edges:
        u = str(u); v = str(v)
        test_pos_by_u.setdefault(u, set()).add(v)
        losers_pool.append(v)
    unique_losers = np.array(sorted(set(losers_pool)))
    rng = np.random.RandomState(seed)
    for u, v, _ in test_edges:
        u = str(u); v = str(v)
        si = rank_map.get(u)
        sj = rank_map.get(v)
        if si is None or sj is None:
            continue
        dv = float(si - sj)
        acc_preds.append(1 if dv > 0 else 0)
        acc_obs.append(1)
        auc_scores.append(dv)
        auc_labels.append(1)
        if unique_losers.size > 1:
            ban = test_pos_by_u.get(u, set())
            mask = np.vectorize(lambda x: (x != v) and (x not in ban))(unique_losers)
            cand = unique_losers[mask]
            if cand.size > 0:
                sample_idx = rng.choice(np.arange(cand.size), size=min(k_neg, cand.size), replace=False)
                for idx in np.atleast_1d(sample_idx):
                    v_neg = str(cand[int(idx)])
                    sjn = rank_map.get(v_neg)
                    if sjn is None:
                        continue
                    auc_scores.append(float(si - sjn))
                    auc_labels.append(0)
        used += 1
    if used == 0:
        return (float('nan'), 0.5, 0)
    acc = float(np.mean(np.array(acc_preds) == np.array(acc_obs)))
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(auc_labels, auc_scores))
    except Exception:
        # Fallback: threshold at 0
        arr = np.array(auc_scores)
        lab = np.array(auc_labels)
        pos = lab == 1
        neg = ~pos
        tpr = (arr[pos] > 0).mean() if pos.any() else 0.5
        fpr = (arr[neg] > 0).mean() if neg.any() else 0.5
        auc = 0.5 * (tpr + (1 - fpr))
    return acc, auc, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--group', choices=['batter','pitcher'], required=True)
    ap.add_argument('--score_type', default='frequency')
    ap.add_argument('--kneg', type=int, default=1)
    args = ap.parse_args()
    ranks = load_ranks(args.year, args.group, args.score_type)
    edges = load_edges(args.year, args.group, args.score_type)
    acc, auc, used = compute_auc_from_ranks(ranks, edges, k_neg=args.kneg)
    print(f"[AUC-from-ranks] {args.score_type} {args.group} {args.year}: ACC={acc:.4f} AUC={auc:.4f} Positives={used}")

if __name__ == '__main__':
    main()
