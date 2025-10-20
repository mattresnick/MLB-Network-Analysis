import os
import pandas as pd

def coverage(group: str, year: int, k: int) -> None:
    edges_path = os.path.join('At Bats', f'{group}_data', 'aware_scores', f'{year}_{group}_edges.csv')
    r_path = os.path.join('At Bats', f'{group}_data', 'aware_scores', f'{year}_R_nk_k{k}.csv')
    if not os.path.isfile(edges_path):
        print(f"missing edges: {edges_path}")
        return
    if not os.path.isfile(r_path):
        print(f"missing R_nk: {r_path}")
        return
    edf = pd.read_csv(edges_path, usecols=['winner','loser'])
    players = set(pd.unique(pd.concat([edf['winner'].astype(str), edf['loser'].astype(str)], ignore_index=True)))
    R = pd.read_csv(r_path)
    if 'Player' not in R.columns:
        print(f"R file missing Player column: {r_path} -> cols={list(R.columns)}")
        return
    Rplayers = set(R['Player'].astype(str))
    inter = players & Rplayers
    cov = (len(inter) / max(len(players), 1)) if players else 0.0
    print(f"{group} {year} K={k}: players={len(players)} Rplayers={len(Rplayers)} intersection={len(inter)} coverage={cov:.3f}")
    if cov < 0.9:
        missing = sorted(list(players - Rplayers))[:15]
        extra = sorted(list(Rplayers - players))[:15]
        print(" sample missing from R:", missing)
        print(" sample missing from edges:", extra)

if __name__ == '__main__':
    for g in ('batter','pitcher'):
        k = 150 if g=='batter' else 400
        coverage(g, 2024, k)
