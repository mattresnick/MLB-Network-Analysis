#!/usr/bin/env python3
import argparse, os, pandas as pd

def write_rnk(raw_path: str, group: str, year: int, k: int):
    raw = pd.read_csv(raw_path)
    col = 'batter_name' if group == 'batter' else 'pitcher_name'
    c = raw[col].astype(str).str.strip()
    # Robust across pandas versions: set axis name before reset_index
    n = c.value_counts().rename_axis('Player').reset_index(name='n')
    n['R_nk'] = n['n'].astype(float) / (n['n'].astype(float) + float(k))
    out_base = os.path.join('At Bats', f'{group}_data', 'aware_scores')
    os.makedirs(out_base, exist_ok=True)
    path = os.path.join(out_base, f"{year}_R_nk_k{int(k)}.csv")
    n[['Player','R_nk','n']].to_csv(path, index=False)
    print(f"wrote {path} players={len(n)}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--batter', type=int)
    ap.add_argument('--pitcher', type=int)
    args = ap.parse_args()
    raw_path = os.path.join('At Bats','general_data', f"at_bat_data_{args.year}.csv")
    if not os.path.isfile(raw_path):
        raise SystemExit(f"missing raw {raw_path}")
    if args.batter:
        write_rnk(raw_path, 'batter', args.year, int(args.batter))
    if args.pitcher:
        write_rnk(raw_path, 'pitcher', args.year, int(args.pitcher))
