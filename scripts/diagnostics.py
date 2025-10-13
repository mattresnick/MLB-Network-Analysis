from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def _resolve_winner_name(df: pd.DataFrame, name: str) -> str | None:
    w = df['winner'].astype(str)
    # Exact first
    if (w == name).any():
        return name
    # Case-insensitive
    m = w.str.casefold() == name.casefold()
    if m.any():
        return w[m].iloc[0]
    # Contains (may be multiple)
    c = w.str.contains(name, case=False, na=False)
    if c.any():
        # Choose the variant with highest total score as a reasonable default
        cand = df[c].groupby('winner', as_index=False)['score'].sum().sort_values('score', ascending=False)
        return cand['winner'].iloc[0]
    return None


def compare_batters(
    year: int,
    batter_a: str,
    batter_b: str,
    *,
    raw_data_dir: str = 'At Bats/general_data',
    edges_type: str = 'frequency',
    use_rate: bool = False,
) -> pd.DataFrame:
    """Compare two batters against their common pitcher opponents.

    Returns a DataFrame with per-pitcher contributions to the i->j edge math.

    Columns:
    - pitcher: common opponent
    - score_a: A's summed batter-win score vs pitcher (or per-PA rate if use_rate)
    - score_b: B's summed batter-win score vs pitcher (or per-PA rate if use_rate)
    - diff_relu: max(score_a - score_b, 0)
    - diff_raw: score_a - score_b (signed)
    - a_pas/b_pas (only if use_rate): plate appearances used for the rate
    """
    edges_path = Path(raw_data_dir) / edges_type / f"{year}_edges_only.csv"
    if not edges_path.is_file():
        raise FileNotFoundError(edges_path)
    df = pd.read_csv(edges_path)
    df = df[df['who_won'] == 'batter'][['winner','loser','score']]

    # Resolve names robustly
    ra = _resolve_winner_name(df, batter_a) or batter_a
    rb = _resolve_winner_name(df, batter_b) or batter_b
    if ra != batter_a:
        print(f"[diag] Using '{ra}' for batter A (matched from '{batter_a}')")
    if rb != batter_b:
        print(f"[diag] Using '{rb}' for batter B (matched from '{batter_b}')")

    a = df[df['winner'] == ra].groupby('loser', as_index=False)['score'].sum().rename(columns={'score':'score_a','loser':'pitcher'})
    b = df[df['winner'] == rb].groupby('loser', as_index=False)['score'].sum().rename(columns={'score':'score_b','loser':'pitcher'})
    merged = a.merge(b, on='pitcher', how='inner')

    # Quick visibility when intersection is empty
    if merged.empty:
        a_ct = len(a)
        b_ct = len(b)
        print(f"[diag] No common pitchers. A pitchers={a_ct}, B pitchers={b_ct}")
        print("[diag] Sample A pitchers:", a.head(10).to_string(index=False))
        print("[diag] Sample B pitchers:", b.head(10).to_string(index=False))
        return merged

    if use_rate:
        # Compute per-PA rates for each batter vs pitcher from raw at-bat data
        raw_csv = Path(raw_data_dir) / f"at_bat_data_{year}.csv"
        if not raw_csv.is_file():
            raise FileNotFoundError(raw_csv)
        raw = pd.read_csv(raw_csv, usecols=['batter_name','pitcher_name','events'])
        # Count PAs where event is present (already filtered in scraper)
        a_pas = raw[raw['batter_name'] == batter_a].groupby('pitcher_name', as_index=False)['events'].count().rename(columns={'events':'a_pas','pitcher_name':'pitcher'})
        b_pas = raw[raw['batter_name'] == batter_b].groupby('pitcher_name', as_index=False)['events'].count().rename(columns={'events':'b_pas','pitcher_name':'pitcher'})
        merged = merged.merge(a_pas, on='pitcher', how='left').merge(b_pas, on='pitcher', how='left')
        merged['a_pas'] = merged['a_pas'].fillna(0).astype(int)
        merged['b_pas'] = merged['b_pas'].fillna(0).astype(int)
        # Avoid divide by zero; if no PAs, keep score as 0 to avoid inflating
        merged['score_a'] = merged.apply(lambda r: (r['score_a'] / r['a_pas']) if r['a_pas'] > 0 else 0.0, axis=1)
        merged['score_b'] = merged.apply(lambda r: (r['score_b'] / r['b_pas']) if r['b_pas'] > 0 else 0.0, axis=1)

    merged['diff_raw'] = merged['score_a'] - merged['score_b']
    merged['diff_relu'] = merged['diff_raw'].clip(lower=0)
    # Sort by largest contributions where A beats B
    merged = merged.sort_values('diff_relu', ascending=False).reset_index(drop=True)
    return merged


def main():
    ap = argparse.ArgumentParser(description='Diagnose batter-vs-batter unipartite edges via common pitchers')
    ap.add_argument('--year', type=int, required=True)
    ap.add_argument('--a', dest='batter_a', required=True, help='Batter A name (as in CSV)')
    ap.add_argument('--b', dest='batter_b', required=True, help='Batter B name (as in CSV)')
    ap.add_argument('--rate', action='store_true', help='Use per-PA rate vs pitcher instead of sums')
    ap.add_argument('--raw-dir', default='At Bats/general_data')
    args = ap.parse_args()
    df = compare_batters(args.year, args.batter_a, args.batter_b, raw_data_dir=args.raw_dir, use_rate=args.rate)
    if df.empty:
        print("[diag] No common pitchers found or no matches for batter names in edges_only. Try checking name spelling/casing or another opponent.")
        return
    # Print a compact summary
    total_relu = df['diff_relu'].sum()
    total_raw = df['diff_raw'].sum()
    print(f"A->B total (relu): {total_relu:.3f}   total (raw): {total_raw:.3f}  (rows={len(df)})")
    print(df.head(50).to_string(index=False))


if __name__ == '__main__':
    main()
