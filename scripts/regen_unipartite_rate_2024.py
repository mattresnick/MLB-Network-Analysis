import os
from config.loader import load_config
from pipeline import to_2_unipartite

if __name__ == "__main__":
    cfg = load_config(os.path.join('config','example_2024.json'))
    raw_dir = cfg['paths']['raw_data_dir']
    year = 2024
    edge_only = os.path.join(raw_dir, 'frequency', f"{year}_edges_only.csv")
    out_b = os.path.join('At Bats','batter_data','frequency_scores', f"{year}_batter_edges.csv")
    out_p = os.path.join('At Bats','pitcher_data','frequency_scores', f"{year}_pitcher_edges.csv")
    print(f"[regen] rate/vectorized rebuild from {edge_only}\n       -> {out_b}\n       -> {out_p}")
    to_2_unipartite(edge_only, out_b, out_p, metric='rate', raw_data_dir=raw_dir)
    print("[regen] done")
