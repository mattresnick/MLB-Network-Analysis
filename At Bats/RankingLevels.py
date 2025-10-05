"""SpringRank-based utilities for computing player ordering levels."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import brentq


_SR_MODULE = None


def _springrank_module():
    global _SR_MODULE
    if _SR_MODULE is None:
        try:
            _SR_MODULE = importlib.import_module("SpringRank.SpringRank")
        except ModuleNotFoundError:
            try:
                _SR_MODULE = importlib.import_module("SpringRank")
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "SpringRank package is required. Install with pip install springrank."
                ) from exc
    return _SR_MODULE


def get_scaled_ranks(A, ranks, a, b, scale=0.75):
    """Scale raw SpringRank scores into an interpretable range."""

    sr = _springrank_module()
    inverse_temperature = brentq(sr.eqs39, a, b, args=(ranks, A))
    scaling_factor = 1 / (np.log(scale / (1 - scale)) / (2 * inverse_temperature))
    scaled_ranks = sr.scale_ranks(ranks, scaling_factor)
    return scaled_ranks


def build_graph_from_frame(df: pd.DataFrame, *, weights: bool = True):
    edge_list = df[["winner", "loser", "score"]].to_numpy()
    G = nx.DiGraph()

    if weights:
        G.add_weighted_edges_from(edge_list)
    else:
        G.add_edges_from(edge_list[:, :2])

    node_list = list(G.nodes())
    A = nx.to_scipy_sparse_matrix(G, dtype=float, nodelist=node_list)
    return G, A, node_list


def compute_springrank(df: pd.DataFrame, *, weights: bool = True):
    G, A, node_list = build_graph_from_frame(df, weights=weights)
    sr = _springrank_module()
    sr_rank = sr.SpringRank(A, alpha=0)
    sr_sorted = [[node_list[i], r] for i, r in enumerate(sr_rank)]
    sr_sorted.sort(reverse=True, key=lambda x: x[1])
    return sr_rank, sr_sorted, node_list, A


def compute_levels(
    df: pd.DataFrame,
    *,
    weights: bool = True,
    scale: float = 0.75,
    bracket: Sequence[float] = (0.01, 20),
    return_scaled: bool = False,
) -> Tuple[float, Iterable[Sequence[float]], Optional[pd.DataFrame]]:
    sr_rank, sr_sorted, node_list, A = compute_springrank(df, weights=weights)
    scaled_ranks = get_scaled_ranks(A, sr_rank, bracket[0], bracket[1], scale=scale)
    level_span = float(np.max(scaled_ranks) - np.min(scaled_ranks))

    scaled_df: Optional[pd.DataFrame] = None
    if return_scaled:
        scaled_df = pd.DataFrame(
            [[node_list[i], scaled_ranks[i]] for i in range(len(node_list))],
            columns=["player", "scaled_rank"],
        ).sort_values(by="scaled_rank", ascending=False)

    return level_span, sr_sorted, scaled_df


def save_levels(
    level_span: float,
    sr_sorted: Iterable[Sequence[float]],
    *,
    output_dir: Path,
    year_label: str,
    scaled_df: Optional[pd.DataFrame] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    levels_path = output_dir / "levels_by_year.csv"

    data = pd.DataFrame([[year_label, level_span]], columns=["Year", "SpringRank Levels"])
    if levels_path.exists():
        existing = pd.read_csv(levels_path)
        combined = pd.concat([existing, data], ignore_index=True)
    else:
        combined = data
    combined.to_csv(levels_path, index=False)

    if scaled_df is not None:
        scaled_df.to_csv(output_dir / f"scaled_ranks_{year_label}.csv", index=False)

    ranks_df = pd.DataFrame(sr_sorted, columns=["player", "spring_rank"])
    ranks_df.to_csv(output_dir / f"ranks_{year_label}.csv", index=False)


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Compute SpringRank levels from an edge list")
    parser.add_argument("input", type=Path, help="CSV with winner, loser, score columns")
    parser.add_argument("--year", type=str, default="unknown", help="Label used in the output CSVs")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write results to")
    parser.add_argument(
        "--store-scaled",
        action="store_true",
        help="Persist scaled SpringRank values per player",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    df = pd.read_csv(args.input)
    level_span, sr_sorted, scaled_df = compute_levels(
        df,
        return_scaled=args.store_scaled,
    )
    save_levels(
        level_span,
        sr_sorted,
        output_dir=args.output_dir,
        year_label=args.year,
        scaled_df=scaled_df,
    )


if __name__ == "__main__":
    main()
