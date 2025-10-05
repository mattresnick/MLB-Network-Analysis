"""Utilities to convert batter/pitcher bipartite edges into unipartite graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def split_by_winner(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    batter_edges = df[df.who_won == "batter"][['winner', 'loser', 'score']].copy()
    pitcher_edges = df[df.who_won == "pitcher"][['winner', 'loser', 'score']].copy()
    return batter_edges, pitcher_edges


def compute_unipartite_edges(group_df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the original nested-loop logic while returning a DataFrame."""

    winners = np.unique(group_df['winner'].to_numpy())
    losers = np.unique(group_df['loser'].to_numpy())
    players = np.unique(np.hstack((winners, losers)))

    group_df = group_df.sort_values(by=['winner', 'loser'])
    group_array = group_df.to_numpy()

    player_edgelist = []
    for idx_a, player_a in enumerate(players):
        player_a_edges = group_array[group_array[:, 0] == player_a]

        for idx_b, player_b in enumerate(players):
            if idx_a == idx_b:
                continue

            player_b_edges = group_array[group_array[:, 0] == player_b]

            common_opponents = np.intersect1d(player_a_edges[:, 1], player_b_edges[:, 1])
            if len(common_opponents) == 0:
                continue

            a_mask = np.isin(player_a_edges[:, 1], common_opponents)
            b_mask = np.isin(player_b_edges[:, 1], common_opponents)

            a_scores = player_a_edges[a_mask][:, 2]
            b_scores = (-1) * player_b_edges[b_mask][:, 2]

            score_diffs = a_scores + b_scores
            relu_diffs = np.where(score_diffs >= 0, score_diffs, 0)
            total_score = float(np.sum(relu_diffs))

            if total_score > 0:
                player_edgelist.append([player_a, player_b, total_score])

    return pd.DataFrame(player_edgelist, columns=['winner', 'loser', 'score'])


def force_no_parallel(df: pd.DataFrame) -> pd.DataFrame:
    edge_array = df[['winner', 'loser', 'score']].to_numpy()
    reduced_edges = []
    visited = set()

    for winner, loser, score in edge_array:
        key = (winner, loser)
        if key in visited:
            continue

        reciprocal_mask = (df['winner'] == loser) & (df['loser'] == winner)
        reciprocal_exists = reciprocal_mask.any()

        if reciprocal_exists:
            reciprocal_score = df.loc[reciprocal_mask, 'score'].iloc[0]
            if score > reciprocal_score:
                reduced_edges.append([winner, loser, float(score - reciprocal_score)])
            elif score < reciprocal_score:
                reduced_edges.append([winner, loser, 0.0])
            else:
                reduced_edges.append([winner, loser, float(score)])
                reduced_edges.append([loser, winner, float(reciprocal_score)])
            visited.add((loser, winner))
        else:
            reduced_edges.append([winner, loser, float(score)])

        visited.add(key)

    return pd.DataFrame(reduced_edges, columns=['winner', 'loser', 'score'])


def convert_bipartite(
    df: pd.DataFrame,
    *,
    apply_parallel_reduction: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    batter_group, pitcher_group = split_by_winner(df)
    batter_unipartite = compute_unipartite_edges(batter_group)
    pitcher_unipartite = compute_unipartite_edges(pitcher_group)

    if apply_parallel_reduction:
        batter_unipartite = force_no_parallel(batter_unipartite)
        pitcher_unipartite = force_no_parallel(pitcher_unipartite)

    return batter_unipartite, pitcher_unipartite


def convert_and_save(
    input_path: Path,
    *,
    batter_output: Optional[Path] = None,
    pitcher_output: Optional[Path] = None,
    apply_parallel_reduction: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(input_path)
    batter_df, pitcher_df = convert_bipartite(df, apply_parallel_reduction=apply_parallel_reduction)

    if batter_output:
        batter_output.parent.mkdir(parents=True, exist_ok=True)
        batter_df.to_csv(batter_output, index=False)
    if pitcher_output:
        pitcher_output.parent.mkdir(parents=True, exist_ok=True)
        pitcher_df.to_csv(pitcher_output, index=False)

    return batter_df, pitcher_df


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Convert bipartite edges to unipartite graphs")
    parser.add_argument("input", type=Path, help="CSV produced by add_edgeinfo.generate_edge_tables")
    parser.add_argument("--batter-output", type=Path, help="Where to save the batter graph edges")
    parser.add_argument("--pitcher-output", type=Path, help="Where to save the pitcher graph edges")
    parser.add_argument(
        "--skip-parallel-reduction",
        action="store_true",
        help="Keep reciprocal edges instead of collapsing them",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    convert_and_save(
        args.input,
        batter_output=args.batter_output,
        pitcher_output=args.pitcher_output,
        apply_parallel_reduction=not args.skip_parallel_reduction,
    )


if __name__ == "__main__":
    main()
