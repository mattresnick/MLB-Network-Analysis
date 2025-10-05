"""Helpers for translating raw at-bat data into weighted edges.

Historically this module executed large batch jobs when imported. The
functions below keep that behaviour opt-in so that the interactive
interface can compose smaller, parameterised workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from collections.abc import Iterable

import pandas as pd


@dataclass(frozen=True)
class ScoringScheme:
    batter: Dict[str, float]
    pitcher: Dict[str, float]


class Scorer:
    def __init__(self, scoring: ScoringScheme):
        self.b_dict = scoring.batter
        self.p_dict = scoring.pitcher

    def score_event(self, batter: str, pitcher: str, event: str):
        """Return winner/loser/score metadata for a single event.

        A positive score indicates the batter "won" the plate appearance.
        Negative scores are flipped so that the returned score is always
        positive while the who_won flag captures the perspective.
        """

        if event in self.b_dict:
            score = self.b_dict[event]
            winner, loser, who_won = batter, pitcher, "batter"
        elif event in self.p_dict:
            score = self.p_dict[event]
            winner, loser, who_won = pitcher, batter, "pitcher"
        else:
            return None

        return winner, loser, float(abs(score)), who_won


HANDCRAFTED_SCORES = ScoringScheme(
    batter={
        "hit_by_pitch": 1,
        "walk": 2,
        "single": 3,
        "double": 6,
        "triple": 9,
        "home_run": 12,
    },
    pitcher={
        "fielders_choice": 1,
        "fielders_choice_out": 1,
        "other_out": 1,
        "field_out": 1,
        "force_out": 2,
        "grounded_into_double_play": 2,
        "strikeout": 6,
    },
)


def result_frequency(df: pd.DataFrame, scoring: Dict[str, float]) -> Dict[str, float]:
    """Scale base scores by their frequency within *df*."""

    events = df["events"].dropna()
    total_results = len(events)
    if total_results == 0:
        return {key: 0.0 for key in scoring}

    scaled: Dict[str, float] = {}
    for event, base_score in scoring.items():
        event_count = (events == event).sum()
        scaled[event] = base_score * (event_count / total_results)
    return scaled


def build_edge_dataframe(
    df: pd.DataFrame,
    scorer: Scorer,
    *,
    drop_zero_scores: bool = True,
) -> pd.DataFrame:
    """Add winner/loser metadata to *df* using the provided *scorer*."""

    working = df.copy()
    pitcher_col = "pitcher_name" if "pitcher_name" in working.columns else "player_name"

    edge_payload = working[["batter_name", pitcher_col, "events"]].apply(
        lambda row: scorer.score_event(row[0], row[1], row[2]), axis=1
    )

    working["edge_info"] = edge_payload
    if drop_zero_scores:
        working = working[working["edge_info"].notna()].copy()

    working["winner"] = working["edge_info"].apply(lambda v: v[0])
    working["loser"] = working["edge_info"].apply(lambda v: v[1])
    working["score"] = working["edge_info"].apply(lambda v: v[2])
    working["who_won"] = working["edge_info"].apply(lambda v: v[3])

    working = working.drop(columns=["edge_info"])
    return working


def aggregate_edges(
    df: pd.DataFrame,
    *,
    slicer: Optional[str] = None,
    filters: Optional[Dict[str, Iterable]] = None,
) -> pd.DataFrame:
    """Collapse plate appearances into a weighted edge list.

    Parameters
    ----------
    df:
        DataFrame returned by :func:uild_edge_dataframe.
    slicer:
        Optional column to retain during aggregation (e.g. `"pitch_type"` or
        `"inning"`). When provided we keep one CSV per distinct slicer value in
        the downstream pipeline.
    """

    group_fields = []
    if slicer:
        group_fields.append(slicer)
    group_fields.extend(["winner", "loser", "who_won"])

    grouped = (
        df.groupby(group_fields, as_index=False)["score"].sum().sort_values(group_fields)
    )
    return grouped


def load_raw_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Clean up legacy index columns if present.
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def get_scorer(
    mode: str,
    *,
    df_for_frequency: Optional[pd.DataFrame] = None,
    base_scheme: ScoringScheme = HANDCRAFTED_SCORES,
) -> Scorer:
    mode = mode.lower()
    if mode == "handcrafted":
        return Scorer(base_scheme)
    if mode == "frequency":
        if df_for_frequency is None:
            raise ValueError("Frequency-based scoring requires a dataframe")
        batter_scores = result_frequency(df_for_frequency, base_scheme.batter)
        pitcher_scores = result_frequency(df_for_frequency, base_scheme.pitcher)
        return Scorer(ScoringScheme(batter=batter_scores, pitcher=pitcher_scores))
    raise ValueError(f"Unknown scoring mode '{mode}'")


def save_edge_files(
    df: pd.DataFrame,
    *,
    output_dir: Path,
    slicer: Optional[str] = None,
    filters: Optional[Dict[str, Iterable]] = None,
    prefix: str = "edges",
) -> Dict[str, Path]:
    """Persist aggregated edges keyed by slicer value.

    Returns a mapping of slicer values (or 'all') to CSV file paths.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}

    if slicer is None:
        filename = output_dir / f"{prefix}_only.csv"
        df.to_csv(filename, index=False)
        saved["all"] = filename
        return saved

    for value, subset in df.groupby(slicer):
        filename = output_dir / f"{value}_{prefix}.csv"
        subset.to_csv(filename, index=False)
        saved[str(value)] = filename
    return saved


def generate_edge_tables(
    data_paths: Iterable[Path],
    *,
    scoring: str = "handcrafted",
    slicer: Optional[str] = None,
    filters: Optional[Dict[str, Iterable]] = None,
) -> pd.DataFrame:
    """Combine multiple season files into a single aggregated edge table."""

    dataframes = [load_raw_data(path) for path in data_paths]
    concat_df = pd.concat(dataframes, ignore_index=True)
    scorer = get_scorer(scoring, df_for_frequency=concat_df if scoring == "frequency" else None)
    edged = build_edge_dataframe(concat_df, scorer)
    aggregated = aggregate_edges(edged, slicer=slicer)
    return aggregated


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Add edge metadata to at-bat CSVs")
    parser.add_argument("inputs", nargs="+", type=Path, help="At-bat CSV files to process")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV file or directory when --slicer is provided",
    )
    parser.add_argument(
        "--scoring",
        choices=["handcrafted", "frequency"],
        default="handcrafted",
        help="Scoring heuristic to apply",
    )
    parser.add_argument(
        "--slicer",
        type=str,
        help="Optional column to retain while aggregating (e.g. pitch_type)",
    )

    return parser.parse_args()


def main():
    args = _parse_args()
    aggregated = generate_edge_tables(args.inputs, scoring=args.scoring, slicer=args.slicer)

    output_path = args.output
    if args.slicer:
        if not output_path.exists() or output_path.is_file():
            output_path.mkdir(parents=True, exist_ok=True)
        save_edge_files(aggregated, output_dir=output_path, slicer=args.slicer)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        aggregated.to_csv(output_path, index=False)


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    main()

