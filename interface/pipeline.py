"""Composable analysis pipeline used by the interactive CLI."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
AT_BATS_DIR = BASE_DIR / "At Bats"
RAW_DATA_DIR = AT_BATS_DIR / "general_data"

if str(AT_BATS_DIR) not in sys.path:
    sys.path.insert(0, str(AT_BATS_DIR))

import add_edgeinfo  # noqa: E402
import at_bat_scraper  # noqa: E402
import BipartiteTo2Unipartite  # noqa: E402
import RankingLevels  # noqa: E402


@dataclass
class AnalysisResult:
    filters: Dict[str, Iterable]
    scoring: str
    years: Sequence[int]
    bipartite_edges: pd.DataFrame
    unipartite_edges: Dict[str, pd.DataFrame]
    batter: Dict[str, object]
    pitcher: Dict[str, object]


def ensure_season_data(
    years: Sequence[int],
    *,
    fetch_missing: bool = False,
    overwrite: bool = False,
) -> Dict[int, Path]:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    available: Dict[int, Path] = {}
    missing: List[int] = []

    for year in years:
        target = RAW_DATA_DIR / f"at_bat_data_{year}.csv"
        if overwrite and target.exists():
            target.unlink()
        if target.exists():
            available[year] = target
            continue
        if fetch_missing:
            at_bat_scraper.scrape_year(year, output_dir=RAW_DATA_DIR, overwrite=True)
            available[year] = target
        else:
            missing.append(year)

    if missing:
        raise FileNotFoundError(
            "Missing raw data for seasons: " + ", ".join(str(y) for y in missing)
        )

    return available


def build_bipartite_edges(
    data_paths: Iterable[Path],
    *,
    scoring: str = "handcrafted",
    filters: Optional[Dict[str, Iterable]] = None,
) -> pd.DataFrame:
    return add_edgeinfo.generate_edge_tables(
        data_paths,
        scoring=scoring,
        filters=filters,
    )


def convert_to_unipartite(edge_df: pd.DataFrame):
    batter_df, pitcher_df = BipartiteTo2Unipartite.convert_bipartite(edge_df)
    return {
        "batter": batter_df,
        "pitcher": pitcher_df,
    }


def _prepare_group_summary(
    df: pd.DataFrame,
    group_name: str,
    *,
    top_n: int = 10,
) -> Dict[str, object]:
    empty_summary = {
        "level_span": 0.0,
        "top_raw": [],
        "top_scaled": pd.DataFrame(columns=["player", "scaled_rank"]),
        "raw_order": [],
        "scaled_ranks": pd.DataFrame(columns=["player", "scaled_rank"]),
    }

    if df.empty:
        return {
            **empty_summary,
            "message": f"No edges generated for {group_name} players with the current filters.",
        }

    try:
        level_span, sr_sorted, scaled_df = RankingLevels.compute_levels(
            df,
            return_scaled=True,
        )
    except ModuleNotFoundError as exc:
        return {**empty_summary, "message": str(exc)}

    top_raw = sr_sorted[:top_n]
    top_scaled = scaled_df.head(top_n)
    return {
        "level_span": level_span,
        "top_raw": top_raw,
        "top_scaled": top_scaled,
        "raw_order": sr_sorted,
        "scaled_ranks": scaled_df,
    }


def run_analysis(
    years: Sequence[int],
    *,
    scoring: str = "handcrafted",
    filters: Optional[Dict[str, Iterable]] = None,
    fetch_missing: bool = False,
    overwrite: bool = False,
    top_n: int = 10,
) -> AnalysisResult:
    data_paths = ensure_season_data(years, fetch_missing=fetch_missing, overwrite=overwrite)
    bipartite = build_bipartite_edges(data_paths.values(), scoring=scoring, filters=filters)
    unipartite = convert_to_unipartite(bipartite)

    batter_summary = _prepare_group_summary(unipartite["batter"], "batter", top_n=top_n)
    pitcher_summary = _prepare_group_summary(unipartite["pitcher"], "pitcher", top_n=top_n)

    return AnalysisResult(
        filters=filters or {},
        scoring=scoring,
        years=tuple(sorted(years)),
        bipartite_edges=bipartite,
        unipartite_edges=unipartite,
        batter=batter_summary,
        pitcher=pitcher_summary,
    )
