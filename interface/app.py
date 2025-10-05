"""Interactive CLI to run MLB network analyses with custom parameters."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pipeline  # noqa: E402


def _parse_year_fragment(fragment: str) -> List[int]:
    fragment = fragment.strip()
    if not fragment:
        return []
    if "-" in fragment:
        start_str, end_str = fragment.split("-", 1)
        start, end = int(start_str), int(end_str)
        if start > end:
            raise ValueError
        return list(range(start, end + 1))
    return [int(fragment)]


def prompt_years(message: str, *, allow_empty: bool = False) -> List[int]:
    while True:
        raw = input(message).strip()
        if not raw and allow_empty:
            return []
        try:
            years: List[int] = []
            for fragment in raw.split(","):
                years.extend(_parse_year_fragment(fragment))
            if not years:
                raise ValueError
            years = sorted(set(years))
            return years
        except ValueError:
            print("  ✗ Please enter years like 2017 or ranges like 2015-2018 (comma separated).")


def prompt_yes_no(message: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input(f"{message} {suffix} ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("  ✗ Please answer y or n.")


def prompt_choice(message: str, options: Sequence[str], default: str) -> str:
    option_map = {str(i + 1): opt for i, opt in enumerate(options)}
    default_idx = options.index(default) + 1
    while True:
        print(message)
        for i, opt in enumerate(options, start=1):
            marker = "*" if opt == default else " "
            print(f"  {i}. {opt}{' (default)' if opt == default else ''}")
        raw = input(f"Select option [{default_idx}]: ").strip()
        if not raw:
            return default
        if raw in option_map:
            return option_map[raw]
        print("  ✗ Invalid selection. Enter the number tied to your choice.")


def collect_unique_values(column: str, data_paths: Iterable[Path]) -> List[str]:
    values: set = set()
    for path in data_paths:
        try:
            series = pd.read_csv(path, usecols=[column])[column]
        except ValueError:
            continue
        values.update(series.dropna().unique().tolist())
    cleaned = sorted(v for v in values if str(v).strip())
    return [str(v) for v in cleaned]


def build_filters(data_paths: Iterable[Path]) -> Dict[str, List[str]]:
    filters: Dict[str, List[str]] = {}

    if prompt_yes_no("Filter by inning?", default=False):
        innings = collect_unique_values("inning", data_paths)
        if innings:
            print("Available innings:", ", ".join(innings))
            selection = input("Enter innings (comma separated, e.g., 1,2,3): ").strip()
            chosen = [s.strip() for s in selection.split(",") if s.strip()]
            if chosen:
                filters["inning"] = chosen
    if prompt_yes_no("Filter by pitch type?", default=False):
        pitch_types = collect_unique_values("pitch_type", data_paths)
        if pitch_types:
            print("Available pitch types:", ", ".join(pitch_types))
            selection = input("Enter pitch types (comma separated): ").strip()
            chosen = [s.strip().upper() for s in selection.split(",") if s.strip()]
            if chosen:
                filters["pitch_type"] = chosen
    return filters


def display_top(title: str, entries: List[Sequence], limit: int = 10):
    print(f"\n{title}")
    if not entries:
        print("  (no data)")
        return
    for idx, (player, score) in enumerate(entries[:limit], start=1):
        print(f"  {idx:>2}. {player:<25} {score:>8.3f}")


def display_group_summary(group: str, summary: Dict[str, object]):
    print(f"\n=== {group.title()} Results ===")
    if "message" in summary:
        print("  " + summary["message"])
        return
    print(f"  SpringRank level span: {summary['level_span']:.3f}")
    display_top("  Top players (raw SpringRank)", summary["top_raw"])
    scaled_df = summary["top_scaled"]
    if not scaled_df.empty:
        print("\n  Top players (scaled)")
        for idx, row in scaled_df.iterrows():
            print(f"  {idx + 1:>2}. {row['player']:<25} {row['scaled_rank']:>8.3f}")


def save_outputs(result: pipeline.AnalysisResult, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    result.bipartite_edges.to_csv(destination / "bipartite_edges.csv", index=False)
    for group, df in result.unipartite_edges.items():
        df.to_csv(destination / f"{group}_unipartite_edges.csv", index=False)
    for group_name, summary in ("batter", result.batter), ("pitcher", result.pitcher):
        raw_df = pd.DataFrame(summary["raw_order"], columns=["player", "spring_rank"])
        raw_df.to_csv(destination / f"{group_name}_springrank.csv", index=False)
        summary["scaled_ranks"].to_csv(destination / f"{group_name}_scaled.csv", index=False)
    metadata_path = destination / "metadata.txt"
    filters_line = ", ".join(f"{k}={list(v)}" for k, v in result.filters.items()) or "none"
    metadata = [
        f"Years: {', '.join(str(y) for y in result.years)}",
        f"Scoring: {result.scoring}",
        f"Filters: {filters_line}",
    ]
    metadata_path.write_text("\n".join(metadata), encoding="utf-8")
    print(f"\nOutput saved to {destination}")


def main():
    print("MLB SpringRank Analysis Interface")
    print("-" * 40)

    years = prompt_years("Enter seasons to include (e.g., 2015-2017,2019): ")

    try:
        season_paths = pipeline.ensure_season_data(years)
    except FileNotFoundError as exc:
        print(str(exc))
        if prompt_yes_no("Attempt to download missing seasons now?", default=True):
            season_paths = pipeline.ensure_season_data(years, fetch_missing=True)
        else:
            print("Cannot continue without the requested data. Exiting.")
            return

    scoring = prompt_choice(
        "Select scoring strategy:",
        ["handcrafted", "frequency"],
        default="handcrafted",
    )

    filters = build_filters(season_paths.values())
    top_n_input = input("How many top players should be displayed? [10]: ").strip()
    try:
        top_n = int(top_n_input) if top_n_input else 10
    except ValueError:
        top_n = 10

    fetch_new = prompt_yes_no("Refresh season data before running?", default=False)
    result = pipeline.run_analysis(
        years,
        scoring=scoring,
        filters=filters if filters else None,
        fetch_missing=fetch_new,
        overwrite=fetch_new,
        top_n=top_n,
    )

    print("\nSummary")
    print(f"  Seasons: {', '.join(str(y) for y in result.years)}")
    print(f"  Scoring: {result.scoring}")
    if result.filters:
        print("  Filters:")
        for key, values in result.filters.items():
            print(f"    - {key}: {', '.join(str(v) for v in values)}")
    else:
        print("  Filters: none")

    display_group_summary("batter", result.batter)
    display_group_summary("pitcher", result.pitcher)

    if prompt_yes_no("Save detailed outputs to CSV?", default=False):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        default_dir = BASE_DIR / "outputs" / f"analysis_{timestamp}"
        target = input(f"Destination directory [{default_dir}]: ").strip()
        destination = Path(target) if target else default_dir
        save_outputs(result, destination)


if __name__ == "__main__":
    main()
