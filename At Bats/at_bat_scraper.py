"""Utilities for downloading MLB Statcast at-bat data."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd


def _load_baseball_scraper():
    import importlib

    return importlib.import_module("baseball_scraper")


# Default regular-season windows that roughly cover each MLB season.
DEFAULT_DATE_RANGES: Dict[int, Tuple[str, str]] = {
    2019: ("2019-03-20", "2019-10-30"),
    2018: ("2018-03-29", "2018-10-28"),
    2017: ("2017-04-02", "2017-11-01"),
    2016: ("2016-04-03", "2016-11-02"),
    2015: ("2015-04-05", "2015-11-01"),
    2014: ("2014-03-22", "2014-10-29"),
    2013: ("2013-03-31", "2013-10-30"),
    2012: ("2012-03-28", "2012-10-28"),
    2011: ("2011-03-31", "2011-10-28"),
    2010: ("2010-04-04", "2010-11-01"),
    2009: ("2009-04-05", "2009-11-04"),
}


def get_default_date_range(year: int) -> Tuple[str, str]:
    """Return the default (start_date, end_date) window for *year*."""

    try:
        return DEFAULT_DATE_RANGES[year]
    except KeyError:
        return f"{year}-03-01", f"{year}-11-30"


def _prepare_at_bat_frame(raw_df: pd.DataFrame, pyb) -> pd.DataFrame:
    if raw_df.empty:
        raise ValueError("The Statcast download returned an empty dataframe.")

    trimmed = raw_df.dropna(subset=["events"]).copy()

    categories = [
        "pitch_type",
        "player_name",
        "batter",
        "events",
        "description",
        "home_team",
        "away_team",
        "inning",
        "stand",
        "p_throws",
        "home_score",
        "away_score",
    ]

    trimmed = trimmed[categories].copy()
    trimmed["batter"] = trimmed["batter"].astype("int64", copy=False)

    player_ids = trimmed["batter"].unique().tolist()
    lookup_df = pyb.playerid_reverse_lookup(player_ids, key_type="mlbam")
    batter_names = lookup_df.loc[:, ("key_mlbam", "name_first", "name_last")].copy()
    batter_names["name_first"] = batter_names["name_first"].str.capitalize()
    batter_names["name_last"] = batter_names["name_last"].str.capitalize()
    batter_names["batter_name"] = (
        batter_names.loc[:, ("name_first", "name_last")].agg(" ".join, axis=1)
    )

    merged = trimmed.join(
        batter_names[["key_mlbam", "batter_name"]].set_index("key_mlbam"),
        on="batter",
        how="left",
    ).drop(columns=["batter"])

    merged = merged.rename(columns={"player_name": "pitcher_name"})
    return merged


def scrape_date_range(
    start_date: str,
    end_date: str,
    *,
    cache_dir: Optional[Path] = None,
    filename: Optional[str] = None,
) -> pd.DataFrame:
    pyb = _load_baseball_scraper()
    raw_df = pyb.statcast(start_date, end_date)
    merged = _prepare_at_bat_frame(raw_df, pyb)

    if filename is not None:
        if cache_dir is None:
            raise ValueError("cache_dir must be provided when filename is set")
        cache_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(cache_dir / filename, index=False)

    return merged


def scrape_year(
    year: int,
    *,
    output_dir: Path,
    overwrite: bool = False,
    date_range: Optional[Tuple[str, str]] = None,
) -> Path:
    start_date, end_date = date_range or get_default_date_range(year)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"at_bat_data_{year}.csv"

    if output_path.exists() and not overwrite:
        return output_path

    scrape_date_range(start_date, end_date, cache_dir=output_dir, filename=output_path.name)
    return output_path


def scrape_years(
    years: Sequence[int],
    *,
    output_dir: Path,
    overwrite: bool = False,
    custom_ranges: Optional[Dict[int, Tuple[str, str]]] = None,
) -> Dict[int, Path]:
    saved_paths: Dict[int, Path] = {}
    for year in years:
        drange = None
        if custom_ranges and year in custom_ranges:
            drange = custom_ranges[year]
        path = scrape_year(
            year,
            output_dir=output_dir,
            overwrite=overwrite,
            date_range=drange,
        )
        saved_paths[year] = path
    return saved_paths


def _parse_args(argv: Optional[Sequence[str]] = None) -> Dict[str, object]:
    import argparse

    parser = argparse.ArgumentParser(description="Download Statcast at-bat data")
    parser.add_argument("years", nargs="*", type=int, help="Years to download")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./general_data"),
        help="Destination directory for the CSV files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing season files instead of skipping them",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Override the start date (ISO format). Applies to single year inputs.",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="Override the end date (ISO format). Applies to single year inputs.",
    )

    args = parser.parse_args(argv)
    custom_range = None
    if args.start and args.end:
        if len(args.years) != 1:
            parser.error("Custom start/end requires exactly one year argument")
        custom_range = {args.years[0]: (args.start, args.end)}

    if not args.years:
        parser.error("Provide at least one year to download")

    return {
        "years": args.years,
        "output_dir": args.output_dir,
        "overwrite": args.overwrite,
        "custom_ranges": custom_range,
    }


def main(argv: Optional[Sequence[str]] = None) -> Dict[int, Path]:
    options = _parse_args(argv)
    return scrape_years(**options)


if __name__ == "__main__":
    main()
