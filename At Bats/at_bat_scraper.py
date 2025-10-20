"""Utilities for downloading MLB Statcast at-bat data."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd


def _load_baseball_scraper():
    import importlib
    print("[scrape] Loading baseball_scraper module and applying compatibility patch if needed...")
    mod = importlib.import_module("baseball_scraper")
    # Apply a small monkey-patch to tolerate missing duplicate-suffixed columns
    try:
        bs_statcast = importlib.import_module("baseball_scraper.statcast")
        if not getattr(bs_statcast, "_mlb_na_patched", False):
            _orig_post = bs_statcast.postprocessing
            import pandas as _pd

            def _safe_postprocessing(data, team=None):
                try:
                    return _orig_post(data, team)
                except KeyError:
                    # If expected numeric columns are missing (e.g., 'pitcher.1'),
                    # fall back to returning the unmodified frame rather than raising.
                    # Opportunistically convert numeric-looking columns.
                    for c in list(data.columns):
                        if data[c].dtype == object:
                            try:
                                data[c] = _pd.to_numeric(data[c], errors='ignore')
                            except Exception:
                                pass
                    return data

            bs_statcast.postprocessing = _safe_postprocessing  # type: ignore
            bs_statcast._mlb_na_patched = True  # type: ignore
            print("[scrape] Patched baseball_scraper.statcast.postprocessing for schema tolerance")
    except Exception:
        # If monkey-patching fails, we still return the module; other fallbacks handle errors.
        pass
    return mod


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


def _prepare_at_bat_frame(raw_df: pd.DataFrame, pyb, season_year: Optional[int] = None) -> pd.DataFrame:
    if raw_df.empty:
        raise ValueError("The Statcast download returned an empty dataframe.")
    print(f"[scrape] Preparing at-bat frame: raw rows={len(raw_df)}; columns={len(raw_df.columns)}")

    # Normalize potential duplicate columns (e.g., 'pitcher.1', 'fielder_2.1')
    # Keep the first occurrence when duplicate base names exist.
    def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
        cols = list(df.columns)
        seen = {}
        keep_idx = []
        for i, c in enumerate(cols):
            base = c.split(".")[0]
            if base not in seen:
                seen[base] = i
                keep_idx.append(i)
        if len(keep_idx) == len(cols):
            return df
        # Rebuild a frame with unique base-named columns only (first occurrence kept)
        new_cols = [cols[i].split(".")[0] for i in keep_idx]
        out = df.iloc[:, keep_idx].copy()
        out.columns = new_cols
        return out

    print("[scrape] Deduplicating dotted columns (e.g., pitcher.1) if present...")
    raw_df = _dedupe_columns(raw_df)
    print(f"[scrape] After dedupe: rows={len(raw_df)}; columns={len(raw_df.columns)}")
    print("[scrape] Dropping rows with missing events...")
    trimmed = raw_df.dropna(subset=["events"]).copy()
    print(f"[scrape] After dropna(events): rows={len(trimmed)}")

    # If present, filter to Major League Baseball games only (regular season/postseason)
    # Statcast frames commonly include 'game_type' (preferred) or 'type' with codes like R (regular), S (spring), P (postseason).
    allowed_types = {"R", "P"}
    gt_col = None
    for cand in ("game_type", "type"):
        if cand in trimmed.columns:
            gt_col = cand
            break
    if gt_col is not None:
        before = len(trimmed)
        trimmed = trimmed[trimmed[gt_col].isin(list(allowed_types))].copy()
        print(f"[scrape] Filtering MLB games by {gt_col} in {allowed_types}: {before} -> {len(trimmed)} rows")
    else:
        # Try to enrich using MLB Stats API via game_pk mapping
        added = False
        try:
            if ("game_pk" in trimmed.columns) and (season_year is not None):
                import json as _json
                from urllib.request import urlopen as _urlopen
                print(f"[scrape] Attempting game_type enrichment from schedule API for season {season_year}...")
                url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={season_year}"
                with _urlopen(url, timeout=20) as resp:
                    sched = _json.loads(resp.read().decode('utf-8'))
                games = []
                for d in sched.get('dates', []):
                    games.extend(d.get('games', []))
                mapping = { int(g.get('gamePk')): g.get('gameType') for g in games if g.get('gamePk') is not None }
                if mapping:
                    trimmed['game_type'] = trimmed['game_pk'].map(mapping)
                    gt_col = 'game_type'
                    before = len(trimmed)
                    trimmed = trimmed[trimmed['game_type'].isin(list(allowed_types))].copy()
                    print(f"[scrape] Enriched and filtered MLB games by game_type in {allowed_types}: {before} -> {len(trimmed)} rows")
                    added = True
        except Exception as ex:
            print(f"[scrape] WARNING: Could not enrich game_type via schedule API: {ex}")
        if not added:
            print("[scrape] WARNING: No game_type/type column found; cannot filter spring/exhibition. Keeping all rows.")

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
    # Optionally include date/game identifiers if present
    for extra in ("game_date", "game_pk"):
        if extra in trimmed.columns and extra not in categories:
            categories.append(extra)
    # Optionally include game_type in final CSV if present for debugging
    if gt_col is not None:
        categories.append(gt_col)

    # Some sources may label pitcher as 'pitcher' instead of 'player_name'
    if "player_name" not in trimmed.columns and "pitcher" in trimmed.columns:
        trimmed["player_name"] = trimmed["pitcher"]

    # Ensure required columns exist before selection
    print("[scrape] Verifying required columns exist before selection...")
    missing = [c for c in categories if c not in trimmed.columns]
    if missing:
        raise KeyError(f"Required columns missing from Statcast data: {missing}")

    print("[scrape] Selecting canonical columns and normalizing dtypes...")
    trimmed = trimmed[categories].copy()
    trimmed["batter"] = trimmed["batter"].astype("int64", copy=False)

    player_ids = trimmed["batter"].unique().tolist()
    # Prefer pybaseball first (more up to date), then fallback to baseball_scraper.
    lookup_df: Optional[pd.DataFrame] = None
    source = None
    try:
        print(f"[scrape] Trying pybaseball reverse lookup for {len(player_ids)} batter IDs...")
        import importlib as _il
        _pb = _il.import_module("pybaseball")
        lookup_df = _pb.playerid_reverse_lookup(player_ids, key_type="mlbam")
        source = "pybaseball"
    except Exception:
        lookup_df = None
    if lookup_df is None or lookup_df.empty:
        try:
            print("[scrape] Falling back to baseball_scraper reverse lookup...")
            lookup_df = pyb.playerid_reverse_lookup(player_ids, key_type="mlbam")
            source = "baseball_scraper"
        except Exception:
            lookup_df = None
    batter_names: pd.DataFrame
    resolved = 0
    if lookup_df is not None and not lookup_df.empty and all(
        c in lookup_df.columns for c in ["key_mlbam", "name_first", "name_last"]
    ):
        print(f"[scrape] Lookup via {source} returned {len(lookup_df)} rows; normalizing keys and composing names...")
        batter_names = lookup_df.loc[:, ("key_mlbam", "name_first", "name_last")].copy()
        # Normalize key type to int to match Statcast batter ids
        try:
            batter_names["key_mlbam"] = pd.to_numeric(batter_names["key_mlbam"], errors="coerce").astype("Int64")
        except Exception:
            pass
        batter_names["name_first"] = batter_names["name_first"].astype(str).str.capitalize()
        batter_names["name_last"] = batter_names["name_last"].astype(str).str.capitalize()
        batter_names["batter_name"] = (
            batter_names.loc[:, ("name_first", "name_last")].agg(" ".join, axis=1)
        )
        # Count how many in our player_ids have a mapping
        try:
            key_set = set(pd.Series(player_ids, dtype="Int64").dropna().tolist())
            resolved = int(batter_names["key_mlbam"].isin(list(key_set)).sum())
        except Exception:
            resolved = 0
    else:
        # Fallback: use mlbam id strings as the batter_name to proceed without failing
        batter_names = pd.DataFrame({
            "key_mlbam": player_ids,
            "batter_name": [f"MLBAM_{pid}" for pid in player_ids],
        })
        source = "fallback"
        resolved = 0

    print("[scrape] Joining batter names onto trimmed frame and renaming pitcher column...")
    merged = trimmed.join(
        batter_names[["key_mlbam", "batter_name"]].set_index("key_mlbam"),
        on="batter",
        how="left",
    )
    # Second-chance resolution: where batter_name looks like MLBAM_<id>, try to resolve to real names
    try:
        unresolved_mask = merged["batter_name"].astype(str).str.match(r"^MLBAM_\d+$", na=False)
        unresolved_ids = (
            merged.loc[unresolved_mask, "batter"].dropna().astype("int64", errors="ignore").unique().tolist()
            if "batter" in merged.columns else []
        )
        if unresolved_ids:
            print(f"[scrape] Second-chance name lookup for {len(unresolved_ids)} unresolved batter IDs...")
            _res_df = None
            try:
                import importlib as _il
                _pb2 = _il.import_module("pybaseball")
                _res_df = _pb2.playerid_reverse_lookup(unresolved_ids, key_type="mlbam")
                src2 = "pybaseball"
            except Exception:
                _res_df = None
            if (_res_df is None) or _res_df.empty:
                try:
                    _res_df = pyb.playerid_reverse_lookup(unresolved_ids, key_type="mlbam")
                    src2 = "baseball_scraper"
                except Exception:
                    _res_df = None
            if (_res_df is not None) and (not _res_df.empty) and all(c in _res_df.columns for c in ["key_mlbam","name_first","name_last"]):
                try:
                    _res = _res_df.loc[:, ["key_mlbam","name_first","name_last"]].copy()
                    _res["key_mlbam"] = pd.to_numeric(_res["key_mlbam"], errors="coerce").astype("Int64")
                    _res["name_first"] = _res["name_first"].astype(str).str.capitalize()
                    _res["name_last"] = _res["name_last"].astype(str).str.capitalize()
                    _res["batter_name_resolved"] = _res.loc[:, ["name_first","name_last"]].agg(" ".join, axis=1)
                    # Map back onto merged using batter id
                    map_sr = _res.set_index("key_mlbam")["batter_name_resolved"]
                    before_cnt = int(unresolved_mask.sum())
                    merged.loc[unresolved_mask, "batter_name"] = merged.loc[unresolved_mask, "batter"].map(map_sr).fillna(merged.loc[unresolved_mask, "batter_name"])
                    after_cnt = int(merged["batter_name"].astype(str).str.match(r"^MLBAM_\d+$", na=False).sum())
                    fixed = before_cnt - after_cnt
                    print(f"[scrape] Second-chance name lookup via {src2}: updated {fixed}/{before_cnt}")
                except Exception:
                    pass
    except Exception:
        pass
    # For any batter IDs that didn't resolve, fill with MLBAM_<id>
    try:
        missing_mask = merged["batter_name"].isna()
        missing_cnt = int(missing_mask.sum())
        if missing_cnt:
            merged.loc[missing_mask, "batter_name"] = merged.loc[missing_mask, "batter"].apply(lambda x: f"MLBAM_{int(x)}")
            print(f"[scrape] Filled {missing_cnt} missing batter names with MLBAM_<id>")
    except Exception:
        pass
    merged = merged.drop(columns=["batter"])

    merged = merged.rename(columns={"player_name": "pitcher_name"})
    try:
        total = len(player_ids)
        pct = (resolved / total * 100.0) if total else 0.0
        print(f"[scrape] Batter name resolution via {source}: {resolved}/{total} ({pct:.1f}%)")
    except Exception:
        pass
    print(f"[scrape] Prepared at-bat frame complete: rows={len(merged)}; columns={len(merged.columns)}")
    return merged


def scrape_date_range(
    start_date: str,
    end_date: str,
    *,
    cache_dir: Optional[Path] = None,
    filename: Optional[str] = None,
) -> pd.DataFrame:
    from datetime import datetime, timedelta

    print(f"[scrape] scrape_date_range start: {start_date}..{end_date}")
    pyb = _load_baseball_scraper()

    # Fetch in weekly chunks for stability and progress visibility
    def _to_date(s: str):
        return datetime.strptime(s, "%Y-%m-%d").date()

    def _date_range(d0, d1):
        cur = d0
        while cur <= d1:
            yield cur
            cur = cur + timedelta(days=1)

    def _has_games(day) -> Optional[bool]:
        # Query MLB Stats API to see if any MLB games are scheduled for this date.
        # Returns True/False if known, or None on network/parsing errors (unknown).
        try:
            import json as _json
            from urllib.request import urlopen as _urlopen
            from urllib.error import URLError as _URLError
            url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={day.strftime('%Y-%m-%d')}"
            with _urlopen(url, timeout=10) as resp:
                data = _json.loads(resp.read().decode('utf-8'))
            dates = data.get('dates', [])
            if not dates:
                return False
            total = data.get('totalGames') if 'totalGames' in data else dates[0].get('totalGames', 0)
            return bool(total and total > 0)
        except Exception:
            return None

    start = _to_date(start_date)
    end = _to_date(end_date)
    all_raw_parts = []
    cur = start
    def _fetch_week(s: str, e: str) -> pd.DataFrame:
        """Fetch a week; on schema-related errors, fallback to per-day and pybaseball if available."""
        try:
            return pyb.statcast(s, e)
        except Exception as ex:
            msg = str(ex)
            # Try weekly-level fallback to pybaseball first
            try:
                import importlib as _il
                pb = _il.import_module("pybaseball")
                wk = pb.statcast(s, e)
                if wk is not None and not wk.empty:
                    print(f"[scrape] pybaseball weekly fallback succeeded for {s}..{e}")
                    return wk
            except Exception:
                pass
            if isinstance(ex, KeyError) or "not in index" in msg.lower():
                print(f"[scrape] Weekly fetch failed due to schema mismatch ({ex}); retrying daily {s}..{e}")
                from datetime import datetime, timedelta as _td
                sd = datetime.strptime(s, "%Y-%m-%d").date()
                ed = datetime.strptime(e, "%Y-%m-%d").date()
                day_parts = []
                day = sd
                while day <= ed:
                    day_s = day.strftime("%Y-%m-%d")
                    # Skip this day quickly if no MLB games scheduled
                    try:
                        from urllib.request import urlopen as _urlopen
                        import json as _json
                        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={day_s}"
                        with _urlopen(url, timeout=10) as resp:
                            data = _json.loads(resp.read().decode('utf-8'))
                        if not data.get('dates') or (data.get('totalGames') == 0):
                            print(f"[scrape] Skipping {day_s} (no MLB games)")
                            day = day + _td(days=1)
                            continue
                    except Exception:
                        pass
                    try:
                        dp = pyb.statcast(day_s, day_s)
                    except Exception as ex_day:
                        dp = None
                        # Optional fallback to pybaseball for that day
                        try:
                            import importlib
                            pb = importlib.import_module("pybaseball")
                            dp = pb.statcast(day_s, day_s)
                            print(f"[scrape] pybaseball fallback succeeded for {day_s}")
                        except Exception:
                            print(f"[scrape] Skipping {day_s} due to error: {ex_day}")
                    if dp is not None and not dp.empty:
                        day_parts.append(dp)
                    day = day + _td(days=1)
                if day_parts:
                    return pd.concat(day_parts, ignore_index=True, sort=False)
                # If no day worked, return empty to signal skip without failing
                import pandas as _pd
                return _pd.DataFrame()
            # Non-schema error: return empty and let caller skip
            import pandas as _pd
            return _pd.DataFrame()

    while cur <= end:
        chunk_end = min(cur + timedelta(days=6), end)
        s = cur.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")
        # Pre-check if any games in this week; if definitively none, skip the week
        week_days = list(_date_range(cur, chunk_end))
        week_flags = [ _has_games(d) for d in week_days ]
        if week_flags and all(f is False for f in week_flags):
            print(f"[scrape] Skipping {s}..{e} (no MLB games scheduled)")
            cur = chunk_end + timedelta(days=1)
            continue
        print(f"[scrape] Fetching Statcast {s}..{e}")
        part = _fetch_week(s, e)
        if part is not None and not part.empty:
            all_raw_parts.append(part)
        cur = chunk_end + timedelta(days=1)

    if not all_raw_parts:
        print(f"[scrape] No Statcast data for {start_date}..{end_date}; skipping")
        return pd.DataFrame()

    total_parts = len(all_raw_parts)
    total_rows = sum(len(p) for p in all_raw_parts)
    print(f"[scrape] Combining {total_parts} parts ({total_rows} rows) into raw frame...")
    raw_df = pd.concat(all_raw_parts, ignore_index=True, sort=False)
    try:
        try:
            season_year = int(start_date[:4])
        except Exception:
            season_year = None
        merged = _prepare_at_bat_frame(raw_df, pyb, season_year=season_year)
    except Exception as ex:
        # Persist the raw combined dataframe for debugging/recovery if possible
        try:
            if filename is not None and cache_dir is not None:
                tmp_name = filename.replace('.csv', '.raw.csv')
                cache_dir.mkdir(parents=True, exist_ok=True)
                raw_df.to_csv(cache_dir / tmp_name, index=False)
                print(f"[scrape] Saved raw data to {cache_dir / tmp_name} after merge failure: {ex}")
        except Exception:
            pass
        # Re-raise after saving raw data; upstream may handle or abort
        raise

    if filename is not None and not merged.empty:
        if cache_dir is None:
            raise ValueError("cache_dir must be provided when filename is set")
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path = cache_dir / filename
        print(f"[scrape] Writing merged CSV -> {out_path}")
        merged.to_csv(out_path, index=False)

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
        print(f"[scrape] scrape_year {year}: exists -> skip ({output_path})")
        return output_path

    print(f"[scrape] scrape_year {year}: downloading to {output_path}")
    scrape_date_range(start_date, end_date, cache_dir=output_dir, filename=output_path.name)
    print(f"[scrape] scrape_year {year}: wrote {output_path}")
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
        print(f"[scrape] === Year {year} ===")
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
    print("[scrape] All requested years processed.")
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
