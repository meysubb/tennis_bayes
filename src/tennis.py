from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Union, Any
import urllib.request
import urllib.error


def fetch_jeff_sackmann_atp_data(
    dest_dir: Union[str, Path] = "data/atp_raw",
    start_year: int = 2000,
    end_year: int = 2024,
    overwrite: bool = False,
    as_polars: bool = False,
) -> Union[List[Path], Any]:
    """Download ATP match CSV files from Jeff Sackmann's GitHub repo.

    By default this downloads atp_matches_<year>.csv for years in [start_year, end_year]
    into `dest_dir`. If `as_polars=True` and polars is installed,
    the function returns a single concatenated DataFrame; otherwise it returns a list of saved file Paths.

    Args:
        dest_dir: Local directory to save CSV files.
        start_year: First year to download (inclusive).
        end_year: Last year to download (inclusive).
        overwrite: If True, re-download files even if present locally.
        as_polars: If True, attempt to load and return a single polars.DataFrame.

    Returns:
        List of pathlib.Path objects pointing at downloaded CSVs, or a polars.DataFrame
        when `as_polars=True` and polars is available.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    base_raw = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
    saved_files: List[Path] = []

    for year in range(start_year, end_year + 1):
        filename = f"atp_matches_{year}.csv"
        url = f"{base_raw}/{filename}"
        out_path = dest / filename
        if out_path.exists() and not overwrite:
            saved_files.append(out_path)
            continue

        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                # write to file in binary mode
                with open(out_path, "wb") as fh:
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        fh.write(chunk)
            saved_files.append(out_path)
        except urllib.error.HTTPError as e:
            # 404 -> skip early/nonexistent years; re-raise other HTTP errors
            if e.code == 404:
                continue
            raise
        except Exception:
            # propagate other exceptions (e.g., network errors)
            raise

    if as_polars:
        try:
            import polars as pl
        except Exception:
            raise RuntimeError("polars is required to return a DataFrame. Install polars or set as_polars=False")
        dfs = []
        for p in saved_files:
            try:
                df = pl.read_csv(str(p))
                dfs.append(df)
            except Exception:
                continue
        if not dfs:
            return []
        return pl.concat(dfs, how="vertical")

    return saved_files


# Small convenience wrapper
def download_atp_matches(*, years: Optional[List[int]] = None, dest_dir: Union[str, Path] = "data/atp_raw", as_polars: bool = False, **kwargs):
    """Convenience wrapper that accepts an explicit list of years.

    Example: download_atp_matches(years=[2018,2019,2020], dest_dir="./data", as_polars=True)
    """
    if years is not None:
        if not years:
            return []
        start_year = min(years)
        end_year = max(years)
        files = fetch_jeff_sackmann_atp_data(dest_dir=dest_dir, start_year=start_year, end_year=end_year, as_polars=as_polars, **kwargs)
        # If the caller provided a sparse list (not a contiguous range), filter results
        if isinstance(files, list) and set(years) != set(range(start_year, end_year + 1)):
            wanted = {f"atp_matches_{y}.csv" for y in years}
            return [p for p in files if p.name in wanted]
        return files
    else:
        return fetch_jeff_sackmann_atp_data(dest_dir=dest_dir, as_polars=as_polars, **kwargs)