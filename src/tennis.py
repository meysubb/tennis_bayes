from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Union, Any
import urllib.request
import urllib.error
import polars as pl


def fetch_jeff_sackmann_atp_data(
    dest_dir: Union[str, Path] = "data/atp_raw",
    start_year: int = 2000,
    end_year: int = 2024,
    overwrite: bool = False,
) -> List[Path]:
    """Download ATP match CSV files from Jeff Sackmann's GitHub repo.

    By default this downloads atp_matches_<year>.csv for years in [start_year, end_year]
    into `dest_dir`. Returns a list of saved file Paths.

    Args:
        dest_dir: Local directory to save CSV files.
        start_year: First year to download (inclusive).
        end_year: Last year to download (inclusive).
        overwrite: If True, re-download files even if present locally.

    Returns:
        List of pathlib.Path objects pointing at downloaded CSVs.
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

    return saved_files

def read_tennis_data(files: List[Path]) -> Any:
    """Load and concatenate tennis CSV files into a polars DataFrame.

    Args:
        files: List of Path objects to CSV files.
    Returns:
        polars.DataFrame containing all rows from the files.
    """    
    dfs = []
    for p in files:
        try:
            df = pl.read_csv(str(p))
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return []
    return pl.concat(dfs, how="vertical")

