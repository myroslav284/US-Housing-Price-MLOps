"""
Preprocessing Pipeline for US Housing Price Prediction.
- Loads raw train/eval/holdout splits from data/raw/ (.parquet).
- Enriches data with latitude and longitude from uscities.csv.
- Removes stale duplicates (excluding date/year).
- Filters extreme listing outliers (> $19M and list/price ratio > 10).
- Converts artificial demographic zeros to NaN.
- Saves cleaned datasets to data/processed/.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CITIES_DATA_PATH,
    MAX_LIST_PRICE,
    MAX_PRICE_RATIO,
    TARGET_COL,
    DEMOGRAPHIC_COLS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def merge_geographic_coordinates(
        df: pd.DataFrame,
        cities_path: Path | str = CITIES_DATA_PATH,
) -> pd.DataFrame:
    """Enrich dataset with lat/lng by matching primary city against uscities.csv."""
    if {'lat', 'lng'}.issubset(df.columns):
        logger.info("Coordinates already present, skipping geo-merge.")
        return df

    cities_path = Path(cities_path)
    if not cities_path:
        logger.warning(f"Cities file not found at {cities_path}, skipping geo-merge.")
        return df

    df_cities = pd.read_csv(cities_path)
    # Deduplicate uscities by keeping the most populous city per name
    cities_coords = (
        df_cities.sort_values("population", ascending=False)
        .drop_duplicates("city")[["city", "lat", "lng"]]
        .rename(columns={"city": "primary_city"})
    )

    # Extract primary city from 'city_full' and handle known exception
    df["primary_city"] = df["city_full"].apply(
        lambda x: "Washington" if x == "DC_Metro" else str(x).split("-")[0].strip()
    )

    df = df.merge(cities_coords, on="primary_city", how="left")
    df = df.drop(columns=["primary_city"])
    missing_count = df["lat"].isnull().sum()
    if missing_count > 0:
        logger.warning(f"{missing_count} rows could not be matched with coordinates.")
    else:
        logger.info("100% of rows successfully matched with lat/lng coordinates.")
    return df


def drop_stale_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate snapshots where metrics didn't change across dates."""
    cols_without_time = df.columns.difference(["date", "year"])
    before = len(df)
    df = df.drop_duplicates(subset=cols_without_time, keep=False).copy()
    logger.info(f"Dropped stale duplicates: {before - len(df):,} rows removed.")
    return df


def filter_outliers(
    df: pd.DataFrame,
    max_list_price: float = MAX_LIST_PRICE,
    max_price_ratio: float = MAX_PRICE_RATIO,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Remove artificial list price typos (> 19M) and extreme non-market ratios."""
    before = len(df)
    mask = df["median_list_price"] <= max_list_price
    if target_col in df.columns:
        ratio = df["median_list_price"] / df[target_col]
        mask = mask & (ratio <= max_price_ratio)
    df_clean = df[mask].copy()
    logger.info(f"Filtered outliers: {before - len(df_clean):,} rows removed.")
    return df_clean


def handle_demographic_zeros(
    df: pd.DataFrame,
    demo_cols: list[str] = DEMOGRAPHIC_COLS,
) -> pd.DataFrame:
    """Convert census demographic zeros (missing census tracts) to NaN."""
    existing_demo_cols = [c for c in demo_cols if c in df.columns]
    zeros_count = (df[existing_demo_cols] == 0).sum().sum()
    if zeros_count > 0:
        df[existing_demo_cols] = df[existing_demo_cols].replace(0, np.nan)
        logger.info(f"Replaced {zeros_count:,} artificial zeros in demographic columns with NaN.")
    return df


def preprocess_split(
    split_name: str,
    raw_dir: Path | str = RAW_DATA_DIR,
    processed_dir: Path | str = PROCESSED_DATA_DIR,
    cities_path: Path | str = CITIES_DATA_PATH,
) -> pd.DataFrame:
    """Preprocess a single data split and save as clean Parquet."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    input_file = raw_dir / f"{split_name}.parquet"
    logger.info(f"--- Processing {split_name.upper()} split from {input_file} ---")
    df = pd.read_parquet(input_file)
    # Apply cleaning steps
    df = merge_geographic_coordinates(df, cities_path=cities_path)
    df = drop_stale_duplicates(df)
    df = filter_outliers(df)
    df = handle_demographic_zeros(df)
    output_file = processed_dir / f"{split_name}_clean.parquet"
    df.to_parquet(output_file, index=False, engine="pyarrow")
    logger.info(f"Saved cleaned split: {output_file} (shape: {df.shape})\n")
    return df


def run_preprocess(splits: tuple[str, ...] = ("train", "eval", "holdout")) -> None:
    """Run preprocessing for all data splits."""
    for split in splits:
        preprocess_split(split)
    logger.info("All splits preprocessed successfully!")
if __name__ == "__main__":
    run_preprocess()