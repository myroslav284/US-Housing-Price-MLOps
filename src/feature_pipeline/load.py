"""
Load & time-split the raw dataset.

- Reads raw dataset and sorts by date.
- Splits into train / eval / holdout by temporal cutoffs.
- Saves splits as Parquet files to data/raw/ (or custom output_dir).
"""

import logging
from pathlib import Path
import pandas as pd

from src.config import (
    HOUSE_DATA_PATH,
    RAW_DATA_DIR,
    CUTOFF_DATE_EVAL,
    CUTOFF_DATE_HOLDOUT,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_and_split_data(
    raw_path: Path | str = HOUSE_DATA_PATH,
    output_dir: Path | str = RAW_DATA_DIR,
    cutoff_eval: str = CUTOFF_DATE_EVAL,
    cutoff_holdout: str = CUTOFF_DATE_HOLDOUT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw dataset, sort by date, split into train/eval/holdout, and save to output_dir."""
    logger.info(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)

    # Ensure datetime + chronological sort
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Temporal split masks
    train_mask = df["date"] < cutoff_eval
    eval_mask = (df["date"] >= cutoff_eval) & (df["date"] < cutoff_holdout)
    holdout_mask = df["date"] >= cutoff_holdout

    train_df = df[train_mask].copy()
    eval_df = df[eval_mask].copy()
    holdout_df = df[holdout_mask].copy()

    # Save to Parquet
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_path = outdir / "train.parquet"
    eval_path = outdir / "eval.parquet"
    holdout_path = outdir / "holdout.parquet"

    train_df.to_parquet(train_path, index=False, engine="pyarrow")
    eval_df.to_parquet(eval_path, index=False, engine="pyarrow")
    holdout_df.to_parquet(holdout_path, index=False, engine="pyarrow")

    logger.info(f"Data split completed (saved to {outdir}):")
    logger.info(f"   Train:   {train_df.shape} | {train_df['date'].min().date()} -> {train_df['date'].max().date()}")
    logger.info(f"   Eval:    {eval_df.shape} | {eval_df['date'].min().date()} -> {eval_df['date'].max().date()}")
    logger.info(f"   Holdout: {holdout_df.shape} | {holdout_df['date'].min().date()} -> {holdout_df['date'].max().date()}")

    return train_df, eval_df, holdout_df


if __name__ == "__main__":
    load_and_split_data()