"""
Feature Engineering Pipeline for US Housing Price Prediction.

This module transforms cleaned tabular data into ML-ready numerical features:
1. Calendar features (year, quarter, month) to capture housing market seasonality.
2. Frequency Encoding for high-cardinality zip codes (>6,000 unique values).
3. Target Encoding for metropolitan areas (city_full) to capture market price levels.
4. Drops target-leakage columns (median_sale_price) and raw categoricals.
5. Serializes fitted encoders to disk (models/) for production inference (FastAPI).
"""

import logging
from pathlib import Path
import pandas as pd
from category_encoders import TargetEncoder
import joblib

from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    TARGET_COL,
)

# Configure structured logging with timestamp and level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar components from 'date' to capture seasonal real estate cycles.
    
    Real estate experiences strong cyclicality: sales and prices peak during 
    late spring/summer (Q2) and decline during winter (Q4/Q1).
    - 'quarter' (1 to 4): high-level seasonal indicator (useful for tree splits).
    - 'month' (1 to 12): fine-grained monthly seasonality.
    - 'year': long-term macroeconomic trend (inflation, interest rates).
    """
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month
    return df


def frequency_encode(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    col: str = "zipcode",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Apply Frequency Encoding to a high-cardinality categorical column.
    
    Why Frequency Encoding for 'zipcode'?
    - There are 6,000+ zip codes. One-Hot Encoding would create 6,000 sparse columns,
      exploding memory and training time (Curse of Dimensionality).
    - Frequency Encoding replaces each zip code with its occurrence count in Train.
      This acts as a proxy for transaction density/market liquidity (how active 
      real estate is in that neighborhood).
      
    MLOps Leakage Protection:
    - Frequencies are calculated STRICTLY on the training set (train_df).
    - Eval and Holdout are mapped using the Train frequencies.
    - If a zip code appears in Eval/Holdout that was NEVER seen in Train, 
      it is safely imputed with 0 (.fillna(0)).
    """
    # 1. Calculate occurrence frequency per zipcode exclusively on Train
    freq_map = train_df[col].value_counts().to_dict()

    # 2. Map frequencies onto all three splits
    train_df[f"{col}_freq"] = train_df[col].map(freq_map)
    eval_df[f"{col}_freq"] = eval_df[col].map(freq_map).fillna(0)
    holdout_df[f"{col}_freq"] = holdout_df[col].map(freq_map).fillna(0)

    logger.info(f"Fitted frequency encoding for '{col}' ({len(freq_map):,} unique zip codes mapped).")
    return train_df, eval_df, holdout_df, freq_map


def target_encode(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    col: str = "city_full",
    target: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TargetEncoder]:
    """
    Apply Smoothed Target Encoding to a categorical column.
    
    Why Target Encoding for 'city_full'?
    - Replaces metropolitan names (e.g. 'San Francisco-Oakland-Berkeley') with the 
      empirical mean target price of that metro area.
    - This creates a continuous monotonic signal for both linear and tree models.
    - Uses empirical Bayes smoothing to prevent overfitting on smaller sub-markets.
    
    MLOps Leakage Protection:
    - .fit_transform() is called ONLY on Train.
    - .transform() is applied to Eval and Holdout (preventing future target leakage).
    """
    encoder = TargetEncoder(cols=[col])
    encoded_col_name = f"{col}_encoded"

    # Fit encoder on Train prices only
    train_df[encoded_col_name] = encoder.fit_transform(train_df[col], train_df[target])
    # Transform validation and test sets using the fitted weights
    eval_df[encoded_col_name] = encoder.transform(eval_df[col])
    holdout_df[encoded_col_name] = encoder.transform(holdout_df[col])

    logger.info(f"Fitted target encoding for '{col}' against target '{target}'.")
    return train_df, eval_df, holdout_df, encoder


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop raw text columns, redundant identifiers, and leakage features.
    
    Columns dropped:
    - 'date': decomposed into 'year', 'quarter', 'month'.
    - 'city_full': transformed into 'city_full_encoded' (and lat/lng coordinates).
    - 'city': short 3-letter metro code (redundant with city_full).
    - 'zipcode': transformed into 'zipcode_freq'.
    - 'median_sale_price': DROPPED TO PREVENT SEVERE TARGET LEAKAGE.
       In EDA, we saw that 'median_sale_price' contains artificial zeros and is 
       essentially another version of the target 'price'. Keeping it would allow 
       the model to cheat instead of learning from market indicators.
    """
    drop_cols = ["date", "city_full", "city", "zipcode", "median_sale_price"]
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


def run_feature_engineering(
    processed_dir: Path | str = PROCESSED_DATA_DIR,
    models_dir: Path | str = MODELS_DIR,
) -> None:
    """
    Execute end-to-end feature engineering pipeline:
    Load cleaned splits -> Add features -> Fit & apply encoders -> Save artifacts.
    """
    processed_dir = Path(processed_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load cleaned Parquet splits from Step 2
    logger.info(f"Loading cleaned data splits from: {processed_dir}")
    train_df = pd.read_parquet(processed_dir / "train_clean.parquet")
    eval_df = pd.read_parquet(processed_dir / "eval_clean.parquet")
    holdout_df = pd.read_parquet(processed_dir / "holdout_clean.parquet")

    logger.info(f"Initial shapes: Train={train_df.shape}, Eval={eval_df.shape}, Holdout={holdout_df.shape}")

    # 2. Extract calendar features
    logger.info("Extracting date features (quarter, month)...")
    train_df = add_date_features(train_df)
    eval_df = add_date_features(eval_df)
    holdout_df = add_date_features(holdout_df)

    # 3. Frequency encoding for zipcode
    train_df, eval_df, holdout_df, freq_map = frequency_encode(
        train_df, eval_df, holdout_df, col="zipcode"
    )
    # Save mapping for FastAPI inference
    freq_path = models_dir / "freq_encoder.pkl"
    joblib.dump(freq_map, freq_path)
    logger.info(f"Saved frequency encoder artifact to: {freq_path}")

    # 4. Target encoding for city_full
    train_df, eval_df, holdout_df, target_encoder = target_encode(
        train_df, eval_df, holdout_df, col="city_full", target=TARGET_COL
    )
    # Save encoder object for FastAPI inference
    te_path = models_dir / "target_encoder.pkl"
    joblib.dump(target_encoder, te_path)
    logger.info(f"Saved target encoder artifact to: {te_path}")

    # 5. Drop raw categoricals and leakage columns
    train_df = drop_unused_columns(train_df)
    eval_df = drop_unused_columns(eval_df)
    holdout_df = drop_unused_columns(holdout_df)

    # 6. Save final feature datasets to Parquet
    splits = {
        "train_features.parquet": train_df,
        "eval_features.parquet": eval_df,
        "holdout_features.parquet": holdout_df,
    }

    for filename, split_df in splits.items():
        save_path = processed_dir / filename
        split_df.to_parquet(save_path, index=False, engine="pyarrow")
        logger.info(f"Saved {filename:<25}: shape {split_df.shape}")

    logger.info("Feature engineering pipeline completed successfully!")


if __name__ == "__main__":
    run_feature_engineering()