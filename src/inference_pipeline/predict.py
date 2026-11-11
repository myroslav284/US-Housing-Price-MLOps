"""
Inference & Feature Alignment Pipeline.

Transforms raw housing input dictionaries/DataFrames using saved preprocessing
rules and fitted encoders, aligns columns strictly with champion_model.pkl,
and outputs predicted prices.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd

from src.config import CITIES_DATA_PATH, MODELS_DIR, TARGET_COL
from src.feature_pipeline.preprocess import (
    merge_geographic_coordinates,
    handle_demographic_zeros,
)
from src.feature_pipeline.feature_engineering import (
    add_date_features,
    drop_unused_columns,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_artifacts(models_dir: Path | str = MODELS_DIR) -> tuple[Any, dict, Any]:
    """
    Load and cache model and fitted encoders in memory.
    Caching avoids slow disk I/O on every FastAPI request.
    """
    models_dir = Path(models_dir)
    model = joblib.load(models_dir / "champion_model.pkl")
    freq_map = joblib.load(models_dir / "freq_encoder.pkl")
    target_encoder = joblib.load(models_dir / "target_encoder.pkl")
    logger.info("Loaded champion model and encoders into memory.")
    return model, freq_map, target_encoder


def prepare_features_for_inference(
    input_df: pd.DataFrame,
    freq_map: dict,
    target_encoder: Any,
    expected_features: list[str],
    cities_path: Path | str = CITIES_DATA_PATH,
) -> pd.DataFrame:
    """Apply identical transformations as training without dropping user rows."""
    df = input_df.copy()

    # 1. Geographic enrichment (lat, lng)
    df = merge_geographic_coordinates(df, cities_path=cities_path)

    # 2. Convert demographic zeros to NaN
    df = handle_demographic_zeros(df)

    # 3. Extract calendar features (year, quarter, month)
    if "date" in df.columns:
        df = add_date_features(df)

    # 4. Apply saved Frequency Encoder (zipcode -> zipcode_freq)
    if "zipcode" in df.columns:
        df["zipcode_freq"] = df["zipcode"].map(freq_map).fillna(0)

    # 5. Apply saved Target Encoder (city_full -> city_full_encoded)
    if "city_full" in df.columns:
        df["city_full_encoded"] = target_encoder.transform(df["city_full"])

    # 6. Drop raw text / leakage columns
    df = drop_unused_columns(df)
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    # 7. Normalize spaces to underscores to match LightGBM's sanitized feature_name_
    df.columns = [str(c).replace(" ", "_") for c in df.columns]

    # 8. Align columns strictly with what LightGBM expects (missing -> NaN)
    df_aligned = df.reindex(columns=expected_features, fill_value=np.nan)
    return df_aligned


def predict_price(
    input_data: pd.DataFrame | dict,
    models_dir: Path | str = MODELS_DIR,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Run end-to-end prediction on raw DataFrame or single house dictionary.
    Returns (predicted_prices_array, aligned_features_dataframe).
    """
    if isinstance(input_data, dict):
        df_raw = pd.DataFrame([input_data])
    else:
        df_raw = input_data.copy()

    model, freq_map, target_encoder = load_artifacts(models_dir)
    expected_features = list(model.feature_name_)

    X_aligned = prepare_features_for_inference(
        df_raw,
        freq_map=freq_map,
        target_encoder=target_encoder,
        expected_features=expected_features,
    )

    predictions = model.predict(X_aligned)
    return predictions, X_aligned


if __name__ == "__main__":
    # Quick smoke test on 3 real rows from raw holdout.parquet
    from src.config import RAW_DATA_DIR

    sample_df = pd.read_parquet(RAW_DATA_DIR / "holdout.parquet").head(3)
    actuals = sample_df[TARGET_COL].values
    preds, _ = predict_price(sample_df)

    for i, (act, pred) in enumerate(zip(actuals, preds)):
        logger.info(f"Sample {i+1} | Actual: ${act:,.2f} | Predicted: ${pred:,.2f} | Diff: ${pred - act:+,.2f}")