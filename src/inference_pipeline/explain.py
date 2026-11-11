"""
SHAP Explainability Module (XAI).

Uses shap.TreeExplainer on our LightGBM champion to decompose each house price
prediction into exact dollar contributions per feature.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import shap

from src.config import MODELS_DIR
from src.inference_pipeline.predict import load_artifacts, predict_price

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_explainer(models_dir: Path | str = MODELS_DIR) -> shap.TreeExplainer:
    """Initialize and cache SHAP TreeExplainer in memory."""
    model, _, _ = load_artifacts(models_dir)
    explainer = shap.TreeExplainer(model)
    logger.info("Initialized and cached SHAP TreeExplainer.")
    return explainer


def explain_prediction(
    input_data: pd.DataFrame | dict,
    top_k: int = 5,
    models_dir: Path | str = MODELS_DIR,
) -> dict[str, Any]:
    """
    Predict price and explain the top positive/negative feature drivers via SHAP.
    
    Parameters
    ----------
    input_data : pd.DataFrame | dict
        Single house record (or DataFrame where the first row is explained).
    top_k : int
        Number of top positive and top negative drivers to extract.
    models_dir : Path | str
        Directory containing trained artifacts.
        
    Returns
    -------
    dict with predicted_price, base_value, top_positive_drivers,
    top_negative_drivers, and full feature contributions.
    """
    predictions, X_aligned = predict_price(input_data, models_dir=models_dir)
    pred_price = float(predictions[0])

    # Explain the first row
    row_features = X_aligned.iloc[[0]]
    explainer = get_explainer(models_dir)

    shap_values = explainer.shap_values(row_features)[0]
    base_value = float(np.asarray(explainer.expected_value).item())

    # Build structured list of feature contributions
    contributions = []
    for feat_name, feat_val, shap_val in zip(
        row_features.columns, row_features.iloc[0].values, shap_values
    ):
        contributions.append({
            "feature": str(feat_name),
            "value": None if pd.isna(feat_val) else float(feat_val),
            "shap_impact": float(shap_val),
        })

    # Top positive drivers (pushed price UP)
    top_positive = sorted(
        [c for c in contributions if c["shap_impact"] > 0],
        key=lambda x: x["shap_impact"],
        reverse=True,
    )[:top_k]

    # Top negative drivers (pulled price DOWN)
    top_negative = sorted(
        [c for c in contributions if c["shap_impact"] < 0],
        key=lambda x: x["shap_impact"],
    )[:top_k]

    return {
        "predicted_price": pred_price,
        "base_value": base_value,
        "top_positive_drivers": top_positive,
        "top_negative_drivers": top_negative,
        "all_contributions": sorted(
            contributions, key=lambda x: abs(x["shap_impact"]), reverse=True
        ),
    }


if __name__ == "__main__":
    # Smoke test on 1 real row from raw holdout.parquet
    from src.config import RAW_DATA_DIR, TARGET_COL

    sample_row = pd.read_parquet(RAW_DATA_DIR / "holdout.parquet").head(1)
    actual_price = float(sample_row[TARGET_COL].iloc[0])

    explanation = explain_prediction(sample_row, top_k=5)

    logger.info(f"Actual Price:    ${actual_price:,.2f}")
    logger.info(f"Predicted Price: ${explanation['predicted_price']:,.2f}")
    logger.info(f"Market Baseline: ${explanation['base_value']:,.2f}")

    logger.info("--- TOP 5 POSITIVE DRIVERS (Pushed Price UP) ---")
    for item in explanation["top_positive_drivers"]:
        logger.info(f"  + {item['feature']:<28} (val={item['value']}) -> +${item['shap_impact']:,.2f}")

    logger.info("--- TOP 5 NEGATIVE DRIVERS (Pulled Price DOWN) ---")
    for item in explanation["top_negative_drivers"]:
        logger.info(f"  - {item['feature']:<28} (val={item['value']}) -> -${abs(item['shap_impact']):,.2f}")