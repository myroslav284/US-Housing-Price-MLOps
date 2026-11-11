"""
Unit tests for the Feature Pipeline (load, preprocess, feature_engineering).
These tests run on tiny synthetic datasets in memory to ensure fast and isolated verification.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.feature_pipeline.load import load_and_split_data
from src.feature_pipeline.preprocess import (
    filter_outliers,
    drop_stale_duplicates,
    handle_demographic_zeros,
)
from src.feature_pipeline.feature_engineering import (
    add_date_features,
    frequency_encode,
    drop_unused_columns,
)


# =====================================================================
# 1. Tests for load.py (Temporal Splitting)
# =====================================================================

def test_load_and_split_data_temporal_boundaries(tmp_path):
    """Confirm temporal split correctly divides dates without overlap."""
    # Create small synthetic dataset spanning 2018 to 2023
    dummy_csv = tmp_path / "dummy_raw.csv"
    dummy_df = pd.DataFrame({
        "date": [
            "2018-05-31", "2019-12-31",  # Train (< 2020)
            "2020-01-31", "2021-06-30",  # Eval (2020 - 2021)
            "2022-01-31", "2023-12-31",  # Holdout (>= 2022)
        ],
        "price": [200_000, 220_000, 250_000, 270_000, 310_000, 350_000],
        "city_full": ["City_A"] * 6,
        "zipcode": [10001] * 6,
    })
    dummy_df.to_csv(dummy_csv, index=False)

    # Run split into temporary directory
    train, eval_df, holdout = load_and_split_data(
        raw_path=dummy_csv,
        output_dir=tmp_path,
        cutoff_eval="2020-01-01",
        cutoff_holdout="2022-01-01",
    )

    # Assertions
    assert len(train) == 2, "Train must contain exactly 2 rows"
    assert len(eval_df) == 2, "Eval must contain exactly 2 rows"
    assert len(holdout) == 2, "Holdout must contain exactly 2 rows"

    # Strict zero-leakage temporal checks
    assert train["date"].max() < pd.Timestamp("2020-01-01")
    assert eval_df["date"].min() >= pd.Timestamp("2020-01-01")
    assert eval_df["date"].max() < pd.Timestamp("2022-01-01")
    assert holdout["date"].min() >= pd.Timestamp("2022-01-01")

    # Verify Parquet files were created
    assert (tmp_path / "train.parquet").exists()
    assert (tmp_path / "eval.parquet").exists()
    assert (tmp_path / "holdout.parquet").exists()


# =====================================================================
# 2. Tests for preprocess.py (Outliers, Duplicates, Demographics)
# =====================================================================

def test_filter_outliers_removes_extreme_prices():
    """Confirm filter_outliers removes >19M typos and ratio > 10 anomalies."""
    df = pd.DataFrame({
        "median_list_price": [
            300_000,       # Normal
            50_000_000,    # Outlier: > $19M
            2_500_000,     # Outlier: ratio 25x > 10x
        ],
        "price": [300_000, 300_000, 100_000],
    })

    cleaned = filter_outliers(df, max_list_price=19_000_000, max_price_ratio=10.0)

    assert len(cleaned) == 1, "Only 1 normal row should survive"
    assert cleaned["median_list_price"].iloc[0] == 300_000


def test_drop_stale_duplicates_removes_identical_rows():
    """Confirm drop_stale_duplicates drops rows identical across metrics."""
    df = pd.DataFrame({
        "date": ["2020-01-31", "2020-02-28"],  # Different dates
        "year": [2020, 2020],                  # Same year
        "price": [300_000, 300_000],          # Identical metrics
        "median_list_price": [320_000, 320_000],
    })

    cleaned = drop_stale_duplicates(df)
    assert len(cleaned) == 0, "All identical snapshot copies must be dropped"


def test_handle_demographic_zeros_converts_to_nan():
    """Confirm artificial zeros in demographic census columns become NaN."""
    df = pd.DataFrame({
        "Total Population": [5000.0, 0.0],
        "Median Age": [35.0, 0.0],
        "price": [200_000, 300_000],
    })

    cleaned = handle_demographic_zeros(df, demo_cols=["Total Population", "Median Age"])

    assert not np.isnan(cleaned["Total Population"].iloc[0])
    assert np.isnan(cleaned["Total Population"].iloc[1])
    assert np.isnan(cleaned["Median Age"].iloc[1])


# =====================================================================
# 3. Tests for feature_engineering.py (Encodings & Feature Selection)
# =====================================================================

def test_add_date_features_extracts_calendar_parts():
    """Confirm year, quarter, month are properly parsed from date."""
    df = pd.DataFrame({"date": ["2023-05-15"]})
    res = add_date_features(df)

    assert res["year"].iloc[0] == 2023
    assert res["quarter"].iloc[0] == 2
    assert res["month"].iloc[0] == 5


def test_frequency_encode_unseen_values_get_zero():
    """Confirm unseen zip codes in evaluation set are safely imputed with 0."""
    train_df = pd.DataFrame({"zipcode": [10001, 10001, 10002]})
    eval_df = pd.DataFrame({"zipcode": [10001, 99999]})  # 99999 is brand new!
    holdout_df = pd.DataFrame({"zipcode": [10002]})

    train_out, eval_out, _, freq_map = frequency_encode(train_df, eval_df, holdout_df, col="zipcode")

    # 10001 appeared twice in train
    assert train_out["zipcode_freq"].iloc[0] == 2
    # 99999 was never seen in train -> must be 0
    assert eval_out["zipcode_freq"].iloc[1] == 0


def test_drop_unused_columns_prevents_leakage():
    """Confirm target leakage (median_sale_price) and raw strings are removed."""
    df = pd.DataFrame({
        "date": ["2020-01-01"],
        "city_full": ["Atlanta-Sandy Springs"],
        "city": ["ATL"],
        "zipcode": [30002],
        "median_sale_price": [250_000],  # Leakage!
        "price": [250_000],              # Target
        "month": [1],
    })

    res = drop_unused_columns(df)

    for leaked_col in ["date", "city_full", "city", "zipcode", "median_sale_price"]:
        assert leaked_col not in res.columns, f"{leaked_col} must be dropped"

    assert "price" in res.columns
    assert "month" in res.columns