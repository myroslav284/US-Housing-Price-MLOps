"""
Hyperparameter Tuning Pipeline using Optuna + MLflow.

- Tunes LGBMRegressor hyperparameters on Train / Eval splits.
- Optimizes Eval RMSE while controlling overfitting via regularization.
- Logs all Optuna trials and the final Champion model to MLflow.
- Performs final unbiased evaluation on the untouched Holdout (2022-2023) split.
- Saves the final champion model to models/champion_model.pkl.
"""

import logging
from pathlib import Path
from typing import Optional
import joblib
import mlflow
import mlflow.lightgbm
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.config import BASE_DIR, PROCESSED_DATA_DIR, MODELS_DIR, TARGET_COL

# Silence verbose Optuna info logs; keep warnings/errors + our own logger
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_predictions(y_true: pd.Series, y_pred) -> dict[str, float]:
    """Compute regression metrics (MAE, RMSE, R2)."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_tuning(
    n_trials: int = 12,
    sample_frac: Optional[float] = 0.5,
    processed_dir: Path | str = PROCESSED_DATA_DIR,
    models_dir: Path | str = MODELS_DIR,
) -> None:
    """
    Tune LightGBM with Optuna, retrain champion on 100% Train,
    and run final evaluation on Eval and Holdout.
    """
    processed_dir = Path(processed_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup MLflow tracking
    db_path = BASE_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("LightGBM_Optuna_Tuning")

    # 2. Load datasets
    logger.info("Loading feature-engineered Parquet datasets...")
    train_df = pd.read_parquet(processed_dir / "train_features.parquet")
    eval_df = pd.read_parquet(processed_dir / "eval_features.parquet")
    holdout_df = pd.read_parquet(processed_dir / "holdout_features.parquet")

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_eval = eval_df.drop(columns=[TARGET_COL])
    y_eval = eval_df[TARGET_COL]
    X_holdout = holdout_df.drop(columns=[TARGET_COL])
    y_holdout = holdout_df[TARGET_COL]

    # Optional subsampling during Optuna search for faster CPU iteration
    if sample_frac is not None and 0 < sample_frac < 1.0:
        train_sample = train_df.sample(frac=sample_frac, random_state=42)
        X_search = train_sample.drop(columns=[TARGET_COL])
        y_search = train_sample[TARGET_COL]
        logger.info(f"Using {sample_frac*100:.0f}% sample ({len(X_search):,} rows) for fast Optuna search.")
    else:
        X_search, y_search = X_train, y_train

    # 3. Define Optuna objective function
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 90),
            "max_depth": trial.suggest_int("max_depth", 5, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        }

        model = LGBMRegressor(**params)
        model.fit(X_search, y_search)

        y_pred = model.predict(X_eval)
        metrics = evaluate_predictions(y_eval, y_pred)

        # Log each trial to MLflow
        with mlflow.start_run(run_name=f"trial_{trial.number:02d}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metrics({
                "eval_rmse": metrics["rmse"],
                "eval_mae": metrics["mae"],
                "eval_r2": metrics["r2"],
            })

        logger.info(
            f"Trial {trial.number:02d} | Eval RMSE: ${metrics['rmse']:,.2f} | "
            f"MAE: ${metrics['mae']:,.2f} | R2: {metrics['r2']:.4f}"
        )
        return metrics["rmse"]

    # 4. Run Optuna optimization
    logger.info(f"Starting Optuna search ({n_trials} trials)...")
    study = optuna.create_study(direction="minimize", study_name="lgbm_housing_tuning")
    study.optimize(objective, n_trials=n_trials)

    logger.info("=== Optuna Search Complete ===")
    logger.info(f"Best Trial Eval RMSE: ${study.best_value:,.2f}")
    logger.info(f"Best Hyperparameters: {study.best_params}")

    # 5. Train Final Champion Model on 100% of Train data
    logger.info("Retraining Champion LightGBM on 100% of Train data...")
    best_params = study.best_params.copy()
    best_params.update({
        "subsample_freq": 1,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    })

    champion_model = LGBMRegressor(**best_params)
    champion_model.fit(X_train, y_train)

    # 6. Evaluate across ALL splits (Train, Eval, and untouched Holdout)
    train_metrics = evaluate_predictions(y_train, champion_model.predict(X_train))
    eval_metrics = evaluate_predictions(y_eval, champion_model.predict(X_eval))
    holdout_metrics = evaluate_predictions(y_holdout, champion_model.predict(X_holdout))

    logger.info("--- FINAL CHAMPION EVALUATION ---")
    logger.info(
        f"TRAIN   (2012-2019): RMSE=${train_metrics['rmse']:,.2f} | "
        f"MAE=${train_metrics['mae']:,.2f} | R2={train_metrics['r2']:.4f}"
    )
    logger.info(
        f"EVAL    (2020-2021): RMSE=${eval_metrics['rmse']:,.2f} | "
        f"MAE=${eval_metrics['mae']:,.2f} | R2={eval_metrics['r2']:.4f}"
    )
    logger.info(
        f"HOLDOUT (2022-2023): RMSE=${holdout_metrics['rmse']:,.2f} | "
        f"MAE=${holdout_metrics['mae']:,.2f} | R2={holdout_metrics['r2']:.4f}"
    )

    # 7. Log Final Champion to MLflow and save to models/champion_model.pkl
    with mlflow.start_run(run_name="FINAL_CHAMPION_LIGHTGBM"):
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "eval_rmse": eval_metrics["rmse"],
            "eval_mae": eval_metrics["mae"],
            "eval_r2": eval_metrics["r2"],
            "holdout_rmse": holdout_metrics["rmse"],
            "holdout_mae": holdout_metrics["mae"],
            "holdout_r2": holdout_metrics["r2"],
        })
        mlflow.lightgbm.log_model(champion_model, artifact_path="model")

    champion_path = models_dir / "champion_model.pkl"
    joblib.dump(champion_model, champion_path)
    logger.info(f"Champion model saved successfully to: {champion_path}")


if __name__ == "__main__":
    run_tuning()