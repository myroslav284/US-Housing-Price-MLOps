"""
Model Factory for Housing Price Regression.
Provides configured Scikit-Learn pipelines and Gradient Boosting regressors:
- Linear Models: Ridge, Lasso (with SimpleImputer + StandardScaler)
- Tree Ensembles: Random Forest (with SimpleImputer)
- Gradient Boostings: XGBoost (hist), LightGBM, CatBoost (native NaN handling)
"""
from typing import Any
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def get_model(model_name: str, **kwargs: Any) -> Any:
    """
    Factory function returning a configured regressor model or pipeline.
    
    Parameters
    ----------
    model_name : str
        One of 'ridge', 'lasso', 'random_forest', 'xgboost', 'lightgbm', 'catboost'.
    **kwargs : dict
        Optional hyperparameter overrides.
    """
    name = model_name.lower().strip()

    if name == "ridge":
        params = {"alpha": 1.0, "random_state": 42}
        params.update(kwargs)
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(**params))
        ])
    
    elif name == "lasso":
        params = {"alpha": 100.0, "max_iter": 2000, "random_state": 42}
        params.update(kwargs)
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Lasso(**params)),
        ])

    elif name in ("rf", "random_forest"):
        params = {
            "n_estimators": 50,
            "max_depth": 12,
            "n_jobs": -1,
            "random_state": 42,
        }
        params.update(kwargs)
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", RandomForestRegressor(**params)),
        ])

    elif name == "xgboost":
        params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": 42,
        }
        params.update(kwargs)
        return XGBRegressor(**params)

    elif name == "lightgbm":
        params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        }
        params.update(kwargs)
        return LGBMRegressor(**params)

    elif name == "catboost":
        params = {
            "iterations": 200,
            "learning_rate": 0.05,
            "depth": 6,
            "verbose": 0,
            "random_seed": 42,
        }
        params.update(kwargs)
        return CatBoostRegressor(**params)
    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. "
            f"Choose from: 'ridge', 'lasso', 'random_forest', 'xgboost', 'lightgbm', 'catboost'."
        )