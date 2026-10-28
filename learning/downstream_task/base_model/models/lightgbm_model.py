"""
LightGBM 回帰モデル (Early Stopping 対応)
(learning/downstream_task/base_model/models/lightgbm_model.py)
"""

from __future__ import annotations
from typing import List, Optional
import lightgbm as lgb
import numpy as np
import pandas as pd

from .base import BaseModalityRegressor


class LightGBMBaseRegressor(BaseModalityRegressor):
    """
    LGBMRegressor によるモダリティ別回帰モデル。
    検証セット (X_val, y_val) を用いた Early Stopping およびサンプル重み付けに対応します。
    """

    def __init__(
        self,
        modality: str = "visual",
        params: Optional[dict] = None,
        early_stopping_rounds: int = 30,
    ):
        default_params = {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 25,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
            "objective": "regression",
            "metric": "rmse",
            "random_state": 42,
            "n_jobs": 1,
            "verbose": -1,
        }
        if params:
            default_params.update(params)
        super().__init__(
            name="lightgbm",
            modality=modality,
            params=default_params,
            early_stopping_rounds=early_stopping_rounds,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight_val: Optional[np.ndarray] = None,
        early_stopping_rounds: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "LightGBMBaseRegressor":
        X_tr = np.asarray(X_train, dtype=np.float32)
        y_tr = np.asarray(y_train, dtype=np.float64)
        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else [f"f_{i}" for i in range(X_tr.shape[1])]
        )

        stop_rounds = (
            int(early_stopping_rounds)
            if early_stopping_rounds is not None
            else self.early_stopping_rounds
        )

        lgb_kwargs = dict(self.params)
        lgb_kwargs.pop("early_stopping_rounds", None)
        lgb_kwargs.setdefault("verbose", -1)
        lgb_kwargs["n_jobs"] = max(1, int(lgb_kwargs.get("n_jobs", 1)))

        self.model = lgb.LGBMRegressor(**lgb_kwargs)

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=np.float64)

        has_val = X_val is not None and y_val is not None and len(y_val) > 0 and stop_rounds > 0
        if has_val:
            X_v = np.asarray(X_val, dtype=np.float32)
            y_v = np.asarray(y_val, dtype=np.float64)
            fit_kwargs["eval_set"] = [(X_v, y_v)]
            if sample_weight_val is not None:
                fit_kwargs["eval_sample_weight"] = [np.asarray(sample_weight_val, dtype=np.float64)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=stop_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ]

        self.model.fit(X_tr, y_tr, **fit_kwargs)

        best_iter = getattr(self.model, "best_iteration_", None)
        if best_iter is not None and best_iter > 0:
            self.best_iteration_ = int(best_iter)
        else:
            self.best_iteration_ = int(lgb_kwargs.get("n_estimators", 100))

        return self

    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return pd.DataFrame(columns=["feature", "importance"])
        names = feature_names or self.feature_names
        importances = self.model.feature_importances_
        if len(names) != len(importances):
            names = [f"f_{i}" for i in range(len(importances))]
        return (
            pd.DataFrame({"feature": names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
