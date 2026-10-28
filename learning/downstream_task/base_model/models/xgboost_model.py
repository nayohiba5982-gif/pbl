"""
XGBoost 回帰モデル (Early Stopping 対応)
(learning/downstream_task/base_model/models/xgboost_model.py)
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd
import xgboost as xgb

from .base import BaseModalityRegressor


class XGBoostBaseRegressor(BaseModalityRegressor):
    """
    XGBRegressor によるモダリティ別回帰モデル。
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
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "random_state": 42,
            "tree_method": "hist",
            "verbosity": 0,
            "n_jobs": 1,
        }
        if params:
            default_params.update(params)
        super().__init__(
            name="xgboost",
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
    ) -> "XGBoostBaseRegressor":
        X_tr = np.asarray(X_train, dtype=np.float32)
        y_tr = np.asarray(y_train, dtype=np.float32)
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

        xgb_kwargs = dict(self.params)
        xgb_kwargs.pop("early_stopping_rounds", None)

        has_val = X_val is not None and y_val is not None and len(y_val) > 0 and stop_rounds > 0
        if has_val:
            xgb_kwargs["early_stopping_rounds"] = stop_rounds

        self.model = xgb.XGBRegressor(**xgb_kwargs)

        fit_kwargs = {"verbose": False}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=np.float32)

        if has_val:
            X_v = np.asarray(X_val, dtype=np.float32)
            y_v = np.asarray(y_val, dtype=np.float32)
            fit_kwargs["eval_set"] = [(X_v, y_v)]
            if sample_weight_val is not None:
                fit_kwargs["sample_weight_eval_set"] = [np.asarray(sample_weight_val, dtype=np.float32)]

        self.model.fit(X_tr, y_tr, **fit_kwargs)

        best_iter = getattr(self.model, "best_iteration", None)
        if best_iter is not None:
            self.best_iteration_ = int(best_iter) + 1
        else:
            self.best_iteration_ = int(xgb_kwargs.get("n_estimators", 100))

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
