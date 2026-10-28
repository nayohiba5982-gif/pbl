"""
Lasso 回帰メタモデル
(learning/downstream_task/meta_model/models/lasso.py)
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
from sklearn.linear_model import Lasso

from .base import BaseMetaRegressor


class LassoMetaRegressorModel(BaseMetaRegressor):
    """L1 正則化線形回帰 (Lasso) によるメタモデル"""

    def __init__(self, params: Optional[dict] = None):
        default_params = {
            "alpha": 0.01,
            "max_iter": 2000,
            "tol": 0.0001,
            "random_state": 42,
        }
        if params:
            default_params.update(params)
        super().__init__(name="lasso", params=default_params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "LassoMetaRegressorModel":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        if feature_names is not None:
            self.feature_names = list(feature_names)

        self.model = Lasso(**self.params)
        if sample_weight is not None:
            try:
                self.model.fit(X_arr, y_arr, sample_weight=np.asarray(sample_weight, dtype=np.float64).ravel())
            except TypeError:
                self.model.fit(X_arr, y_arr)
        else:
            self.model.fit(X_arr, y_arr)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LassoMetaRegressorModel は未学習です。")
        X_arr = np.asarray(X, dtype=np.float64)
        preds = self.model.predict(X_arr)
        return np.clip(np.asarray(preds, dtype=np.float64).ravel(), 0.0, 1.0)

    def get_coefficients(self) -> Dict[str, float]:
        if self.model is None or not hasattr(self.model, "coef_"):
            return {}
        coefs = np.asarray(self.model.coef_).ravel()
        intercept = float(np.asarray(self.model.intercept_).ravel()[0]) if hasattr(self.model, "intercept_") else 0.0
        res = {
            self.feature_names[i] if i < len(self.feature_names) else f"w_{i}": float(coefs[i])
            for i in range(len(coefs))
        }
        res["intercept"] = intercept
        return res
