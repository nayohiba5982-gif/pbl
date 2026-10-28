"""
ロジスティック回帰メタモデル (重み付きソフトターゲット交差エントロピー最小化)
(learning/downstream_task/meta_model/models/logistic_regression.py)
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import BaseMetaRegressor


class LogisticMetaRegressorModel(BaseMetaRegressor):
    """
    Visual / Vital 予測スコアを統合するロジスティック回帰メタモデル。
    連続値ソフトターゲット y_soft ∈ [0, 1] および sample_weight を受け取り、
    正例 (重み w * y) と負例 (重み w * (1 - y)) に展開して重み付き交差エントロピー損失を最小化し、
    推論時は連続確率 predict_proba(X)[:, 1] ∈ [0, 1] を返します。
    """

    def __init__(self, params: Optional[dict] = None):
        default_params = {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": 42,
        }
        if params:
            default_params.update(params)
        super().__init__(name="logistic_regression", params=default_params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "LogisticMetaRegressorModel":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.clip(np.asarray(y, dtype=np.float64).ravel(), 0.0, 1.0)
        if feature_names is not None:
            self.feature_names = list(feature_names)

        w_arr = (
            np.ones_like(y_arr, dtype=np.float64)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64).ravel()
        )

        X_dup = np.vstack([X_arr, X_arr])
        y_dup = np.concatenate([
            np.ones(len(y_arr), dtype=int),
            np.zeros(len(y_arr), dtype=int),
        ])
        w_dup = np.concatenate([w_arr * y_arr, w_arr * (1.0 - y_arr)])

        valid_mask = w_dup > 1e-12
        self.model = LogisticRegression(**self.params)
        self.model.fit(X_dup[valid_mask], y_dup[valid_mask], sample_weight=w_dup[valid_mask])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LogisticMetaRegressorModel は未学習です。")
        X_arr = np.asarray(X, dtype=np.float64)
        preds = self.model.predict_proba(X_arr)[:, 1]
        return np.clip(np.asarray(preds, dtype=np.float64).ravel(), 0.0, 1.0)

    def get_coefficients(self) -> Dict[str, float]:
        if self.model is None or not hasattr(self.model, "coef_"):
            return {}
        coefs = self.model.coef_.ravel()
        intercept = float(self.model.intercept_.ravel()[0]) if hasattr(self.model, "intercept_") else 0.0
        res = {
            self.feature_names[i] if i < len(self.feature_names) else f"w_{i}": float(coefs[i])
            for i in range(len(coefs))
        }
        res["intercept"] = intercept
        return res
