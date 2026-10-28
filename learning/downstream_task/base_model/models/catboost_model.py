"""
CatBoost 回帰モデル (Early Stopping 対応)
(learning/downstream_task/base_model/models/catboost_model.py)
"""

from __future__ import annotations
from typing import List, Optional
from catboost import CatBoostRegressor, Pool
import numpy as np
import pandas as pd

from .base import BaseModalityRegressor


class CatBoostBaseRegressor(BaseModalityRegressor):
    """
    CatBoostRegressor によるモダリティ別回帰モデル。
    検証セット (X_val, y_val) を用いた Early Stopping およびサンプル重み付けに対応します。
    """

    def __init__(
        self,
        modality: str = "visual",
        params: Optional[dict] = None,
        early_stopping_rounds: int = 30,
    ):
        default_params = {
            "iterations": 300,
            "learning_rate": 0.03,
            "depth": 4,
            "l2_leaf_reg": 3.0,
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": 42,
            "verbose": False,
            "thread_count": 1,
            "allow_writing_files": False,
        }
        if params:
            default_params.update(params)
        super().__init__(
            name="catboost",
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
    ) -> "CatBoostBaseRegressor":
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

        cb_kwargs = dict(self.params)
        cb_kwargs.pop("early_stopping_rounds", None)
        cb_kwargs.setdefault("verbose", False)
        cb_kwargs.setdefault("allow_writing_files", False)

        self.model = CatBoostRegressor(**cb_kwargs)

        train_pool = Pool(
            data=X_tr,
            label=y_tr,
            weight=np.asarray(sample_weight, dtype=np.float64) if sample_weight is not None else None,
            feature_names=self.feature_names,
        )

        has_val = X_val is not None and y_val is not None and len(y_val) > 0 and stop_rounds > 0
        if has_val:
            val_pool = Pool(
                data=np.asarray(X_val, dtype=np.float32),
                label=np.asarray(y_val, dtype=np.float64),
                weight=np.asarray(sample_weight_val, dtype=np.float64) if sample_weight_val is not None else None,
                feature_names=self.feature_names,
            )
            self.model.fit(
                train_pool,
                eval_set=val_pool,
                early_stopping_rounds=stop_rounds,
                use_best_model=True,
                verbose=False,
            )
        else:
            self.model.fit(train_pool, verbose=False)

        best_iter = self.model.get_best_iteration()
        if best_iter is not None and best_iter >= 0:
            self.best_iteration_ = int(best_iter) + 1
        else:
            self.best_iteration_ = int(self.model.tree_count_ or cb_kwargs.get("iterations", 150))

        return self

    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if self.model is None or not hasattr(self.model, "get_feature_importance"):
            return pd.DataFrame(columns=["feature", "importance"])
        names = feature_names or self.feature_names
        importances = self.model.get_feature_importance()
        if len(names) != len(importances):
            names = [f"f_{i}" for i in range(len(importances))]
        return (
            pd.DataFrame({"feature": names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
