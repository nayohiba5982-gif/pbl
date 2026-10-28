"""
RandomForest 回帰モデル (warm_start + Early Stopping 対応)
(learning/downstream_task/base_model/models/random_forest.py)
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .base import BaseModalityRegressor


class RandomForestBaseRegressor(BaseModalityRegressor):
    """
    RandomForestRegressor に warm_start を用いた段階的木追加による Early Stopping を実装したクラス。
    検証データ (X_val, y_val) の重み付き RMSE を監視し、指定ラウンド改善が見られない場合は
    ベストスコア時点の決定木本数 (best_iteration_) に切り詰めます。
    """

    def __init__(
        self,
        modality: str = "visual",
        params: Optional[dict] = None,
        early_stopping_rounds: int = 30,
    ):
        default_params = {
            "n_estimators": 200,
            "step_estimators": 5,
            "max_depth": 6,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": 1,
        }
        if params:
            default_params.update(params)
        super().__init__(
            name="random_forest",
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
    ) -> "RandomForestBaseRegressor":
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

        rf_kwargs = dict(self.params)
        max_trees = int(rf_kwargs.pop("n_estimators", 200))
        step_trees = max(1, int(rf_kwargs.pop("step_estimators", 5)))
        rf_kwargs.pop("early_stopping_rounds", None)

        if X_val is not None and y_val is not None and stop_rounds > 0 and max_trees > step_trees:
            X_v = np.asarray(X_val, dtype=np.float32)
            y_v = np.asarray(y_val, dtype=np.float64)

            rf_kwargs["warm_start"] = True
            rf_kwargs["n_estimators"] = step_trees
            self.model = RandomForestRegressor(**rf_kwargs)

            best_loss = float("inf")
            best_n_trees = step_trees
            trees_without_improvement = 0

            current_trees = step_trees
            while current_trees <= max_trees:
                self.model.set_params(n_estimators=current_trees)
                self.model.fit(X_tr, y_tr, sample_weight=sample_weight)

                val_preds = self.model.predict(X_v)
                val_loss = self._weighted_rmse(y_v, val_preds, sample_weight_val)

                if val_loss < best_loss - 1e-6:
                    best_loss = val_loss
                    best_n_trees = current_trees
                    trees_without_improvement = 0
                else:
                    trees_without_improvement += step_trees
                    if trees_without_improvement >= stop_rounds:
                        break

                if current_trees == max_trees:
                    break
                current_trees = min(max_trees, current_trees + step_trees)

            # 最適本数で決定木リストを切り詰め
            if hasattr(self.model, "estimators_") and len(self.model.estimators_) > best_n_trees:
                self.model.estimators_ = self.model.estimators_[:best_n_trees]
                self.model.n_estimators = best_n_trees
            self.best_iteration_ = best_n_trees
        else:
            rf_kwargs["warm_start"] = False
            rf_kwargs["n_estimators"] = max_trees
            self.model = RandomForestRegressor(**rf_kwargs)
            self.model.fit(X_tr, y_tr, sample_weight=sample_weight)
            self.best_iteration_ = max_trees

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
