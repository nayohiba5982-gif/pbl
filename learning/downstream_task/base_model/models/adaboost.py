"""
AdaBoost 回帰モデル (staged_predict による Early Stopping 対応)
(learning/downstream_task/base_model/models/adaboost.py)
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

from .base import BaseModalityRegressor


class AdaBoostBaseRegressor(BaseModalityRegressor):
    """
    AdaBoostRegressor に staged_predict を用いた検証ロス監視型 Early Stopping を実装したクラス。
    各ステージ (1〜T) の検証データ (X_val, y_val) に対する重み付き RMSE を計算し、
    指定ラウンド改善が見られない場合はベストステージ (best_iteration_) で
    estimators_ と estimator_weights_ を切り詰めます。
    """

    def __init__(
        self,
        modality: str = "visual",
        params: Optional[dict] = None,
        early_stopping_rounds: int = 30,
    ):
        default_params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "loss": "linear",
            "max_depth": 3,
            "min_samples_leaf": 5,
            "random_state": 42,
        }
        if params:
            default_params.update(params)
        super().__init__(
            name="adaboost",
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
    ) -> "AdaBoostBaseRegressor":
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

        ada_kwargs = dict(self.params)
        ada_kwargs.pop("early_stopping_rounds", None)
        max_depth = int(ada_kwargs.pop("max_depth", 3))
        min_samples_leaf = int(ada_kwargs.pop("min_samples_leaf", 5))
        random_state = ada_kwargs.get("random_state", 42)

        base_tree = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        self.model = AdaBoostRegressor(estimator=base_tree, **ada_kwargs)
        self.model.fit(X_tr, y_tr, sample_weight=sample_weight)

        n_fitted = len(self.model.estimators_)
        if X_val is not None and y_val is not None and len(y_val) > 0 and stop_rounds > 0 and n_fitted > 1:
            X_v = np.asarray(X_val, dtype=np.float32)
            y_v = np.asarray(y_val, dtype=np.float64)

            best_loss = float("inf")
            best_stage = 1
            no_improve_count = 0

            for stage_idx, stage_pred in enumerate(self.model.staged_predict(X_v), start=1):
                loss = self._weighted_rmse(y_v, stage_pred, sample_weight_val)
                if loss < best_loss - 1e-6:
                    best_loss = loss
                    best_stage = stage_idx
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= stop_rounds:
                        break

            # ベストステージで estimators_ / estimator_weights_ / estimator_errors_ を切り詰め
            self.model.estimators_ = self.model.estimators_[:best_stage]
            self.model.estimator_weights_ = self.model.estimator_weights_[:best_stage]
            self.model.estimator_errors_ = self.model.estimator_errors_[:best_stage]
            self.best_iteration_ = best_stage
        else:
            self.best_iteration_ = n_fitted

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
