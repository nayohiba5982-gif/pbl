"""
ベースモデル (Visual / Vital) 共通抽象基底クラス
(learning/downstream_task/base_model/models/base.py)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional
import joblib
import numpy as np
import pandas as pd


class BaseModalityRegressor(ABC):
    """
    Visual / Vital モダリティ別ベース回帰モデルの共通基底クラス。
    すべてのモデル (RandomForest, XGBoost, LightGBM, CatBoost, AdaBoost) で
    Early Stopping・サンプル重み付け学習・特徴量重要度抽出・モデル保存/読込を提供します。
    """

    def __init__(
        self,
        name: str,
        modality: str = "visual",
        params: Optional[dict] = None,
        early_stopping_rounds: int = 30,
    ):
        self.name = name
        self.modality = modality
        self.params = dict(params) if params else {}
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.model: Any = None
        self.feature_names: List[str] = []
        self.best_iteration_: Optional[int] = None

    @abstractmethod
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
    ) -> "BaseModalityRegressor":
        """Early Stopping 対応のモデル学習メソッド"""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """連続値予測スコアを返し、[0.0, 1.0] にクリップします"""
        if self.model is None:
            raise RuntimeError(f"モデル '{self.name}' ({self.modality}) は未学習です。")
        X_arr = np.asarray(X, dtype=np.float32)
        preds = self.model.predict(X_arr)
        return np.clip(np.asarray(preds, dtype=np.float64).ravel(), 0.0, 1.0)

    @abstractmethod
    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """特徴量重要度 DataFrame (Columns: ['feature', 'importance']) を降順で返します"""
        pass

    @staticmethod
    def _weighted_rmse(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        diff_sq = (np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)) ** 2
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float64)
            w_sum = np.sum(w)
            if w_sum > 0:
                return float(np.sqrt(np.sum(w * diff_sq) / w_sum))
        return float(np.sqrt(np.mean(diff_sq)))

    def save(self, output_path: str | Path, extra_metadata: Optional[dict] = None) -> Path:
        """モデルおよびメタデータを joblib 形式で保存します"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "modality": self.modality,
            "params": self.params,
            "early_stopping_rounds": self.early_stopping_rounds,
            "best_iteration": self.best_iteration_,
            "feature_names": self.feature_names,
            "model": self.model,
        }
        if extra_metadata:
            payload.update(extra_metadata)
        joblib.dump(payload, path)
        return path

    def load(self, model_path: str | Path) -> "BaseModalityRegressor":
        """保存済みモデルを読み込みます"""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {path}")
        payload = joblib.load(path)
        if isinstance(payload, dict) and "model" in payload:
            self.name = payload.get("name", self.name)
            self.modality = payload.get("modality", self.modality)
            self.params = payload.get("params", self.params)
            self.early_stopping_rounds = payload.get("early_stopping_rounds", self.early_stopping_rounds)
            self.best_iteration_ = payload.get("best_iteration")
            self.feature_names = payload.get("feature_names", [])
            self.model = payload["model"]
        else:
            self.model = payload
        return self
