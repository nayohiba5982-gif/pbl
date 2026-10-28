"""
メタモデル (Visual + Vital 予測値統合) 共通抽象基底クラス
(learning/downstream_task/meta_model/models/base.py)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np


class BaseMetaRegressor(ABC):
    """
    Visual ベースモデルと Vital ベースモデルの予測値 [y_pred_visual, y_pred_vital] を入力とし、
    最終的な集中度スコア [0.0, 1.0] を推論するメタモデルの共通基底クラス。
    """

    def __init__(self, name: str, params: Optional[dict] = None):
        self.name = name
        self.params = dict(params) if params else {}
        self.model: Any = None
        self.feature_names: List[str] = ["visual_pred", "vital_pred"]

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "BaseMetaRegressor":
        """メタモデルを学習します"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """最終的な連続値集中度スコア [0.0, 1.0] を出力します"""
        pass

    @abstractmethod
    def get_coefficients(self) -> Dict[str, float]:
        """学習されたモダリティ重み (visual_weight, vital_weight, intercept) を返します"""
        pass

    def save(self, output_path: str | Path, extra_metadata: Optional[dict] = None) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "params": self.params,
            "feature_names": self.feature_names,
            "coefficients": self.get_coefficients() if self.model is not None else {},
            "model": self.model,
        }
        if extra_metadata:
            payload.update(extra_metadata)
        joblib.dump(payload, path)
        return path

    def load(self, model_path: str | Path) -> "BaseMetaRegressor":
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"メタモデルファイルが見つかりません: {path}")
        payload = joblib.load(path)
        if isinstance(payload, dict) and "model" in payload:
            self.name = payload.get("name", self.name)
            self.params = payload.get("params", self.params)
            self.feature_names = payload.get("feature_names", self.feature_names)
            self.model = payload["model"]
        else:
            self.model = payload
        return self
