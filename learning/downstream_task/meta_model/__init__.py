"""
learning/downstream_task/meta_model パッケージ初期化モジュール
Visual / Vital ベースモデルの予測値を統合するメタモデル (LogisticRegression, Ridge, Lasso, ElasticNet)
"""

from .models import (
    BaseMetaRegressor,
    LogisticMetaRegressorModel,
    RidgeMetaRegressorModel,
    LassoMetaRegressorModel,
    ElasticNetMetaRegressorModel,
    DEFAULT_META_MODELS,
    META_MODEL_REGISTRY,
    normalize_meta_model_name,
    create_meta_model,
)
from .runner import run_single_meta_model_loso
from .pipeline import MetaModelPipeline
from .reporter import save_meta_model_comparison_report

__all__ = [
    "BaseMetaRegressor",
    "LogisticMetaRegressorModel",
    "RidgeMetaRegressorModel",
    "LassoMetaRegressorModel",
    "ElasticNetMetaRegressorModel",
    "DEFAULT_META_MODELS",
    "META_MODEL_REGISTRY",
    "normalize_meta_model_name",
    "create_meta_model",
    "run_single_meta_model_loso",
    "MetaModelPipeline",
    "save_meta_model_comparison_report",
]
