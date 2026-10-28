"""
learning/downstream_task/base_model パッケージ初期化モジュール
Visual / Vital 各モダリティのベース回帰モデル (RandomForest, XGBoost, LightGBM, CatBoost, AdaBoost)
"""

from .models import (
    BaseModalityRegressor,
    RandomForestBaseRegressor,
    XGBoostBaseRegressor,
    LightGBMBaseRegressor,
    CatBoostBaseRegressor,
    AdaBoostBaseRegressor,
    DEFAULT_BASE_MODELS,
    BASE_MODEL_REGISTRY,
    normalize_base_model_name,
    create_base_model,
)
from .runner import LOSOFoldData, prepare_loso_folds, run_single_base_model_loso
from .pipeline import BaseModelPipeline
from .reporter import (
    save_modality_comparison_report,
    save_overall_base_model_report,
    plot_dual_modality_comparison,
)

__all__ = [
    "BaseModalityRegressor",
    "RandomForestBaseRegressor",
    "XGBoostBaseRegressor",
    "LightGBMBaseRegressor",
    "CatBoostBaseRegressor",
    "AdaBoostBaseRegressor",
    "DEFAULT_BASE_MODELS",
    "BASE_MODEL_REGISTRY",
    "normalize_base_model_name",
    "create_base_model",
    "LOSOFoldData",
    "prepare_loso_folds",
    "run_single_base_model_loso",
    "BaseModelPipeline",
    "save_modality_comparison_report",
    "save_overall_base_model_report",
    "plot_dual_modality_comparison",
]
