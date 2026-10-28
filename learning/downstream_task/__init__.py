"""
learning/downstream_task パッケージ初期化モジュール
Visual / Vital モダリティ別 Base Model (Early Stopping 対応 5種) & Meta Model (4種) 2段階構成
"""

from .pipeline import DownstreamTaskPipeline, LOSORegressionPipeline, DEFAULT_DOWNSTREAM_MODELS
from .pls import DualStreamPLS
from .data_loader import (
    load_embeddings_and_meta,
    load_explicit_features,
    split_modality_feature_indices,
    build_modality_feature_names,
    standardize_subject_wise,
)
from .metrics import (
    compute_fold_metrics,
    compute_oof_and_fold_summary,
    build_summary_row,
)
from .reporter import (
    plot_model_comparison,
    generate_visualizations,
    save_markdown_report,
)
from .base_model import (
    BaseModelPipeline,
    BaseModalityRegressor,
    DEFAULT_BASE_MODELS,
    create_base_model,
)
from .meta_model import (
    MetaModelPipeline,
    BaseMetaRegressor,
    DEFAULT_META_MODELS,
    create_meta_model,
)
from .main import main

__all__ = [
    "DownstreamTaskPipeline",
    "LOSORegressionPipeline",
    "DEFAULT_DOWNSTREAM_MODELS",
    "DualStreamPLS",
    "load_embeddings_and_meta",
    "load_explicit_features",
    "split_modality_feature_indices",
    "build_modality_feature_names",
    "standardize_subject_wise",
    "compute_fold_metrics",
    "compute_oof_and_fold_summary",
    "build_summary_row",
    "plot_model_comparison",
    "generate_visualizations",
    "save_markdown_report",
    "BaseModelPipeline",
    "BaseModalityRegressor",
    "DEFAULT_BASE_MODELS",
    "create_base_model",
    "MetaModelPipeline",
    "BaseMetaRegressor",
    "DEFAULT_META_MODELS",
    "create_meta_model",
    "main",
]
