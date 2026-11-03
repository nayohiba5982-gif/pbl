"""
learning/downstream_task/threshold_decision パッケージ初期化モジュール

LOSO-CV の各 Fold 内で検証データ（Val）からデータリークを起こさずに
最適カットオフ閾値 th* ∈ [0, 1] を決定する機能を提供します。
  - ① Youden's J index (`youden_j`): J(th) = Recall(th) + Specificity(th) - 1
  - ② F1-Macro 最大化 (`f1_macro`): argmax_th F1-Macro(th)
  - ③ Precision-Recall バランス (`pr_balance`): |Precision - Recall| 最小化 または Precision * Recall 最大化
"""

from .criteria import (
    SUPPORTED_CRITERIA,
    CRITERION_LABELS,
    normalize_criterion_name,
    build_candidate_thresholds,
    evaluate_threshold_candidates,
    select_best_threshold_from_curve,
    find_optimal_threshold,
)
from .runner import (
    select_validation_indices_for_fold,
    run_loso_threshold_tuning,
)
from .reporter import (
    generate_threshold_visualizations,
    save_single_threshold_report,
    save_threshold_decision_summary_report,
)
from .pipeline import ThresholdDecisionPipeline, THREE_MAJOR_CRITERIA
from .main import main

__all__ = [
    "SUPPORTED_CRITERIA",
    "CRITERION_LABELS",
    "THREE_MAJOR_CRITERIA",
    "normalize_criterion_name",
    "build_candidate_thresholds",
    "evaluate_threshold_candidates",
    "select_best_threshold_from_curve",
    "find_optimal_threshold",
    "select_validation_indices_for_fold",
    "run_loso_threshold_tuning",
    "generate_threshold_visualizations",
    "save_single_threshold_report",
    "save_threshold_decision_summary_report",
    "ThresholdDecisionPipeline",
    "main",
]
