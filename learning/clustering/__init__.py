"""
集中度推定 GMM クラスタリング・分析・ソフトラベリング統合パッケージ (clustering)

【内包モジュール】
- decision.py: GMM 最適クラスタ数 (K) & 共分散タイプの自動探索 (ClusterNumberDecision)
- gmm.py: GMM クラスタリングの学習 (Trainのみ) & 推論 (GMMClustering)
- analysis/: クラスタごとの生体・物理特徴量集計 & レポート出力 (ClusterAnalyzer)
- probability/: GMM 所属確率と重みによるサンプル別期待値ソフトターゲット y ∈ [0, 1] 算出 (ProbabilityLabelCalculator)
"""

from .pipeline import ClusteringPipeline
from .decision import ClusterNumberDecision, determine_optimal_clusters
from .gmm import GMMClustering, run_clustering
from .analysis import ClusterAnalyzer, run_cluster_analysis
from .probability import (
    ProbabilityLabelCalculator,
    run_cal_probability,
    compute_soft_labels,
    generate_default_weights,
    find_latest_gmm_model,
    load_probabilities_for_split,
    plot_soft_label_distribution,
)

__all__ = [
    "ClusteringPipeline",
    # Step 2: クラスタ数探索
    "ClusterNumberDecision",
    "determine_optimal_clusters",
    # Step 3: GMM クラスタリング
    "GMMClustering",
    "run_clustering",
    # Step 4: クラスタ物理分析
    "ClusterAnalyzer",
    "run_cluster_analysis",
    # Step 5: 期待値ソフトラベル算出
    "ProbabilityLabelCalculator",
    "run_cal_probability",
    "compute_soft_labels",
    "generate_default_weights",
    "find_latest_gmm_model",
    "load_probabilities_for_split",
    "plot_soft_label_distribution",
]
