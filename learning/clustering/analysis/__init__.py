"""
GMM クラスタ特徴量集計・分析サブパッケージ (clustering.analysis)
"""

from .analyzer import ClusterAnalyzer, run_cluster_analysis
from .metrics import DEFAULT_FEATURE_COLS, attach_physical_metrics_to_df
from .statistics import compute_cross_tabulation, aggregate_cluster_profiles
from .plotter import plot_cluster_features
from .reporter import generate_cluster_report

__all__ = [
    "ClusterAnalyzer",
    "run_cluster_analysis",
    "DEFAULT_FEATURE_COLS",
    "attach_physical_metrics_to_df",
    "compute_cross_tabulation",
    "aggregate_cluster_profiles",
    "plot_cluster_features",
    "generate_cluster_report",
]
