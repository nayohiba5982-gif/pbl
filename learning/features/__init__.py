"""
learning/features パッケージ
ドメイン知識に基づく明示的物理特徴量 (時系列統計量 24次元: 瞬目比率・頭部姿勢・視線・心拍数・HRV主要4指標) の集約モジュール
"""

from .constants import (
    DEFAULT_FEATURE_COLS,
    DEFAULT_HRV_COLS,
    DEFAULT_HEAD_POSE_COLS,
    DEFAULT_GAZE_COLS,
    DEFAULT_EXPLICIT_STATS_DIMS,
    DEFAULT_VISUAL_DIMS,
    DEFAULT_VITAL_DIMS,
    DEFAULT_TOTAL_LATENT_DIMS,
)
from .explicit_stats import compute_explicit_statistics

# 後方互換性用エイリアス
explicit_statistics = explicit_stats

__all__ = [
    "compute_explicit_statistics",
    "DEFAULT_FEATURE_COLS",
    "DEFAULT_HRV_COLS",
    "DEFAULT_HEAD_POSE_COLS",
    "DEFAULT_GAZE_COLS",
    "DEFAULT_EXPLICIT_STATS_DIMS",
    "DEFAULT_VISUAL_DIMS",
    "DEFAULT_VITAL_DIMS",
    "DEFAULT_TOTAL_LATENT_DIMS",
    "explicit_stats",
    "explicit_statistics",
]
