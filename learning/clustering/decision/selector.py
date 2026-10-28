"""
最適 GMM モデル選定モジュール (Optimal Model Selector)

【責務】
- グリッド評価結果テーブル (df_metrics) から、指定基準 (BIC, AIC, Silhouette) に基づく最適 (k, cov_type) の選定
"""

import numpy as np
import pandas as pd

import sys
from pathlib import Path
_LEARNING_DIR = Path(__file__).resolve().parent.parent.parent
if str(_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(_LEARNING_DIR))

try:
    from config_loader import log
except ImportError:
    import logging
    log = logging.getLogger("clustering.decision.selector")


def select_optimal_model(
    df_metrics: pd.DataFrame, criterion: str = "bic"
) -> tuple[int, str]:
    """
    評価テーブルから指定基準に基づいて最適な (k, covariance_type) を選択します。
    """
    valid_df = df_metrics.dropna(subset=["bic", "aic", "silhouette"])
    if valid_df.empty:
        raise ValueError("すべてのモデル評価が失敗したため、最適モデルを決定できません。")

    crit = criterion.lower()

    if crit == "bic":
        best_idx = valid_df["bic"].idxmin()
        best_row = valid_df.loc[best_idx]
        log.info(
            f"★ 最適モデル選定 (BIC最小): k={int(best_row['k'])}, "
            f"共分散='{best_row['covariance_type']}' (BIC: {best_row['bic']:.2f})"
        )
    elif crit == "aic":
        best_idx = valid_df["aic"].idxmin()
        best_row = valid_df.loc[best_idx]
        log.info(
            f"★ 最適モデル選定 (AIC最小): k={int(best_row['k'])}, "
            f"共分散='{best_row['covariance_type']}' (AIC: {best_row['aic']:.2f})"
        )
    elif crit == "silhouette":
        best_idx = valid_df["silhouette"].idxmax()
        best_row = valid_df.loc[best_idx]
        log.info(
            f"★ 最適モデル選定 (Silhouette最大): k={int(best_row['k'])}, "
            f"共分散='{best_row['covariance_type']}' (Silhouette: {best_row['silhouette']:.4f})"
        )
    else:
        log.warning(f"未知の基準 '{criterion}' が指定されたため、デフォルトの 'bic' を使用します。")
        best_idx = valid_df["bic"].idxmin()
        best_row = valid_df.loc[best_idx]

    optimal_k = int(best_row["k"])
    optimal_cov = str(best_row["covariance_type"])

    return optimal_k, optimal_cov
