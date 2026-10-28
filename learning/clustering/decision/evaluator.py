"""
GMM モデル選択グリッド評価モジュール (GMM Model Selection Evaluator)

【責務】
- 指定されたクラスタ数 (k_min〜k_max) および 4共分散タイプ (full, tied, diag, spherical) の全探索
- BIC (ベイズ情報量規準), AIC (赤池情報量規準), シルエットスコアの算出
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

import sys
from pathlib import Path
_LEARNING_DIR = Path(__file__).resolve().parent.parent.parent
if str(_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(_LEARNING_DIR))

try:
    from config_loader import log
except ImportError:
    import logging
    log = logging.getLogger("clustering.decision.evaluator")

ALL_COVARIANCE_TYPES = ["full", "tied", "diag", "spherical"]


def evaluate_gmm_grid(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    covariance_types: list[str] | None = None,
    random_state: int = 42,
    max_iter: int = 100,
    n_init: int = 5,
    max_silhouette_samples: int = 10000,
) -> pd.DataFrame:
    """
    指定された共分散タイプおよび k_min から k_max までの各クラスタ数の
    すべての組み合わせに対して GMM を学習し、BIC, AIC, Silhouette Score を算出します。
    """
    if covariance_types is None:
        covariance_types = ALL_COVARIANCE_TYPES

    results = []
    n_samples = len(X)
    sample_size_for_silhouette = min(max_silhouette_samples, n_samples)

    log.info("=" * 60)
    log.info(f"▶ GMM モデル選択グリッド探索開始 (k: {k_min}〜{k_max}, 共分散: {covariance_types})")
    log.info("=" * 60)

    for cov_type in covariance_types:
        log.info(f"--- 共分散タイプ: [{cov_type}] の探索中 ---")
        for k in range(k_min, k_max + 1):
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=cov_type,
                    random_state=random_state,
                    max_iter=max_iter,
                    n_init=n_init,
                )
                gmm.fit(X)

                bic = gmm.bic(X)
                aic = gmm.aic(X)
                labels = gmm.predict(X)

                # シルエットスコア (クラスタ数が2以上かつ複数クラスタが予測された場合のみ)
                unique_labels = np.unique(labels)
                if len(unique_labels) > 1:
                    if n_samples > sample_size_for_silhouette:
                        rng = np.random.default_rng(random_state)
                        sub_indices = rng.choice(n_samples, size=sample_size_for_silhouette, replace=False)
                        sil = silhouette_score(X[sub_indices], labels[sub_indices])
                    else:
                        sil = silhouette_score(X, labels)
                else:
                    sil = -1.0

                results.append({
                    "k": k,
                    "covariance_type": cov_type,
                    "bic": bic,
                    "aic": aic,
                    "silhouette": sil,
                    "converged": gmm.converged_,
                    "n_iter": gmm.n_iter_,
                })

                log.info(
                    f"  [k={k:2d} | {cov_type:<9}] BIC: {bic:10.2f} | AIC: {aic:10.2f} | "
                    f"Silhouette: {sil:.4f} (収束: {gmm.converged_})"
                )

            except Exception as e:
                log.warning(f"  [k={k} | {cov_type}] フィッティング失敗: {e}")
                results.append({
                    "k": k,
                    "covariance_type": cov_type,
                    "bic": np.nan,
                    "aic": np.nan,
                    "silhouette": np.nan,
                    "converged": False,
                    "n_iter": 0,
                })

    df = pd.DataFrame(results)
    return df
