"""
GMM モデル学習モジュール (GMM Trainer)

【責務】
- Train データのみを用いた GaussianMixture モデルの初期化とフィッティング (Strict Leakage-Free)
"""

import numpy as np
from sklearn.mixture import GaussianMixture

import sys
from pathlib import Path
_LEARNING_DIR = Path(__file__).resolve().parent.parent.parent
if str(_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(_LEARNING_DIR))

try:
    from config_loader import log
except ImportError:
    import logging
    log = logging.getLogger("clustering.gmm.trainer")


def fit_gmm(
    X_train: np.ndarray,
    n_clusters: int,
    cov_type: str = "full",
    random_state: int = 42,
    max_iter: int = 100,
    n_init: int = 5,
) -> GaussianMixture:
    """
    GMM モデルを初期化し、Train データのみで学習します。
    """
    log.info(f"GMM 学習開始 (k = {n_clusters}, 共分散: {cov_type}, max_iter: {max_iter})")
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type=cov_type,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init,
    )
    gmm.fit(X_train)
    log.info(f"✓ GMM 学習完了 (収束: {gmm.converged_}, 反復回数: {gmm.n_iter_})")
    return gmm
