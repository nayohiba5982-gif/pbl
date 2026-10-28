"""
Dual-Stream PLS (Partial Least Squares) 次元削減モジュール
(learning/downstream_task/pls.py)

視覚ストリーム (32d) と生体ストリーム (32d) を個別に教師ラベル y に対して射影し、
指定次元（例: 3d + 3d = 6d）に圧縮します。
"""

from __future__ import annotations
import numpy as np
from sklearn.cross_decomposition import PLSRegression


class DualStreamPLS:
    """視覚・生体ストリームを個別に次元削減する PLS クラス"""

    def __init__(self, n_vis: int = 3, n_bio: int = 3, scale: bool = True):
        self.n_vis = n_vis
        self.n_bio = n_bio
        self.scale = scale
        self.pls_vis = PLSRegression(n_components=n_vis, scale=scale)
        self.pls_bio = PLSRegression(n_components=n_bio, scale=scale)

    def fit(self, X_vis: np.ndarray, X_bio: np.ndarray, y: np.ndarray):
        """Train データのみを用いて視覚・生体の PLS 射影軸を学習"""
        self.pls_vis.fit(X_vis, y)
        self.pls_bio.fit(X_bio, y)
        return self

    def transform(self, X_vis: np.ndarray, X_bio: np.ndarray) -> np.ndarray:
        """学習済み射影軸に従って潜在特徴量 (z_vis, z_bio) を抽出し水平結合"""
        z_v, _ = self.pls_vis.transform(X_vis, np.zeros((len(X_vis), 1)))
        z_b, _ = self.pls_bio.transform(X_bio, np.zeros((len(X_bio), 1)))
        return np.concatenate([z_v, z_b], axis=1).astype(np.float32)
