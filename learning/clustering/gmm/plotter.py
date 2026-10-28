"""
GMM クラスタ散布図可視化モジュール (GMM Cluster Plotter)

【責務】
- PCA による 2次元圧縮およびクラスタ散布図の描画・画像保存
"""

from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

import sys
_LEARNING_DIR = Path(__file__).resolve().parent.parent.parent
if str(_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(_LEARNING_DIR))

try:
    from config_loader import log
except ImportError:
    import logging
    log = logging.getLogger("clustering.gmm.plotter")


def plot_pca_clusters(
    X: np.ndarray, labels: np.ndarray, n_clusters: int, save_path: Path, window_size: str = "10s"
):
    """PCAで2次元に圧縮してクラスタの散布図を描画・保存します"""
    try:
        import matplotlib.pyplot as plt

        log.info("PCA による 2 次元圧縮およびクラスタ散布図の作成中...")
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)

        var_ratio = pca.explained_variance_ratio_
        log.info(f"PCA 寄与率: PC1 = {var_ratio[0]:.2%}, PC2 = {var_ratio[1]:.2%} (合計: {np.sum(var_ratio):.2%})")

        plt.figure(figsize=(9, 7))
        cmap = plt.get_cmap("tab10" if n_clusters <= 10 else "tab20")

        for k in range(n_clusters):
            mask = labels == k
            plt.scatter(
                X_2d[mask, 0], X_2d[mask, 1], label=f"Cluster {k} (n={np.sum(mask)})",
                color=cmap(k), alpha=0.6, edgecolors="none", s=25,
            )

        plt.xlabel(f"PC1 ({var_ratio[0]:.1%})", fontsize=11)
        plt.ylabel(f"PC2 ({var_ratio[1]:.1%})", fontsize=11)
        plt.title(f"GMM Clustering on TS2Vec Embeddings (k={n_clusters}, Window={window_size})", fontsize=13)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        log.info(f"📊 クラスタ散布図プロットを保存しました: {save_path}")

    except Exception as e:
        log.warning(f"散布図プロット作成中にエラーが発生しました: {e}")
