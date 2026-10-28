"""
GMM クラスタリング実行オーケストレータ (GMM Clustering Runner)

【責務】
- データローダー (data_loader) からのデータ取得
- 最適クラスタ数決定 (decision), GMM学習 (trainer), 事後確率付与 (predictor), 可視化 (plotter) の統括
"""

from pathlib import Path
import joblib
import numpy as np

import sys
_LEARNING_DIR = Path(__file__).resolve().parent.parent.parent
if str(_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(_LEARNING_DIR))

try:
    from config_loader import load_config, log
except ImportError:
    import logging
    log = logging.getLogger("clustering.gmm")
    def load_config(f="config.yaml"): return {}

from ..data_loader import load_clustering_data
from ..decision import determine_optimal_clusters
from .trainer import fit_gmm
from .predictor import apply_and_save_clusters
from .plotter import plot_pca_clusters


class GMMClustering:
    """GMM クラスタリング実行および確率ラベル付与オーケストレータクラス"""

    def __init__(self, cfg: dict, weights: list[float] | None = None):
        self.cfg = cfg
        self.paths = cfg.get("paths", {})
        self.dataset_cfg = cfg.get("dataset", {})
        self.clustering_cfg = cfg.get("clustering", {})
        self.gmm_cfg = self.clustering_cfg.get("gmm", {})
        self.soft_cfg = cfg.get("soft_labeling", {})

        self.window_size = self.dataset_cfg.get("window_size", "10s")
        data_root = self.paths.get("data_root", _LEARNING_DIR / "data")
        self.data_dir = Path(data_root) / self.window_size
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path(data_root) / self.window_size

        self.weights = weights if weights is not None else self.soft_cfg.get("weights", None)
        self.target_col = self.soft_cfg.get("target_col", "y_soft")
        self.save_npy = self.soft_cfg.get("save_npy", True)

        self.model_dir = Path(self.paths.get("model_dir", "models"))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.save_scatter_plot = self.clustering_cfg.get("save_scatter_plot", True)
        self.scatter_filename = self.clustering_cfg.get(
            "scatter_filename", f"cluster_scatter_pca_{self.window_size}.png"
        )

    def load_split_data(self) -> dict | None:
        """共通データローダーからデータを取得"""
        return load_clustering_data(self.cfg, window_size=self.window_size)

    def get_or_determine_model_params(
        self, n_clusters: int | None = None, X_train: np.ndarray = None, covariance_type: str | None = None
    ) -> tuple[int | None, str]:
        """クラスタ数および共分散タイプを決定"""
        cov_type = covariance_type or self.gmm_cfg.get("covariance_type", "full")
        if covariance_type is not None:
            log.info(f"引数より共分散タイプを指定: cov = '{covariance_type}'")

        if n_clusters is not None:
            log.info(f"引数よりクラスタ数を指定: k = {n_clusters}")
            return int(n_clusters), cov_type

        cfg_k = self.clustering_cfg.get("n_clusters")
        if cfg_k is not None:
            log.info(f"config.yaml よりクラスタ数を読み込み: k = {cfg_k}")
            return int(cfg_k), cov_type

        log.info("クラスタ数が未指定のため、decision による自動探索を開始します...")
        res = determine_optimal_clusters(self.cfg, X=X_train, return_tuple=True)
        if res is not None:
            optimal_k, optimal_cov = res
            return optimal_k, optimal_cov

        return None, cov_type

    def run(
        self,
        n_clusters: int | None = None,
        weights: list[float] | None = None,
        covariance_type: str | None = None,
    ) -> bool:
        """GMM クラスタリングパイプラインの全工程を実行"""
        log.info("=" * 60)
        log.info(f"▶ GMM Clustering パイプライン開始 [Window: {self.window_size}]")
        log.info("=" * 60)

        # 1. データ読み込み (previous_task で被験者別Zスコア標準化済みのデータを取得)
        split_data = self.load_split_data()
        if split_data is None:
            return False

        # 【厳格なデータリークフリー設計】
        X_train = split_data["train"]["emb"]
        log.info(f"🔒 [Leakage-Free] GMM の学習 (fit) は previous_task 標準化済み Train データのみ (N={len(X_train)}) で実行します")

        # 2. クラスタ数 K & 共分散タイプの決定
        k, cov_type = self.get_or_determine_model_params(n_clusters=n_clusters, X_train=X_train, covariance_type=covariance_type)
        if k is None:
            log.error("❌ クラスタ数を決定できませんでした。処理を中断します。")
            return False

        log.info(f"使用モデル設定: k = {k}, 共分散タイプ = '{cov_type}'")

        # 3. GMM モデルの学習 (Train のみ)
        gmm = fit_gmm(
            X_train=X_train,
            n_clusters=k,
            cov_type=cov_type,
            random_state=self.gmm_cfg.get("random_state", 42),
            max_iter=self.gmm_cfg.get("max_iter", 100),
            n_init=self.gmm_cfg.get("n_init", 5),
        )

        # 4. モデル保存
        model_save_path = self.model_dir / f"gmm_model_{self.window_size}_k{k}_{cov_type}.joblib"
        joblib.dump(gmm, model_save_path)
        log.info(f"💾 GMM モデルを保存しました: {model_save_path}")

        # 5. 各 Split への適用 & 保存
        clustered_dfs = apply_and_save_clusters(
            gmm=gmm,
            split_data=split_data,
            n_clusters=k,
            weights=weights or self.weights,
            target_col=self.target_col,
            save_npy=self.save_npy,
            data_dir=self.data_dir,
            window_size=self.window_size,
        )

        # 6. 可視化 (PCA 散布図)
        if self.save_scatter_plot:
            train_labels = clustered_dfs["train"]["cluster"].values
            plot_path = self.data_dir / self.scatter_filename
            plot_pca_clusters(X_train, train_labels, n_clusters=k, save_path=plot_path, window_size=self.window_size)
            pls_dir = _LEARNING_DIR / "data" / "pls"
            if pls_dir.exists():
                plot_pca_clusters(X_train, train_labels, n_clusters=k, save_path=pls_dir / self.scatter_filename, window_size=self.window_size)

        log.info("✅ GMM クラスタリング・確率ラベル出力処理が正常に完了しました！")
        return True


def run_clustering(
    cfg: dict = None,
    n_clusters: int | None = None,
    weights: list[float] | None = None,
    covariance_type: str | None = None,
) -> bool:
    if cfg is None:
        cfg = load_config()
    clustering_task = GMMClustering(cfg, weights=weights)
    return clustering_task.run(n_clusters=n_clusters, weights=weights, covariance_type=covariance_type)
