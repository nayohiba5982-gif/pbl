"""
最適クラスタ数・モデル探索オーケストレータ (Cluster Number Decision Runner)

【責務】
- 設定の読み込みとデータローダーからの Train 特徴量取得
- グリッド評価 (evaluator), 最適選定 (selector), グラフ可視化 (plotter) の統括実行
"""

from pathlib import Path
import numpy as np
import pandas as pd

import sys
_LEARNING_DIR = Path(__file__).resolve().parent.parent.parent
if str(_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(_LEARNING_DIR))

try:
    from config_loader import load_config, log
except ImportError:
    import logging
    log = logging.getLogger("clustering.decision")
    def load_config(f="config.yaml"): return {}

from ..data_loader import load_train_embeddings
from .evaluator import evaluate_gmm_grid, ALL_COVARIANCE_TYPES
from .selector import select_optimal_model
from .plotter import plot_model_selection_metrics


class ClusterNumberDecision:
    """GMM のクラスタ数および共分散タイプの自動探索オーケストレータクラス"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.paths = cfg.get("paths", {})
        self.dataset_cfg = cfg.get("dataset", {})
        self.clustering_cfg = cfg.get("clustering", {})
        self.decision_cfg = self.clustering_cfg.get("decision", {})
        self.gmm_cfg = self.clustering_cfg.get("gmm", {})

        self.window_size = self.dataset_cfg.get("window_size", "10s")
        data_root = self.paths.get("data_root", _LEARNING_DIR / "data")
        self.data_dir = Path(data_root) / self.window_size
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path(data_root) / self.window_size

        self.k_min = int(self.decision_cfg.get("k_min", 2))
        self.k_max = int(self.decision_cfg.get("k_max", 10))
        self.covariance_types = self.decision_cfg.get("covariance_types", ALL_COVARIANCE_TYPES)
        self.criterion = self.decision_cfg.get("criterion", "bic").lower()
        self.save_plot = self.decision_cfg.get("save_plot", True)

        raw_csv = self.decision_cfg.get("csv_filename", "cluster_selection_metrics.csv")
        raw_plot = self.decision_cfg.get("plot_filename", "cluster_selection_metrics.png")
        self.csv_filename = f"{Path(raw_csv).stem}_{self.window_size}{Path(raw_csv).suffix}" if self.window_size not in raw_csv else raw_csv
        self.plot_filename = f"{Path(raw_plot).stem}_{self.window_size}{Path(raw_plot).suffix}" if self.window_size not in raw_plot else raw_plot

        self.optimal_k: int | None = None
        self.optimal_cov_type: str | None = None
        self.df_metrics: pd.DataFrame | None = None

    def load_embeddings(self) -> np.ndarray | None:
        """データローダーモジュールを呼び出して Train 特徴量を取得"""
        return load_train_embeddings(self.cfg, window_size=self.window_size)

    def evaluate_clusters(self, X: np.ndarray) -> pd.DataFrame:
        """グリッド評価の実行"""
        return evaluate_gmm_grid(
            X=X,
            k_min=self.k_min,
            k_max=self.k_max,
            covariance_types=self.covariance_types,
            random_state=self.gmm_cfg.get("random_state", 42),
            max_iter=self.gmm_cfg.get("max_iter", 100),
            n_init=self.gmm_cfg.get("n_init", 5),
        )

    def determine_optimal_model(self, df_metrics: pd.DataFrame) -> tuple[int, str]:
        """最適モデル選定"""
        return select_optimal_model(df_metrics, criterion=self.criterion)

    def plot_metrics(self, df_metrics: pd.DataFrame, optimal_k: int, optimal_cov: str, save_path: Path):
        """可視化プロット生成"""
        plot_model_selection_metrics(
            df_metrics=df_metrics,
            optimal_k=optimal_k,
            optimal_cov=optimal_cov,
            save_path=save_path,
            window_size=self.window_size,
            k_min=self.k_min,
            k_max=self.k_max,
            covariance_types=self.covariance_types,
        )

    def run(self, X: np.ndarray = None, return_tuple: bool = False) -> int | tuple[int, str] | None:
        """探索パイプラインの全工程を実行"""
        log.info("=" * 60)
        log.info(f"▶ Decision Class Number (4共分散タイプ & 最適クラスタ数探索) 開始 [Window: {self.window_size}]")
        log.info("=" * 60)

        if X is None:
            X = self.load_embeddings()
        if X is None:
            return None

        # 1. グリッド評価
        self.df_metrics = self.evaluate_clusters(X)

        # 2. 最適モデル決定
        self.optimal_k, self.optimal_cov_type = self.determine_optimal_model(self.df_metrics)

        # 3. CSV 保存
        csv_path = self.data_dir / self.csv_filename
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.df_metrics.to_csv(csv_path, index=False)
        log.info(f"💾 評価結果テーブルを保存しました: {csv_path}")

        # PLS フォルダにも複製保存
        pls_dir = _LEARNING_DIR / "data" / "pls"
        if pls_dir.exists():
            pls_csv_path = pls_dir / self.csv_filename
            self.df_metrics.to_csv(pls_csv_path, index=False)
            log.info(f"💾 PLS フォルダにも評価テーブルを保存しました: {pls_csv_path}")

        # 4. グラフ可視化・保存
        if self.save_plot:
            plot_path = self.data_dir / self.plot_filename
            self.plot_metrics(self.df_metrics, self.optimal_k, self.optimal_cov_type, plot_path)
            if pls_dir.exists():
                self.plot_metrics(self.df_metrics, self.optimal_k, self.optimal_cov_type, pls_dir / self.plot_filename)

        if return_tuple:
            return self.optimal_k, self.optimal_cov_type
        return self.optimal_k


def determine_optimal_clusters(
    cfg: dict = None, X: np.ndarray = None, return_tuple: bool = False
) -> int | tuple[int, str] | None:
    if cfg is None:
        cfg = load_config()
    decision_task = ClusterNumberDecision(cfg)
    return decision_task.run(X, return_tuple=return_tuple)
