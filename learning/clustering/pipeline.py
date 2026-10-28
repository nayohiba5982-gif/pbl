"""
GMM クラスタリング・最適クラスタ数探索・特徴量集計・確率ラベル付与統合パイプライン
(learning/clustering/pipeline.py)

【サポートステージ】
- decision_class_number (decision_k): 最適クラスタ数探索 (BIC/AIC/Silhouette)
- clustering: GMM クラスタリング (Trainのみfit) & 推論
- cluster_analysis (analysis): クラスタ別特徴量集計 & タスククロス集計
- cal_probability (soft_label): サンプルごとの期待値ソフトターゲット y 算出
- all: 上記全工程を順次一括実行
"""

import argparse
import copy
from pathlib import Path
import sys
import time

_CURRENT_DIR = Path(__file__).resolve().parent
_LEARNING_DIR = _CURRENT_DIR.parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))
if str(_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(_LEARNING_DIR))

try:
    from config_loader import load_config, log
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("clustering.pipeline")

    def load_config(config_filename: str = "config.yaml") -> dict:
        return {}

try:
    from clustering.decision import ClusterNumberDecision, determine_optimal_clusters
    from clustering.gmm import GMMClustering, run_clustering
    from clustering.analysis import ClusterAnalyzer, run_cluster_analysis
    from clustering.probability import ProbabilityLabelCalculator, run_cal_probability
except ImportError:
    from decision import ClusterNumberDecision, determine_optimal_clusters
    from gmm import GMMClustering, run_clustering
    from analysis import ClusterAnalyzer, run_cluster_analysis
    from probability import ProbabilityLabelCalculator, run_cal_probability


class ClusteringPipeline:
    """
    GMM クラスタリング関連の全工程を統括するパイプラインクラス
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.decision_task = ClusterNumberDecision(cfg)
        self.clustering_task = GMMClustering(cfg)
        self.analysis_task = ClusterAnalyzer(cfg)
        self.prob_calculator = ProbabilityLabelCalculator(cfg)

    def run_decision_class_number(self):
        """Step 2: GMM による最適クラスタ数 K の自動決定"""
        return self.decision_task.run()

    def run_clustering(
        self,
        n_clusters: int | None = None,
        weights: list[float] | None = None,
        covariance_type: str | None = None,
    ):
        """Step 3: GMM クラスタリングの実行 & ラベル保存"""
        return self.clustering_task.run(n_clusters=n_clusters, weights=weights, covariance_type=covariance_type)

    def run_cluster_analysis(self):
        """Step 4: GMM クラスタの生体特徴量集計・タスククロス集計 (Trainのみ)"""
        return self.analysis_task.run()

    def run_cal_probability(self, weights: list[float] | None = None):
        """Step 5: GMM 所属確率とクラスタ重みから期待値ソフトラベル (y ∈ [0, 1]) を算出"""
        return self.prob_calculator.run(weights=weights)

    def run_all(
        self,
        n_clusters: int | None = None,
        weights: list[float] | None = None,
        covariance_type: str | None = None,
    ) -> bool:
        """クラスタリング関連の全工程を順次実行"""
        log.info("🚀 [Clustering Pipeline] 全工程の一括実行を開始します")

        clustered = self.run_clustering(
            n_clusters=n_clusters,
            weights=weights,
            covariance_type=covariance_type,
        )
        if not clustered:
            log.warning("⚠️ クラスタリングが完了しなかったため、以降の工程をスキップします。")
            return False

        self.run_cluster_analysis()
        self.run_cal_probability(weights=weights)

        log.info("🎉 [Clustering Pipeline] 全工程が正常に完了しました")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="GMM クラスタリング・分析・確率ラベル付与統合パイプライン"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="設定YAMLファイル名 (デフォルト: config.yaml)"
    )
    parser.add_argument(
        "--stage",
        nargs="+",
        choices=[
            "all",
            "decision_class_number", "decision_k",
            "clustering",
            "cluster_analysis", "analysis",
            "cal_probability", "cal_probality", "soft_label",
        ],
        default=["all"],
        help="実行ステージの指定 (例: --stage all / --stage clustering)"
    )
    parser.add_argument(
        "--window-size",
        "--window-sizes",
        "-w",
        nargs="+",
        default=None,
        help="ウィンドウサイズの指定 (例: 10s / 5s 10s 15s / all: 全て一括実行)"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="クラスタ数 (k) の直接指定"
    )
    parser.add_argument(
        "--covariance-type",
        "--covariance",
        "--cov",
        type=str,
        default=None,
        choices=["full", "tied", "diag", "spherical"],
        help="GMM 共分散タイプの指定"
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="各クラスタに対する重み w_k (例: --weights 1.0 0.5 0.0)"
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default=None,
        help="クラスタとラベルの手動マッピング指定 (例: '0:1,1:0')"
    )

    args = parser.parse_args()

    ALL_SUPPORTED_WINDOWS = ["5s", "10s", "15s"]
    base_cfg = load_config(args.config)

    if args.window_size:
        raw_list = [w.strip() for w in args.window_size]
        if "all" in [w.lower() for w in raw_list]:
            target_windows = ALL_SUPPORTED_WINDOWS
        else:
            target_windows = []
            for w in raw_list:
                w_lower = w.lower()
                if w_lower in ["5", "5s"]:
                    target_windows.append("5s")
                elif w_lower in ["10", "10s"]:
                    target_windows.append("10s")
                elif w_lower in ["15", "15s"]:
                    target_windows.append("15s")
                elif w_lower == "all":
                    target_windows.extend(ALL_SUPPORTED_WINDOWS)
                else:
                    target_windows.append(w)
            target_windows = list(dict.fromkeys(target_windows))
    else:
        default_ws = base_cfg.get("dataset", {}).get("window_size", "10s")
        target_windows = [default_ws]

    log.info("=" * 70)
    log.info(f" 🚀 [Clustering Pipeline] 開始 [対象ステージ: {', '.join(args.stage)}]")
    log.info(f" ⏱️ 対象ウィンドウサイズ: {target_windows}")
    log.info("=" * 70)

    total_start_time = time.time()
    stages = set(args.stage)

    for idx, ws in enumerate(target_windows, 1):
        if len(target_windows) > 1:
            log.info("=" * 70)
            log.info(f" 🔄 [{idx}/{len(target_windows)}] ウィンドウサイズ: {ws} の処理を開始します")
            log.info("=" * 70)

        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("dataset", {})["window_size"] = ws

        if args.n_clusters:
            cfg.setdefault("clustering", {})["n_clusters"] = args.n_clusters
        if args.mapping:
            mapping_dict = {}
            for pair in args.mapping.split(","):
                if ":" in pair:
                    k_str, v_str = pair.split(":", 1)
                    mapping_dict[int(k_str.strip())] = int(v_str.strip())
            cfg.setdefault("cluster_analysis", {})["method"] = "manual"
            cfg["cluster_analysis"]["manual_mapping"] = mapping_dict

        pipeline = ClusteringPipeline(cfg)

        if "all" in stages:
            pipeline.run_all(
                n_clusters=args.n_clusters,
                weights=args.weights,
                covariance_type=args.covariance_type,
            )
        else:
            if "decision_class_number" in stages or "decision_k" in stages:
                pipeline.run_decision_class_number()

            if "clustering" in stages:
                pipeline.run_clustering(
                    n_clusters=args.n_clusters,
                    weights=args.weights,
                    covariance_type=args.covariance_type,
                )

            if "cluster_analysis" in stages or "analysis" in stages:
                pipeline.run_cluster_analysis()

            if "cal_probability" in stages or "cal_probality" in stages or "soft_label" in stages:
                pipeline.run_cal_probability(weights=args.weights)

    total_elapsed = time.time() - total_start_time
    log.info("=" * 70)
    log.info(f" 🎉 [Clustering Pipeline] 全処理が完了しました (所要時間: {total_elapsed:.2f} 秒)")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
