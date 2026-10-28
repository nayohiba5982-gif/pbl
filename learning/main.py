"""
集中度推定 AI モデル学習・表現学習・クラスタリング・下流タスク統合メインエントリポイント
(learning/main.py)

各サブディレクトリのパイプライン (ClusteringPipeline, HybridFeaturePipeline, DownstreamTaskPipeline)
および PreviousTask (TS2Vec) を呼び出して一括またはステージ単位で実行します。

※設計上の役割分担:
  Step 2〜6（GMM・ソフトラベル・特徴量融合）はデータセット全体の統計的性質を把握するための
  探索的データ分析 (EDA) / プロファイリング であり、モデルの真の汎化性能評価は
  Step 7（各Fold内で独立処理を行う完全リークフリーLOSO-CV）によって担保されています。
"""

import argparse
import copy
from pathlib import Path
import sys
import time

_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

from config_loader import load_config, log
from previous_task import PreviousTask, normalize_window_list
from clustering.pipeline import ClusteringPipeline
from downstream_task.pipeline import DownstreamTaskPipeline


def main():
    parser = argparse.ArgumentParser(
        description="集中度推定 AI モデル学習・表現学習・クラスタリング統合パイプライン"
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
            "previous_task",
            "pretext",               # previous_task のエイリアス
            "decision_class_number", # GMM 最適クラスタ数探索
            "decision_k",            # decision_class_number のエイリアス
            "clustering",            # GMM クラスタリング & ラベル付与
            "cluster_analysis",      # GMM クラスタの特徴量集計・分析
            "analysis",              # cluster_analysis のエイリアス
            "cal_probability",       # サンプル別確率ラベル（ソフトターゲット y）の算出
            "cal_probality",         # cal_probability のエイリアス
            "soft_label",            # cal_probability のエイリアス
            "hybrid_feature",        # 物理統計量集約 & TS2Vec 特徴量結合 (Feature Fusion)
            "fusion",                # hybrid_feature のエイリアス
            "hybrid",                # hybrid_feature のエイリアス
            "base_model",            # Visual / Vital 各モダリティのベースモデル作成・評価
            "base",                  # base_model のエイリアス
            "meta_model",            # 選択ベースモデルを用いたメタモデル作成・最終推論
            "meta",                  # meta_model のエイリアス
            "downstream_task",       # 下流タスク一括 (base_model + meta_model)
            "downstream",            # downstream_task のエイリアス
            "tree_models",           # downstream_task のエイリアス
            "loso_cv",               # LOSO-CV 完全データリークフリー回帰学習
            "loso_regression",       # loso_cv のエイリアス
            "loso",                  # loso_cv のエイリアス
        ],
        default=["all"],
        help="実行するパイプラインの工程を指定 (例: --stage all / --stage base_model / --stage meta_model)"
    )

    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="各クラスタに対する重み w_k (例: --weights 1.0 0.5 0.0)"
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
        "--covariance-type",
        "--covariance",
        "--cov",
        type=str,
        default=None,
        choices=["full", "tied", "diag", "spherical"],
        help="GMM 共分散タイプの指定 (full, tied, diag, spherical)"
    )

    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="クラスタ数 (k) の直接指定 (指定時は自動探索をスキップして指定した k でクラスタリング)"
    )

    parser.add_argument(
        "--mapping",
        type=str,
        default=None,
        help="クラスタと目的変数 y の手動マッピング (例: '0:1,1:0' -> Cluster 0=集中(1), Cluster 1=リラックス(0))"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=[
            "random_forest", "rf",
            "xgboost", "xgb",
            "lightgbm", "lgbm", "lgb",
            "catboost", "cb",
            "adaboost", "ada",
            "logistic_regression", "lr",
            "ridge",
            "lasso",
            "elastic_net", "elasticnet", "en",
        ],
        help="学習対象のベースモデルまたはメタモデル (未指定時は config.yaml の設定を参照)"
    )

    parser.add_argument(
        "--visual-model",
        type=str,
        default=None,
        help="Meta Model 学習時に採用する Visual ベースモデル指定"
    )

    parser.add_argument(
        "--vital-model",
        type=str,
        default=None,
        help="Meta Model 学習時に採用する Vital ベースモデル指定"
    )

    parser.add_argument(
        "--test-mock",
        action="store_true",
        help="合成モックデータを用いて木モデル学習・評価動作検証を実行"
    )

    args = parser.parse_args()

    # 1. 対象ウィンドウサイズの正規化 (単一選択、複数選択、または all)
    base_cfg = load_config(args.config)
    default_ws = base_cfg.get("dataset", {}).get("window_size", "10s")
    prev_selected_ws = base_cfg.get("previous_task", {}).get(
        "selected_window", base_cfg.get("previous_task", {}).get("selected_windows", default_ws)
    )

    if args.window_size:
        target_windows = normalize_window_list(args.window_size, fallback=default_ws)
    else:
        target_windows = normalize_window_list(prev_selected_ws, fallback=default_ws)

    log.info("=" * 70)
    log.info(f" 🚀 パイプライン実行開始 [対象ステージ: {', '.join(args.stage)}]")
    log.info(f" ⚙️ 設定ファイル: {args.config}")
    log.info(f" ⏱️ 実行対象ウィンドウサイズ: {target_windows}")
    if args.n_clusters:
        log.info(f" 🎯 クラスタ数 (直接指定): k = {args.n_clusters}")
    log.info("=" * 70)

    total_start_time = time.time()
    stages = set(args.stage)

    # 2. 各ウィンドウサイズに対する順次実行
    for idx, ws in enumerate(target_windows, 1):
        if len(target_windows) > 1:
            log.info("=" * 70)
            log.info(f" 🔄 [{idx}/{len(target_windows)}] ウィンドウサイズ: {ws} の処理を開始します")
            log.info("=" * 70)

        # 設定のディープコピーとウィンドウサイズ反映
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("dataset", {})["window_size"] = ws
        cfg.setdefault("previous_task", {})["selected_window"] = [ws]

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
            log.info(f" 🏷️ 手動クラスタマッピング指定: {mapping_dict}")

        # 各サブパッケージパイプラインのインスタンス化
        prev_task = PreviousTask(cfg, window_size=ws)
        clustering_pipe = ClusteringPipeline(cfg)
        downstream_pipe = DownstreamTaskPipeline(cfg)

        # ステージ実行
        if "all" in stages:
            # Step 1: Pretext Task (TS2Vec)
            success = prev_task.run()
            if not success:
                log.warning("⚠️ 前段タスクが完了しなかったか、データが存在しませんでした。")
                continue

            # Step 2-5: GMM クラスタリングパイプライン (最適クラスタ数・クラスタリング・分析・確率ラベル)
            clustered = clustering_pipe.run_all(
                n_clusters=args.n_clusters,
                weights=args.weights,
                covariance_type=args.covariance_type,
            )
            if not clustered:
                log.warning("⚠️ クラスタリングが完了しなかったため、以降の工程をスキップします。")
                continue

            # Step 7: 下流タスク (Base Model [Visual/Vital] ➔ Meta Model)
            downstream_pipe.run_all(
                stage="all",
                base_models=args.models,
                visual_model=args.visual_model,
                vital_model=args.vital_model,
                weights=args.weights,
                covariance_type=args.covariance_type,
            )
        else:
            if "previous_task" in stages or "pretext" in stages:
                prev_task.run()

            if "decision_class_number" in stages or "decision_k" in stages:
                clustering_pipe.run_decision_class_number()

            if "clustering" in stages:
                clustering_pipe.run_clustering(
                    n_clusters=args.n_clusters,
                    weights=args.weights,
                    covariance_type=args.covariance_type,
                )

            if "cluster_analysis" in stages or "analysis" in stages:
                clustering_pipe.run_cluster_analysis()

            if "cal_probability" in stages or "cal_probality" in stages or "soft_label" in stages:
                clustering_pipe.run_cal_probability(weights=args.weights)

            if "hybrid_feature" in stages or "fusion" in stages or "hybrid" in stages:
                log.info("ℹ️ hybrid_feature は Step 7 (downstream_task) の内部処理に統合されました。")

            if any(s in stages for s in [
                "downstream_task", "downstream", "tree_models",
                "loso_cv", "loso_regression", "loso"
            ]):
                downstream_pipe.run_downstream_task(
                    stage="all",
                    base_models=args.models,
                    visual_model=args.visual_model,
                    vital_model=args.vital_model,
                    weights=args.weights,
                    covariance_type=args.covariance_type,
                )
            else:
                if "base_model" in stages or "base" in stages:
                    downstream_pipe.run_base_models(
                        models=args.models,
                        weights=args.weights,
                        covariance_type=args.covariance_type,
                    )
                if "meta_model" in stages or "meta" in stages:
                    downstream_pipe.run_meta_models(
                        models=args.models,
                        visual_model=args.visual_model,
                        vital_model=args.vital_model,
                    )

    total_elapsed = time.time() - total_start_time
    log.info("=" * 70)
    log.info(f" 🎉 全ウィンドウサイズ ({target_windows}) の処理が完了しました (総所要時間: {total_elapsed:.2f} 秒)")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
