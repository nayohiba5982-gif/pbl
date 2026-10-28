import argparse
import time
from pathlib import Path
from config_loader import load_config, log
from pipeline import DataPipeline

def main():
    parser = argparse.ArgumentParser(
        description="生体シグナル（顔ランドマーク & 脈波/rPPG）前処理・特徴量抽出統合パイプライン"
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
            # 一括実行
            "all",
            "download",
            "cutting",
            "lm",
            "rgb",
            "merge",
            "windowing",
            "streamlined",  # 中間ファイルゼロ・被験者別インメモリ一括実行
            # ランドマーク個別工程
            "lm_clean",
            "lm_resample",
            "lm_filter",
            "lm_detrend",
            "lm_feat",
            "lm_calibration",
            "lm_calib",
            "lm_norm",
            # RGB個別工程
            "rgb_clean",
            "rgb_resample",
            "rgb_pos",
            "rgb_filter",
            "rgb_hr",
            "rgb_hrv",
            "hrv",
            "rppg_calib",
            "rgb_calib",
            # 共通エイリアス (後方互換)
            "cleaning",
            "resampling",
            "features",
        ],
        default=["all"],
        help="実行するパイプラインの工程を指定 (複数指定可: --stage lm rgb merge / --stage lm_clean lm_resample 等)"
    )

    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        default=False,
        help="中間ファイルを削除せずに保持する (デフォルト: False, 成功時は最終ファイルのみ残す)"
    )

    args = parser.parse_args()
    
    # 1. 設定のロード
    cfg = load_config(args.config)
    
    # 2. パイプラインクラスのインスタンス化
    pipeline = DataPipeline(cfg)

    default_cleanup = cfg.get("cleanup_intermediates", True)
    keep_intermediates = args.keep_intermediates or (not default_cleanup)

    log.info("=" * 70)
    log.info(f" 🚀 パイプライン実行開始 [対象ステージ: {', '.join(args.stage)}]")
    log.info(f" ⚙️ 設定ファイル: {args.config}")
    log.info(f" 🧹 中間ファイル削除: {'無効 (保持)' if keep_intermediates else '有効 (成功時に削除)'}")
    log.info("=" * 70)

    start_time = time.time()

    # 3. 指定されたステージの順次実行
    stages = set(args.stage)

    try:
        if "all" in stages:
            pipeline.run_all(keep_intermediates=keep_intermediates)
        elif "streamlined" in stages:
            # 中間ファイルゼロ・被験者別インメモリ一括実行 (ダウンロード以降の全工程)
            pipeline.run_streamlined_pipeline()
        else:
            # Step 1: ダウンロード
            if "download" in stages:
                pipeline.run_download()

            # Step 2: 切り出し
            if "cutting" in stages:
                pipeline.run_cutting()

            # --- ランドマーク工程 ---
            if "lm" in stages:
                need_lm_intermediates = keep_intermediates or ("rgb" in stages) or ("merge" in stages) or ("windowing" in stages)
                pipeline.run_lm_pipeline(keep_intermediates=need_lm_intermediates)
            else:
                if "lm_clean" in stages or "cleaning" in stages:
                    pipeline.run_lm_cleaning()
                if "lm_resample" in stages or "resampling" in stages:
                    pipeline.run_lm_resampling()
                if "lm_filter" in stages:
                    pipeline.run_lm_filtering()
                if "lm_detrend" in stages:
                    pipeline.run_lm_detrending()
                if "lm_feat" in stages:
                    pipeline.run_lm_features()
                if "lm_calibration" in stages or "lm_calib" in stages or "lm_norm" in stages or "features" in stages:
                    pipeline.run_lm_calibration()

            # --- RGB工程 ---
            if "rgb" in stages:
                need_rgb_intermediates = keep_intermediates or ("merge" in stages) or ("windowing" in stages)
                pipeline.run_rgb_pipeline(keep_intermediates=need_rgb_intermediates)
            else:
                if "rgb_clean" in stages or "cleaning" in stages:
                    pipeline.run_rgb_cleaning()
                if "rgb_resample" in stages or "resampling" in stages:
                    pipeline.run_rgb_resampling()
                if "rgb_pos" in stages:
                    pipeline.run_rgb_pos()
                if "rgb_filter" in stages:
                    pipeline.run_rgb_filtering()
                if "rgb_hr" in stages or "features" in stages or "rgb_hrv" in stages or "hrv" in stages or "rppg_calib" in stages or "rgb_calib" in stages:
                    pipeline.run_rgb_hr_extraction()

            # --- 統合 ---
            if "merge" in stages:
                pipeline.run_merge_features()

            # --- スライディングウィンドウ作成 ---
            if "windowing" in stages:
                pipeline.run_windowing()

            # 指定ステージ群がすべて正常完了した場合のクリーンアップ
            if not keep_intermediates:
                pipeline.cleanup_for_stages(stages)

    except Exception as e:
        log.error("=" * 70)
        log.error(f" ❌ パイプライン実行中にエラーが発生しました: {e}")
        log.warning(" ⚠️ 途中の工程で失敗したため、成功した部分のファイルは保持されます。")
        log.error("=" * 70)
        raise e

    elapsed = time.time() - start_time
    log.info("=" * 70)
    log.info(f" 🎉 全ての指定された処理が正常に完了しました！ (所要時間: {elapsed:.2f} 秒)")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
