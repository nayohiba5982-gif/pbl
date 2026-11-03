import os
import shutil
import json
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import sys

# RGB工程の中間生成物キー一覧 (最終成果物は rppg_hr_extracted_dir)
INTERMEDIATE_RGB_KEYS = [
    "rgb_dir",
    "rgb_cleaned_dir",
    "rgb_resampled_dir",
    "rppg_pos_extracted_dir",
    "rppg_filtered_dir",
]

_PARENT_DIR = str(Path(__file__).resolve().parent.parent)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from config_loader import log
from collect import firebase_handler as fh
from collect import data_processor as dp
from collect import utils
from cutting.cutting import cut_data_only
from cleaning.rgb_cleaner import clean_rppg_outliers
from resampling.rgb_resampler import resample_rgb_dataframe
from features.rppg.pos_algorithm import extract_pos_bvp_dataframe
from filtering.bandpass_filter import process_filtering_dataframe
from features.rppg.hr_extraction import process_hr_extraction
from calibration.rppg_calibration import calibrate_rppg_dataframe


class RGBPipeline:
    """
    RGB輝度シグナル / 脈波 (rPPG) 前処理・特徴量抽出パイプライン。

    【RGB値工程】
      1. Storageからファイル取得 (run_download)
      2. Landmark, RGB値別にファイルを作成 (run_cutting)
      3. 欠損値・外れ値除去 (run_rgb_cleaning)
      4. リサンプリング (50Hz) (run_rgb_resampling)
      5. POS Methodから脈波を検出 (run_rgb_pos)
      6. 脈波をbandpassフィルタリング (run_rgb_filtering)
      7. 生体指標算出 (run_rgb_hr_extraction)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.paths = cfg["paths"]
        self.fs = float(self.cfg.get("rppg", {}).get("fs", 50.0))

    # =========================================================================
    # 共通前段工程 (Download & Cutting)
    # =========================================================================

    def run_download(self):
        """Step 1: Firebase Storage からデータ取得 & 初期変換 (gz -> json -> CSV & calib)"""
        log.info("=" * 60)
        log.info("▶ Step 1: Firebase Storage からデータ取得 & 初期変換")
        log.info("=" * 60)
        try:
            bucket = fh.init_firebase()
            gz_files = fh.download_gz_files(bucket)
        except Exception as e:
            log.warning(f"Firebase接続をスキップまたは失敗しました ({e})。ローカル既存ファイルを使用します。")
            gz_files = list(self.paths["data_root"].glob("*.gz"))

        if not gz_files:
            log.info("ダウンロード対象の .gz ファイルがありません。既存のJSON/CSVを確認します。")

        json_files = dp.extract_gz_to_json(gz_files)
        if not json_files:
            json_files = list(self.cfg["firebase"]["output_dir"].glob("*.json"))

        if json_files:
            calibration_rows = []
            for json_path in json_files:
                if json_path.suffix == ".json":
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            record = json.load(f)
                        cal_row = dp.extract_calibration_data(record, json_path.stem)
                        if cal_row:
                            calibration_rows.append(cal_row)
                    except Exception as e:
                        log.error(f"  ✗ キャリブレーション抽出失敗 [{json_path.name}]: {e}")
                    
                    dp.convert_json_to_csv(json_path)
            
            if calibration_rows:
                dp.save_calibration_master(calibration_rows)
            utils.pack_json_to_zip(json_files)

    def run_cutting(self):
        """Step 2: Landmark, RGB値別にファイルを作成 (raw/*.csv -> lm/*.csv, rgb/*.csv)"""
        log.info("=" * 60)
        log.info("▶ Step 2: データ切り出し (Raw CSV -> LM / RGB 分離)")
        log.info("=" * 60)
        cut_data_only(self.cfg)

    # =========================================================================
    # RGB値個別工程 (Steps 3 - 7)
    # =========================================================================


    def process_dataframe(
        self,
        df_rgb: pd.DataFrame,
        df_lm_clean: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        単一被験者の RGB DataFrame に対し、Step 3〜7.5 をメモリ上で順次実行します。
        中間ディスク I/O を介さず高速に処理します。

        Returns:
            pd.DataFrame: 生体指標 (BVP, BPM, bpm_diff 等) が付与された DataFrame
        """
        fps = int(self.cfg.get("landmarks", {}).get("default_fps", 30))
        time_col = self.cfg.get("landmarks", {}).get("timestamp_col", "time")

        pos_cfg = self.cfg.get("rppg", {}).get("pos", {})
        roi_regions = pos_cfg.get("roi_regions", {
            "both": "rppg_bothCheeks",
            "left": "rppg_leftCheek",
            "right": "rppg_rightCheek",
        })
        win_sec = float(pos_cfg.get("window_sec", 1.6))

        filt_cfg = self.cfg.get("rppg", {}).get("filter", {})
        lowcut = float(filt_cfg.get("lowcut", 0.75))
        highcut = float(filt_cfg.get("highcut", 2.5))
        order = int(filt_cfg.get("order", 2))

        hr_cfg = self.cfg.get("rppg", {}).get("hr_extraction", {})
        roi_cols = hr_cfg.get("roi_cols", {
            "both": "bvp_clean_both",
            "left": "bvp_clean_left",
            "right": "bvp_clean_right",
        })

        ma_cfg = self.cfg.get("moving_average", self.cfg.get("rppg", {}).get("moving_average", {}))
        ma_enabled = ma_cfg.get("enabled", True)
        ma_window_sec = float(ma_cfg.get("window_sec", 5.0))
        ma_center = ma_cfg.get("center", False)

        # HRV 設定 (デフォルト True: 45秒スライディング窓 4大指標算出 & タイムスタンプ同期)
        hrv_cfg = self.cfg.get("hrv", self.cfg.get("rppg", {}).get("hrv", {}))
        compute_hrv = hrv_cfg.get("enabled", True)
        hrv_window_sec = float(hrv_cfg.get("window_sec", 45.0))
        hrv_stride_sec = float(hrv_cfg.get("stride_sec", 1.0))

        # Step 3: クレンジング
        df_rgb_clean = clean_rppg_outliers(df_rgb, df_lm=df_lm_clean, fps=fps)

        # Step 4: リサンプリング (50Hz)
        df_resampled = resample_rgb_dataframe(df_rgb_clean, target_fps=self.fs, time_col=time_col)

        # Step 5: POS 脈波抽出
        df_pos = extract_pos_bvp_dataframe(df_resampled, region_prefix=roi_regions, fs=self.fs, window_sec=win_sec)

        # Step 6: 脈波帯域通過フィルタ
        df_filtered = process_filtering_dataframe(
            df_pos,
            target_col="bvp_raw",
            fs=self.fs,
            lowcut=lowcut,
            highcut=highcut,
            order=order,
        )

        # Step 7: HR推定 (瞬時BPM & 移動平均)
        res = process_hr_extraction(
            df_filtered,
            roi_cols=roi_cols,
            fs=self.fs,
            ma_window_sec=ma_window_sec,
            ma_enabled=ma_enabled,
            ma_center=ma_center,
            compute_hrv=compute_hrv,
            hrv_window_sec=hrv_window_sec,
            hrv_stride_sec=hrv_stride_sec,
        )
        if compute_hrv:
            df_hr, selected_roi, snr, _ = res
        else:
            df_hr, selected_roi, snr = res

        # Step 7.5: 10秒移動窓 相対特徴差分キャリブレーション (デフォルト False: 生心拍数をそのまま保持)
        rppg_calib_cfg = self.cfg.get("rppg_calibration", {})
        if rppg_calib_cfg.get("enabled", False):
            df_hr = calibrate_rppg_dataframe(
                df=df_hr,
                target_cols=rppg_calib_cfg.get("target_cols", ["bpm", "bvp_selected"]),
                window_sec=float(rppg_calib_cfg.get("window_sec", 10.0)),
                fs=self.fs,
                method=rppg_calib_cfg.get("baseline_method", "mean"),
                center=bool(rppg_calib_cfg.get("center", False)),
            )

        return df_hr

    def run_rgb_cleaning(self):
        """Step 3 (RGB): 欠損値・外れ値除去 (LM顔未検出と同期)"""
        log.info("=" * 60)
        log.info("▶ Step 3 (RGB): 輝度シグナル外れ値・スパイク除去 (LM同期)")
        log.info("=" * 60)
        in_dir = Path(self.paths["rgb_dir"])
        out_dir = Path(self.paths["rgb_cleaned_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        lm_clean_dir = Path(self.paths["lm_cleaned_dir"])

        rgb_files = sorted(list(in_dir.glob("*.csv")))
        if not rgb_files:
            log.warning(f"処理対象のRGBファイルが見つかりません: {in_dir}")
            return

        for file_path in rgb_files:
            log.info(f"RGB クレンジング中: {file_path.name}")
            df_rgb = pd.read_csv(file_path)

            # 同期用LMクレンジングデータの探索
            lm_name = file_path.name.replace("rgb_", "lm_")
            lm_path = lm_clean_dir / lm_name
            df_lm = pd.read_csv(lm_path) if lm_path.exists() else None

            fps = self.cfg.get("landmarks", {}).get("default_fps", 30)
            df_rgb_clean = clean_rppg_outliers(df_rgb, df_lm=df_lm, fps=int(fps))
            df_rgb_clean.to_csv(out_dir / file_path.name, index=False)

    def run_rgb_resampling(self):
        """Step 4 (RGB): リサンプリング (50Hz PCHIP補間)"""
        log.info("=" * 60)
        log.info(f"▶ Step 4 (RGB): 輝度シグナル リサンプリング ({self.fs}Hz, PCHIP)")
        log.info("=" * 60)
        in_dir = Path(self.paths["rgb_cleaned_dir"])
        out_dir = Path(self.paths["rgb_resampled_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(list(in_dir.glob("*.csv")))
        if not files:
            log.warning(f"処理対象ファイルが見つかりません: {in_dir}")
            return

        time_col = self.cfg.get("landmarks", {}).get("timestamp_col", "time")
        for file_path in files:
            log.info(f"RGB リサンプリング中: {file_path.name}")
            df = pd.read_csv(file_path)
            df_resampled = resample_rgb_dataframe(df, target_fps=self.fs, time_col=time_col)
            df_resampled.to_csv(out_dir / file_path.name, index=False)

    def run_rgb_pos(self):
        """Step 5 (RGB): POS Methodから脈波を検出"""
        log.info("=" * 60)
        log.info("▶ Step 5 (RGB): POS Method による脈波 (BVP) 抽出")
        log.info("=" * 60)
        in_dir = Path(self.paths["rgb_resampled_dir"])
        out_dir = Path(self.paths["rppg_pos_extracted_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(list(in_dir.glob("*.csv")))
        if not files:
            log.warning(f"処理対象ファイルが見つかりません: {in_dir}")
            return

        pos_cfg = self.cfg.get("rppg", {}).get("pos", {})
        roi_regions = pos_cfg.get("roi_regions", {
            "both": "rppg_bothCheeks",
            "left": "rppg_leftCheek",
            "right": "rppg_rightCheek",
        })
        win_sec = float(pos_cfg.get("window_sec", 1.6))

        for file_path in files:
            log.info(f"POS 脈波抽出中: {file_path.name}")
            df = pd.read_csv(file_path)
            df_pos = extract_pos_bvp_dataframe(df, region_prefix=roi_regions, fs=self.fs, window_sec=win_sec)
            df_pos.to_csv(out_dir / file_path.name, index=False)

    def run_rgb_filtering(self):
        """Step 6 (RGB): 脈波をbandpassフィルタリング (0.75 - 2.5Hz Butterworth)"""
        log.info("=" * 60)
        log.info("▶ Step 6 (RGB): 脈波 バンドパスフィルタリング (Butterworth 0.75-2.5Hz)")
        log.info("=" * 60)
        in_dir = Path(self.paths["rppg_pos_extracted_dir"])
        out_dir = Path(self.paths["rppg_filtered_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(list(in_dir.glob("*.csv")))
        if not files:
            log.warning(f"処理対象ファイルが見つかりません: {in_dir}")
            return

        filt_cfg = self.cfg.get("rppg", {}).get("filter", {})
        lowcut = float(filt_cfg.get("lowcut", 0.75))
        highcut = float(filt_cfg.get("highcut", 2.5))
        order = int(filt_cfg.get("order", 2))

        for file_path in files:
            log.info(f"脈波バンドパスフィルタ中: {file_path.name}")
            df = pd.read_csv(file_path)
            df_filtered = process_filtering_dataframe(
                df,
                target_col="bvp_raw",
                fs=self.fs,
                lowcut=lowcut,
                highcut=highcut,
                order=order
            )
            df_filtered.to_csv(out_dir / file_path.name, index=False)

    def run_rgb_hr_extraction(self):
        """Step 7 (RGB): 生体指標算出
        【HR 推定ライン】最適ROI選択、瞬時BPM、5秒移動平均平滑化、PPGピーク、SNR
        【HRV 特徴量専用ライン】サブサンプル補間 (放物線近似)、IBI外れ値補正 (Kubios)、スライディング窓4大指標
        """
        log.info("=" * 60)
        log.info("▶ Step 7 (RGB): 生体指標算出 (HR推定ライン & HRV特徴量専用ライン)")
        log.info("=" * 60)
        in_dir = Path(self.paths["rppg_filtered_dir"])
        out_dir = Path(self.paths["rppg_hr_extracted_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        hrv_dir = Path(self.paths.get("rppg_hrv_dir", "data/rppg_hrv"))

        files = sorted(list(in_dir.glob("*.csv")))
        if not files:
            log.warning(f"処理対象ファイルが見つかりません: {in_dir}")
            return

        hr_cfg = self.cfg.get("rppg", {}).get("hr_extraction", {})
        roi_cols = hr_cfg.get("roi_cols", {
            "both": "bvp_clean_both",
            "left": "bvp_clean_left",
            "right": "bvp_clean_right",
        })

        # 移動平均設定
        ma_cfg = self.cfg.get("moving_average", self.cfg.get("rppg", {}).get("moving_average", {}))
        ma_enabled = ma_cfg.get("enabled", True)
        ma_window_sec = float(ma_cfg.get("window_sec", 5.0))
        ma_center = ma_cfg.get("center", False)

        # HRV 設定
        hrv_cfg = self.cfg.get("hrv", self.cfg.get("rppg", {}).get("hrv", {}))
        compute_hrv = hrv_cfg.get("enabled", True)
        hrv_window_sec = float(hrv_cfg.get("window_sec", 45.0))
        hrv_stride_sec = float(hrv_cfg.get("stride_sec", 1.0))
        hrv_save_csv = hrv_cfg.get("save_csv", False)
        if compute_hrv and hrv_save_csv:
            hrv_dir.mkdir(parents=True, exist_ok=True)

        for file_path in files:
            log.info(f"生体指標算出中: {file_path.name}")
            df = pd.read_csv(file_path)
            res = process_hr_extraction(
                df,
                roi_cols=roi_cols,
                fs=self.fs,
                ma_window_sec=ma_window_sec,
                ma_enabled=ma_enabled,
                ma_center=ma_center,
                compute_hrv=compute_hrv,
                hrv_window_sec=hrv_window_sec,
                hrv_stride_sec=hrv_stride_sec,
            )
            if compute_hrv:
                df_hr, selected_roi, snr, df_hrv = res
            else:
                df_hr, selected_roi, snr = res
                df_hrv = None

            # 生体指標 10秒ウィンドウ内完結 差分キャリブレーション適用 (デフォルト False: 生心拍数をそのまま保持)
            rppg_calib_cfg = self.cfg.get("rppg_calibration", {})
            if rppg_calib_cfg.get("enabled", False):
                df_hr = calibrate_rppg_dataframe(
                    df=df_hr,
                    target_cols=rppg_calib_cfg.get("target_cols", ["bpm", "bvp_selected"]),
                    window_sec=float(rppg_calib_cfg.get("window_sec", 10.0)),
                    fs=self.fs,
                    method=rppg_calib_cfg.get("baseline_method", "mean"),
                    center=bool(rppg_calib_cfg.get("center", False)),
                )

            log.info(f"  └─ 採択領域: {selected_roi}, SNR: {snr:.2f} dB, 平均心拍数: {df_hr['bpm'].mean():.1f} BPM")
            df_hr.to_csv(out_dir / file_path.name, index=False)

            if df_hrv is not None and not df_hrv.empty and hrv_save_csv:
                stem = file_path.stem.replace("rgb_", "")
                hrv_path = hrv_dir / f"hrv_{stem}.csv"
                df_hrv.to_csv(hrv_path, index=False)
                log.info(f"  └─ HRV特徴量テーブル保存完了: {hrv_path.name} ({len(df_hrv)} 窓, 平均RMSSD: {df_hrv['rmssd'].mean():.2f}ms, 平均SDNN: {df_hrv['sdnn'].mean():.2f}ms)")

    # エイリアス (簡潔な呼び出し用)
    run_cleaning = run_rgb_cleaning
    run_resampling = run_rgb_resampling
    run_pos = run_rgb_pos
    run_filtering = run_rgb_filtering
    run_hr_extraction = run_rgb_hr_extraction

    def cleanup_intermediates(self, include_cutting: bool = False):
        """
        RGB工程の中間生成物ディレクトリをクリーンアップします。
        最終成果物 (rppg_hr_extracted_dir) は保持されます。
        """
        log.info("🧹 RGB/rPPG中間生成物のクリーンアップを開始します...")
        keys = list(INTERMEDIATE_RGB_KEYS)
        if not include_cutting and "rgb_dir" in keys:
            keys.remove("rgb_dir")

        deleted_count = 0
        for key in keys:
            dir_path = Path(self.paths.get(key, ""))
            if dir_path.exists() and dir_path.is_dir():
                try:
                    shutil.rmtree(dir_path)
                    log.info(f"  🗑️ 中間ディレクトリを削除しました: {dir_path.name}")
                    deleted_count += 1
                except Exception as e:
                    log.warning(f"  ⚠️ ディレクトリ削除失敗 ({dir_path}): {e}")
        log.info(f"✨ RGB/rPPG中間クリーンアップ完了 ({deleted_count} ディレクトリ削除)")
        log.info(f"📁 最終成果物を保持: {self.paths.get('rppg_hr_extracted_dir')}")

    def run_pipeline(self, keep_intermediates: bool = None):
        """RGB値の全7工程 (Step 3〜7) を順次実行"""
        if keep_intermediates is None:
            keep_intermediates = not self.cfg.get("cleanup_intermediates", True)

        log.info("\n" + "#" * 60)
        log.info("  🔴 RGB値 パイプライン全工程 (Step 3〜7) 開始")
        log.info("#" * 60)
        try:
            self.run_rgb_cleaning()       # Step 3
            self.run_rgb_resampling()     # Step 4
            self.run_rgb_pos()            # Step 5
            self.run_rgb_filtering()      # Step 6
            self.run_rgb_hr_extraction() # Step 7
        except Exception as e:
            log.error(f"❌ RGB値工程中にエラーが発生しました: {e}")
            log.warning("⚠️ 処理が中断されたため、成功した工程の中間ファイルはそのまま保持されます。")
            raise e

        if not keep_intermediates:
            self.cleanup_intermediates(include_cutting=False)
        log.info("✓ RGB値全工程が完了しました\n")

    def run_all(self, keep_intermediates: bool = None):
        """ダウンロード・切り出しを含めたRGB値全工程を実行"""
        if keep_intermediates is None:
            keep_intermediates = not self.cfg.get("cleanup_intermediates", True)

        log.info("🚀 RGB値 パイプライン (ダウンロード〜全7工程) 開始")
        try:
            self.run_download()
            self.run_cutting()
            self.run_pipeline(keep_intermediates=True)
        except Exception as e:
            log.error(f"❌ RGB値 パイプライン中にエラーが発生しました: {e}")
            log.warning("⚠️ 処理が中断されたため、成功した工程の中間ファイルはそのまま保持されます。")
            raise e

        if not keep_intermediates:
            self.cleanup_intermediates(include_cutting=True)
        log.info("🎉 RGB値処理が正常に完了しました！")


if __name__ == "__main__":
    import argparse
    from config_loader import load_config

    parser = argparse.ArgumentParser(description="RGB輝度シグナル / 脈波 (rPPG) 前処理・生体指標抽出パイプライン")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定YAMLファイル名")
    parser.add_argument(
        "--stage",
        nargs="+",
        default=["pipeline"],
        help="実行ステージ (all, pipeline, download, cutting, clean, resample, pos, filter, hr)",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        default=False,
        help="中間ファイルを削除せずに保持する (デフォルト: False, 成功時は最終ファイルのみ残す)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipeline = RGBPipeline(cfg)

    default_cleanup = cfg.get("cleanup_intermediates", True)
    keep = args.keep_intermediates or (not default_cleanup)

    stages = set(args.stage)
    if "all" in stages:
        pipeline.run_all(keep_intermediates=keep)
    elif "pipeline" in stages:
        pipeline.run_pipeline(keep_intermediates=keep)
    else:
        try:
            if "download" in stages:
                pipeline.run_download()
            if "cutting" in stages:
                pipeline.run_cutting()
            if "clean" in stages or "rgb_clean" in stages:
                pipeline.run_rgb_cleaning()
            if "resample" in stages or "rgb_resample" in stages:
                pipeline.run_rgb_resampling()
            if "pos" in stages or "rgb_pos" in stages:
                pipeline.run_rgb_pos()
            if "filter" in stages or "rgb_filter" in stages:
                pipeline.run_rgb_filtering()
            if "hr" in stages or "rgb_hr" in stages or "hrv" in stages or "rgb_hrv" in stages:
                pipeline.run_rgb_hr_extraction()
        except Exception as e:
            log.error(f"❌ RGB個別ステージ実行中にエラーが発生しました: {e}")
            log.warning("⚠️ 途中の工程で失敗したため、成功した部分のファイルは保持されます。")
            raise e
