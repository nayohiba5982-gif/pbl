import os
import shutil
import json
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import sys

# ランドマーク工程の中間生成物キー一覧 (最終成果物は lm_calibrated_dir / lm_normalized_dir)
INTERMEDIATE_LM_KEYS = [
    "lm_dir",
    "lm_cleaned_dir",
    "lm_resampled_dir",
    "lm_filtered_dir",
    "lm_detrended_dir",
    "lm_features_dir",
]

_PARENT_DIR = str(Path(__file__).resolve().parent.parent)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from config_loader import log
from collect import firebase_handler as fh
from collect import data_processor as dp
from collect import utils
from cutting.cutting import cut_data_only
from cleaning.lm_cleaner import clean_landmark_outliers
from resampling.lm_resampler import resample_lm_dataframe
from filtering.lowpass_filter import apply_lowpass_to_dataframe
from detrending.detrend import detrend_landmark_dataframe
from features.lm.lm_extractor import extract_lm_features_dataframe
from calibration.lm_calibration import load_calibration_master, calibrate_landmarks
from normalization.normalizer import (
    LandmarkFeaturePipeline,
)


class LandmarkPipeline:
    """
    顔ランドマーク前処理・特徴量抽出パイプライン。
    
    【ランドマーク工程】
      1. Storageからファイル取得 (run_download)
      2. Landmark, RGB値別にファイルを作成 (run_cutting)
      3. 欠損値・外れ値除去 (run_lm_cleaning)
      4. リサンプリング (50Hz) (run_lm_resampling)
      5. フィルタリング (ローパスフィルタ) (run_lm_filtering)
      6. トレンド除去 (run_lm_detrending)
      7. 特徴量計算 (EAR, Head Pose, Eye Rotation等) (run_lm_features)
      8. キャリブレーションを用いた正規化 (run_lm_calibration)
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
        log.info("▶ Step 1: Firebase Storage データ取得 & CSV変換")
        log.info("=" * 60)
        raw_dir = Path(self.paths["raw_dir"])
        raw_dir.mkdir(parents=True, exist_ok=True)

        fb_cfg = self.cfg.get("firebase", {})
        cred_path = Path(_PARENT_DIR) / fb_cfg.get("credential_path", "")
        bucket_name = fb_cfg.get("bucket_name", "")
        prefix = fb_cfg.get("storage_prefix", "measurements/")

        if not cred_path.exists():
            log.warning(f"Firebase認証情報が見つかりません: {cred_path}")
            log.info("ローカル既存ファイルを使用して処理を継続します。")
            return

        db_handler = fh.FirebaseHandler(credential_path=str(cred_path), bucket_name=bucket_name)
        processor = dp.DataProcessor(output_dir=str(raw_dir))

        user_dirs = db_handler.list_subdirectories(prefix=prefix)
        log.info(f"Storageから {len(user_dirs)} 件の被験者ディレクトリを検出しました")

        for u_dir in user_dirs:
            user_id = u_dir.rstrip("/").split("/")[-1]
            local_user_dir = raw_dir / user_id
            local_user_dir.mkdir(parents=True, exist_ok=True)

            zip_blob_name = f"{u_dir.rstrip('/')}.zip"
            local_zip_path = raw_dir / f"{user_id}.zip"

            if db_handler.blob_exists(zip_blob_name):
                log.info(f"  ⬇️ {user_id}: zip一括ダウンロード開始")
                db_handler.download_file(zip_blob_name, str(local_zip_path))
                utils.unzip_file(str(local_zip_path), str(local_user_dir))
                if local_zip_path.exists():
                    local_zip_path.unlink()
            else:
                log.info(f"  ⬇️ {user_id}: ディレクトリ内ファイル順次ダウンロード開始")
                db_handler.download_directory(u_dir, str(local_user_dir))

            gz_files = list(local_user_dir.glob("*.gz"))
            if gz_files:
                for gz_f in gz_files:
                    utils.decompress_gz(str(gz_f))
                    gz_f.unlink()

            json_files = list(local_user_dir.glob("*.json"))
            calib_files = [f for f in json_files if "calibration" in f.name.lower()]
            measure_files = [f for f in json_files if "calibration" not in f.name.lower()]

            if measure_files:
                processor.process_measurements(measure_files, output_filename=f"{user_id}.csv")
            if calib_files:
                processor.process_calibration(calib_files, subject_id=user_id)

            if local_user_dir.exists():
                shutil.rmtree(local_user_dir)

        log.info("✓ Step 1: ダウンロード・CSV変換が完了しました")

    def run_cutting(self):
        """Step 2: Landmark, RGB値別にファイルを作成 (raw/*.csv -> lm/*.csv, rgb/*.csv)"""
        log.info("=" * 60)
        log.info("▶ Step 2: データ切り出し (LM & RGB 分離)")
        log.info("=" * 60)
        raw_dir = Path(self.paths["raw_dir"])
        lm_dir = Path(self.paths["lm_dir"])
        rgb_dir = Path(self.paths["rgb_dir"])

        lm_dir.mkdir(parents=True, exist_ok=True)
        rgb_dir.mkdir(parents=True, exist_ok=True)

        raw_candidates = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("**/*.csv"))
        seen = set()
        raw_files = []
        for p in raw_candidates:
            if p.is_file() and "calibration_master" not in p.name and p.resolve() not in seen:
                seen.add(p.resolve())
                raw_files.append(p)
        raw_files = sorted(raw_files)

        if not raw_files:
            log.warning(f"切り出し対象の生データが見つかりません: {raw_dir}")
            return

        for f in raw_files:
            cut_data_only(f, lm_dir, rgb_dir)
        log.info("✓ Step 2: 切り出しが完了しました")

    # =========================================================================
    # インメモリ処理 (Streamlined Pipeline 用)
    # =========================================================================

    def process_single_user_in_memory(
        self,
        df_lm: pd.DataFrame,
        subject_id: str,
        master_df: pd.DataFrame | None = None,
        r_eye: float = 12.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        1被験者のランドマークDataFrameに対して、Step 3〜8 をメモリ上で順次実行します。
        
        Returns:
            df_final (pd.DataFrame): キャリブレーション正規化完了データ
            df_clean (pd.DataFrame): RGB工程の同期用クレンジング済みデータ
        """
        # Step 3: 外れ値除去
        df_clean = clean_landmark_outliers(df_lm)

        # Step 4: リサンプリング (50Hz)
        df_resampled = resample_lm_dataframe(df_clean, target_fs=self.fs)

        # Step 5: ローパスフィルタ
        df_filtered = apply_lowpass_to_dataframe(df_resampled, fs=self.fs, default_cutoff=3.0)

        # Step 6: トレンド除去
        df_detrended = detrend_landmark_dataframe(df_filtered, fs=self.fs)

        # Step 7: 幾何特徴量抽出
        df_features = extract_lm_features_dataframe(df_detrended, r_eye=r_eye)

        # Step 8: 個人内キャリブレーション正規化 (最終成果物)
        calib_row = {}
        if master_df is not None and subject_id in master_df.index:
            calib_row = master_df.loc[subject_id]
        df_calib = calibrate_landmarks(df_features, calib_row, self.cfg)

        return df_calib, df_clean

    def process_dataframe(
        self,
        df_lm: pd.DataFrame,
        subject_id: str,
        master_df: pd.DataFrame | None = None,
        r_eye: float = 12.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """process_single_user_in_memory のエイリアス"""
        return self.process_single_user_in_memory(
            df_lm=df_lm,
            subject_id=subject_id,
            master_df=master_df,
            r_eye=r_eye,
        )

    # =========================================================================
    # ディスクI/O個別工程 (Step 3〜8)
    # =========================================================================

    def run_lm_cleaning(self):
        """Step 3 (LM): 欠損値・外れ値除去"""
        log.info("=" * 60)
        log.info("▶ Step 3 (LM): ランドマーク欠損値・外れ値除去")
        log.info("=" * 60)
        out_dir = Path(self.paths["lm_cleaned_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        lm_files = sorted(list(Path(self.paths["lm_dir"]).glob("*.csv")))
        if not lm_files:
            log.warning(f"処理対象のLMファイルが見つかりません: {self.paths['lm_dir']}")
            return

        for f in lm_files:
            log.info(f"LM クレンジング処理中: {f.name}")
            df = pd.read_csv(f)
            cleaned_df = clean_landmark_outliers(df)
            cleaned_df.to_csv(out_dir / f.name, index=False)

    def run_lm_resampling(self):
        """Step 4 (LM): リサンプリング (50Hz)"""
        log.info("=" * 60)
        log.info("▶ Step 4 (LM): ランドマークリサンプリング (50Hz)")
        log.info("=" * 60)
        in_dir = Path(self.paths["lm_cleaned_dir"])
        out_dir = Path(self.paths["lm_resampled_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(list(in_dir.glob("*.csv")))
        if not files:
            log.warning(f"処理対象ファイルが見つかりません: {in_dir}")
            return

        for f in files:
            log.info(f"LM リサンプリング処理中: {f.name}")
            df = pd.read_csv(f)
            df_resampled = resample_lm_dataframe(df, target_fs=self.fs)
            df_resampled.to_csv(out_dir / f.name, index=False)

    def run_lm_filtering(self):
        """Step 5 (LM): フィルタリング (ローパスフィルタ)"""
        log.info("=" * 60)
        log.info("▶ Step 5 (LM): ランドマークローパスフィルタ")
        log.info("=" * 60)
        in_dir = Path(self.paths["lm_resampled_dir"])
        out_dir = Path(self.paths["lm_filtered_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(list(in_dir.glob("*.csv")))
        if not files:
            log.warning(f"処理対象ファイルが見つかりません: {in_dir}")
            return

        for f in files:
            log.info(f"LM フィルタリング処理中: {f.name}")
            df = pd.read_csv(f)
            df_filtered = apply_lowpass_to_dataframe(df, fs=self.fs, default_cutoff=3.0)
            df_filtered.to_csv(out_dir / f.name, index=False)

    def run_lm_detrending(self):
        """Step 6 (LM): トレンド除去"""
        log.info("=" * 60)
        log.info("▶ Step 6 (LM): ランドマークトレンド除去")
        log.info("=" * 60)
        in_dir = Path(self.paths["lm_filtered_dir"])
        out_dir = Path(self.paths["lm_detrended_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(list(in_dir.glob("*.csv")))
        if not files:
            log.warning(f"処理対象ファイルが見つかりません: {in_dir}")
            return

        for f in files:
            log.info(f"LM トレンド除去処理中: {f.name}")
            df = pd.read_csv(f)
            df_detrended = detrend_landmark_dataframe(df, fs=self.fs)
            df_detrended.to_csv(out_dir / f.name, index=False)

    def run_lm_features(self):
        """Step 7 (LM): 特徴量計算 (EAR, Head Pose, Eye Rotation等)"""
        log.info("=" * 60)
        log.info("▶ Step 7 (LM): 幾何特徴量抽出 (EAR, 頭部姿勢, 視線)")
        log.info("=" * 60)
        in_dir = Path(self.paths["lm_detrended_dir"])
        out_dir = Path(self.paths["lm_features_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(list(in_dir.glob("*.csv")))
        if not files:
            log.warning(f"処理対象ファイルが見つかりません: {in_dir}")
            return

        for f in files:
            log.info(f"LM 幾何特徴量抽出中: {f.name}")
            df = pd.read_csv(f)
            df_features = extract_lm_features_dataframe(df)
            df_features.to_csv(out_dir / f.name, index=False)

    def run_lm_calibration(self):
        """Step 8 (LM): キャリブレーションを用いた正規化 (ランドマーク最終工程)"""
        log.info("=" * 60)
        log.info("▶ Step 8 (LM): キャリブレーション正規化 (最終出力)")
        log.info("=" * 60)
        in_dir = Path(self.paths.get("lm_features_dir", Path(self.paths["data_root"]) / "lm_features"))
        out_dir = Path(self.paths.get("lm_calibrated_dir", Path(self.paths["data_root"]) / "lm_calibrated"))
        out_dir.mkdir(parents=True, exist_ok=True)

        norm_dir = Path(self.paths.get("lm_normalized_dir", ""))
        if norm_dir and norm_dir != out_dir:
            norm_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(list(in_dir.glob("*.csv")))
        if not files:
            log.warning(f"処理対象ファイルが見つかりません: {in_dir}")
            return

        calib_master_path = Path(self.paths.get("calib_master_path", "data/raw/calibration_master.csv"))
        master_df = None
        if calib_master_path.exists():
            try:
                master_df = load_calibration_master(calib_master_path)
                log.info(f"✓ キャリブレーションマスター読み込み成功 ({len(master_df)} 件)")
            except Exception as e:
                log.warning(f"キャリブレーションマスターのロードに失敗しました: {e}")
        else:
            log.warning(f"キャリブレーションマスターが見つかりません: {calib_master_path}")

        for file_path in files:
            stem = file_path.stem
            subject_id = stem.replace("lm_", "")
            log.info(f"LM キャリブレーション正規化処理中: {file_path.name} (ID: {subject_id})")

            df = pd.read_csv(file_path)

            calib_row = {}
            if master_df is not None:
                if subject_id in master_df.index:
                    calib_row = master_df.loc[subject_id]
                else:
                    log.warning(f"  ⚠️ ID '{subject_id}' がマスターに存在しないため個人キャリブレーションなしで継続します")

            df_calib = calibrate_landmarks(df, calib_row, self.cfg)
            df_calib.to_csv(out_dir / file_path.name, index=False)
            if norm_dir and norm_dir != out_dir:
                df_calib.to_csv(norm_dir / file_path.name, index=False)

    def run_lm_postprocessing(self):
        """Step 8 (LM) を実行（後方互換用）"""
        self.run_lm_calibration()

    # エイリアス (簡潔な呼び出し用)
    run_cleaning = run_lm_cleaning
    run_resampling = run_lm_resampling
    run_filtering = run_lm_filtering
    run_detrending = run_lm_detrending
    run_features = run_lm_features
    run_calibration = run_lm_calibration

    def cleanup_intermediates(self, include_cutting: bool = False):
        """
        ランドマーク工程の中間生成物ディレクトリをクリーンアップします。
        最終成果物 (lm_calibrated_dir / lm_normalized_dir) は保持されます。
        """
        log.info("🧹 ランドマーク中間生成物のクリーンアップを開始します...")
        keys = list(INTERMEDIATE_LM_KEYS)
        if not include_cutting and "lm_dir" in keys:
            keys.remove("lm_dir")

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
        log.info(f"✨ ランドマーク中間クリーンアップ完了 ({deleted_count} ディレクトリ削除)")
        log.info(f"📁 最終成果物を保持: {self.paths.get('lm_calibrated_dir', self.paths.get('lm_normalized_dir'))}")

    def run_pipeline(self, keep_intermediates: bool = None):
        """ランドマークの前処理・特徴量抽出全工程 (Step 3〜8) を順次実行"""
        if keep_intermediates is None:
            keep_intermediates = not self.cfg.get("cleanup_intermediates", True)

        log.info("\n" + "#" * 60)
        log.info("  🔷 ランドマーク パイプライン全工程 (Step 3〜8) 開始")
        log.info("#" * 60)
        try:
            self.run_lm_cleaning()       # Step 3
            self.run_lm_resampling()     # Step 4
            self.run_lm_filtering()      # Step 5
            self.run_lm_detrending()     # Step 6
            self.run_lm_features()       # Step 7
            self.run_lm_calibration()    # Step 8
        except Exception as e:
            log.error(f"❌ ランドマーク工程中にエラーが発生しました: {e}")
            log.warning("⚠️ 処理が中断されたため、成功した工程の中間ファイルはそのまま保持されます。")
            raise e

        if not keep_intermediates:
            self.cleanup_intermediates(include_cutting=False)
        log.info("✓ ランドマーク全工程が完了しました\n")

    def run_all(self, keep_intermediates: bool = None):
        """ダウンロード・切り出しを含めたランドマーク全工程を実行"""
        if keep_intermediates is None:
            keep_intermediates = not self.cfg.get("cleanup_intermediates", True)

        log.info("🚀 ランドマーク パイプライン (ダウンロード〜全8工程) 開始")
        try:
            self.run_download()
            self.run_cutting()
            self.run_pipeline(keep_intermediates=True)
        except Exception as e:
            log.error(f"❌ ランドマーク パイプライン中にエラーが発生しました: {e}")
            log.warning("⚠️ 処理が中断されたため、成功した工程の中間ファイルはそのまま保持されます。")
            raise e

        if not keep_intermediates:
            self.cleanup_intermediates(include_cutting=True)
        log.info("🎉 ランドマーク処理が正常に完了しました！")


if __name__ == "__main__":
    import argparse
    from config_loader import load_config

    parser = argparse.ArgumentParser(description="顔ランドマーク前処理・特徴量抽出パイプライン")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定YAMLファイル名")
    parser.add_argument(
        "--stage",
        nargs="+",
        default=["pipeline"],
        help="実行ステージ (all, pipeline, download, cutting, clean, resample, filter, detrend, feat, calib)",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        default=False,
        help="中間ファイルを削除せずに保持する (デフォルト: False, 成功時は最終ファイルのみ残す)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipeline = LandmarkPipeline(cfg)

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
            if "clean" in stages or "lm_clean" in stages:
                pipeline.run_lm_cleaning()
            if "resample" in stages or "lm_resample" in stages:
                pipeline.run_lm_resampling()
            if "filter" in stages or "lm_filter" in stages:
                pipeline.run_lm_filtering()
            if "detrend" in stages or "lm_detrend" in stages:
                pipeline.run_lm_detrending()
            if "feat" in stages or "lm_feat" in stages:
                pipeline.run_lm_features()
            if "calib" in stages or "lm_calib" in stages or "lm_calibration" in stages:
                pipeline.run_lm_calibration()
        except Exception as e:
            log.error(f"❌ ランドマーク個別ステージ実行中にエラーが発生しました: {e}")
            log.warning("⚠️ 途中の工程で失敗したため、成功した部分のファイルは保持されます。")
            raise e
