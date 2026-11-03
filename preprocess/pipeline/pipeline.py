import os
import shutil
import json
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import sys

# 全工程実行時の中間生成物キー一覧 (最終成果物: raw, calibration_master, features, windowed)
INTERMEDIATE_ALL_KEYS = [
    "lm_dir",
    "rgb_dir",
    "lm_cleaned_dir",
    "rgb_cleaned_dir",
    "lm_resampled_dir",
    "rgb_resampled_dir",
    "lm_filtered_dir",
    "lm_detrended_dir",
    "lm_features_dir",
    "rppg_pos_extracted_dir",
    "rppg_filtered_dir",
    "rppg_hr_extracted_dir",
]

# 親ディレクトリ（preprocess）を sys.path に追加して各前処理モジュールを解決
_PARENT_DIR = str(Path(__file__).resolve().parent.parent)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from config_loader import log
from calibration.lm_calibration import load_calibration_master

try:
    from .lm_pipeline import LandmarkPipeline
    from .rgb_pipeline import RGBPipeline
except (ImportError, ValueError):
    from lm_pipeline import LandmarkPipeline
    from rgb_pipeline import RGBPipeline

from windowing.sliding_windows import run_windowing


class DataPipeline:
    """
    生体シグナル（ランドマーク & RGB/rPPG）前処理・特徴量抽出統合パイプライン。
    
    【ランドマーク工程】 (lm_pipeline.py: 計8工程)
      1. Storageからファイル取得 (run_download)
      2. Landmark, RGB値別にファイルを作成 (run_cutting)
      3. 欠損値・外れ値除去 (run_lm_cleaning)
      4. リサンプリング (50Hz) (run_lm_resampling)
      5. フィルタリング (ローパスフィルタ) (run_lm_filtering)
      6. トレンド除去 (run_lm_detrending)
      7. 特徴量計算 (EAR, Head Pose, Eye Rotation等) (run_lm_features)
      8. キャリブレーションを用いた正規化 (run_lm_calibration)
      
    【RGB値工程】 (rgb_pipeline.py: 計7工程)
      1. Storageからファイル取得 (run_download)
      2. Landmark, RGB値別にファイルを作成 (run_cutting)
      3. 欠損値・外れ値除去 (run_rgb_cleaning)
      4. リサンプリング (50Hz) (run_rgb_resampling)
      5. POS Methodから脈波を検出 (run_rgb_pos)
      6. 脈波をbandpassフィルタリング (run_rgb_filtering)
      7. 生体指標算出 (run_rgb_hr_extraction)
      
    【統合】
      - run_merge_features: ランドマーク特徴量と生体指標を被験者単位でマージ
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.paths = cfg["paths"]
        self.fs = float(self.cfg.get("rppg", {}).get("fs", 50.0))
        self.lm_pipeline = LandmarkPipeline(cfg)
        self.rgb_pipeline = RGBPipeline(cfg)

    # =========================================================================
    # 共通工程 (Common Stages)
    # =========================================================================

    def run_download(self):
        """Step 1: Firebase Storage からデータ取得 & 初期変換 (gz -> json -> CSV & calib)"""
        self.lm_pipeline.run_download()

    def run_cutting(self):
        """Step 2: Landmark, RGB値別にファイルを作成 (raw/*.csv -> lm/*.csv, rgb/*.csv)"""
        self.lm_pipeline.run_cutting()

    # =========================================================================
    # ランドマーク工程 (Landmark Pipeline: Steps 3 - 8)
    # =========================================================================

    def run_lm_cleaning(self):
        """Step 3 (LM): 欠損値・外れ値除去"""
        self.lm_pipeline.run_lm_cleaning()

    def run_lm_resampling(self):
        """Step 4 (LM): リサンプリング (50Hz 等間隔化)"""
        self.lm_pipeline.run_lm_resampling()

    def run_lm_filtering(self):
        """Step 5 (LM): フィルタリング (ローパスフィルタ)"""
        self.lm_pipeline.run_lm_filtering()

    def run_lm_detrending(self):
        """Step 6 (LM): トレンド除去 (移動平均差分、ベースライン補正、線形デトレンド)"""
        self.lm_pipeline.run_lm_detrending()

    def run_lm_features(self):
        """Step 7 (LM): 特徴量計算 (EAR, 頭部姿勢, 眼球回転角)"""
        self.lm_pipeline.run_lm_features()

    def run_lm_calibration(self):
        """Step 8 (LM): キャリブレーションを用いた正規化 (最終出力)"""
        self.lm_pipeline.run_lm_calibration()

    def run_lm_postprocessing(self):
        """Step 8 (LM) を実行（後方互換用）"""
        self.lm_pipeline.run_lm_postprocessing()

    # エイリアス
    run_lm_clean = run_lm_cleaning
    run_lm_resample = run_lm_resampling
    run_lm_filter = run_lm_filtering
    run_lm_detrend = run_lm_detrending
    run_lm_feat = run_lm_features
    run_lm_calib = run_lm_calibration

    # =========================================================================
    # 脈波・RGB値工程 (RGB Pipeline: Steps 3 - 7)
    # =========================================================================

    def run_rgb_cleaning(self):
        """Step 3 (RGB): ランドマーク外れ値に基づく同期欠損処理"""
        self.rgb_pipeline.run_rgb_cleaning()

    def run_rgb_resampling(self):
        """Step 4 (RGB): リサンプリング (50Hz 等間隔化)"""
        self.rgb_pipeline.run_rgb_resampling()

    def run_rgb_pos(self):
        """Step 5 (RGB): POS法による脈波抽出"""
        self.rgb_pipeline.run_rgb_pos()

    def run_rgb_filtering(self):
        """Step 6 (RGB): バンドパスフィルタリング (0.7 - 2.5Hz)"""
        self.rgb_pipeline.run_rgb_filtering()

    def run_rgb_hr_extraction(self):
        """Step 7 (RGB): ピーク検出・HR・SNR・HRV算出"""
        self.rgb_pipeline.run_rgb_hr_extraction()

    # エイリアス
    run_rgb_clean = run_rgb_cleaning
    run_rgb_resample = run_rgb_resampling
    run_rgb_filter = run_rgb_filtering
    run_rgb_hr = run_rgb_hr_extraction

    # =========================================================================
    # サブパイプライン実行
    # =========================================================================

    def run_lm_pipeline(self, keep_intermediates: bool = None):
        """ランドマークの前処理・特徴量抽出全工程 (Step 3〜8) を実行"""
        self.lm_pipeline.run_pipeline(keep_intermediates=keep_intermediates)

    def run_rgb_pipeline(self, keep_intermediates: bool = None):
        """脈波の前処理・特徴量抽出全工程 (Step 3〜7) を実行"""
        self.rgb_pipeline.run_pipeline(keep_intermediates=keep_intermediates)

    # =========================================================================
    # 特徴量統合工程 (Merge Features)
    # =========================================================================

    def run_merge_features(self):
        """
        ランドマーク特徴量 (Step 8 出力) と RGB生体指標 (Step 7 出力) を被験者単位で統合し、
        最終データセットとして output_dir (data/features) へ保存します。
        """
        log.info("=" * 60)
        log.info("▶ 最終工程: ランドマーク特徴量 & RGB生体指標の統合 (Merge Features)")
        log.info("=" * 60)
        lm_dir = Path(self.paths.get("lm_calibrated_dir", self.paths.get("lm_normalized_dir", "data/lm_calibrated")))
        if not lm_dir.exists() or not list(lm_dir.glob("*.csv")):
            alt_dir = Path(self.paths.get("lm_normalized_dir", "data/lm_normalized"))
            if alt_dir.exists() and list(alt_dir.glob("*.csv")):
                lm_dir = alt_dir

        rgb_dir = Path(self.paths["rppg_hr_extracted_dir"])
        out_dir = Path(self.paths["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        lm_files = sorted(list(lm_dir.glob("*.csv")))
        if not lm_files:
            log.warning(f"統合対象のLMデータが見つかりません: {lm_dir}")
            return

        merged_count = 0
        for lm_path in lm_files:
            stem = lm_path.stem.replace("lm_", "")
            rgb_path = rgb_dir / f"rgb_{stem}.csv"

            df_lm = pd.read_csv(lm_path)

            if rgb_path.exists():
                df_rgb = pd.read_csv(rgb_path)

                # RGB生体指標から結合対象とするカラムを抽出
                rppg_feature_cols = [
                    c for c in df_rgb.columns 
                    if c in [
                        'bvp_selected', 'selected_roi', 'snr_db', 'bpm', 'bpm_raw', 'ppg_peaks', 
                        'bvp_clean', 'bvp_clean_both', 'bpm_diff', 'bvp_selected_diff', 
                        'bpm_base_10s', 'bvp_selected_base_10s',
                        'rmssd', 'sdnn', 'sd2_sd1', 'mean_pp',
                    ]
                ]

                # タイムスタンプ結合（時間同期）
                if 'time' in df_lm.columns and 'time' in df_rgb.columns:
                    # 時間軸が完全に一致している場合は直接代入
                    if len(df_lm) == len(df_rgb) and np.allclose(df_lm['time'].values, df_rgb['time'].values, atol=1e-3):
                        for col in rppg_feature_cols:
                            df_lm[col] = df_rgb[col].values
                    else:
                        # タイムスタンプ最近傍結合 (許容誤差 0.05秒)
                        merged = pd.merge_asof(
                            df_lm.sort_values('time'),
                            df_rgb[['time'] + rppg_feature_cols].sort_values('time'),
                            on='time',
                            direction='nearest',
                            tolerance=0.05
                        )
                        df_lm = merged
                else:
                    min_len = min(len(df_lm), len(df_rgb))
                    df_lm = df_lm.iloc[:min_len].copy()
                    for col in rppg_feature_cols:
                        df_lm[col] = df_rgb[col].iloc[:min_len].values

                log.info(f"  ✓ 結合成功: {stem} (LM + RGB生体指標, 行数: {len(df_lm)})")
            else:
                log.warning(f"  ⚠️ RGBデータが見つかりません ({rgb_path.name})。LMデータのみ保存します。")

            output_file = out_dir / f"{stem}.csv"
            df_lm.to_csv(output_file, index=False)
            merged_count += 1

        log.info(f"🏆 合計 {merged_count} 件の最終特徴量ファイルを保存完了 ➔ {out_dir}")

    # =========================================================================
    # スライディングウィンドウ作成 (Windowing)
    # =========================================================================

    def run_windowing(self):
        """スライディングウィンドウ作成 (被験者別・ラベル 0/1 別 & 集約)"""
        log.info("=" * 60)
        log.info("▶ スライディングウィンドウ分割 (Windowing by User & Label)")
        log.info("=" * 60)
        run_windowing(self.cfg)

    # =========================================================================
    # 中間ファイル クリーンアップ処理
    # =========================================================================

    def cleanup_all_intermediates(self):
        """全中間ディレクトリのクリーンアップ"""
        log.info("🧹 全パイプライン中間生成物のクリーンアップを開始します...")
        keys = list(INTERMEDIATE_ALL_KEYS)
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
        log.info(f"✨ 中間クリーンアップ完了 ({deleted_count} ディレクトリ削除)")
        log.info("📁 保持された最終成果物:")
        log.info(f"  - 生データ: {self.paths.get('raw_dir')}")
        log.info(f"  - 統合特徴量: {self.paths.get('output_dir')}")
        win_out = self.cfg.get("windowing", {}).get("output_dir", "")
        if win_out:
            log.info(f"  - ウィンドウ化データ: {win_out}")

    def cleanup_for_stages(self, stages: set[str]):
        """
        指定されたステージセットに応じた中間生成物のクリーンアップを行います。
        """
        if "all" in stages:
            self.cleanup_all_intermediates()
            return

        # merge または windowing が含まれている場合
        if "merge" in stages or "windowing" in stages:
            self.cleanup_all_intermediates()
            return

        # lm 単体
        if "lm" in stages and not ("rgb" in stages or "merge" in stages or "windowing" in stages):
            self.lm_pipeline.cleanup_intermediates(include_cutting=False)
            return

        # rgb 単体
        if "rgb" in stages and not ("lm" in stages or "merge" in stages or "windowing" in stages):
            self.rgb_pipeline.cleanup_intermediates(include_cutting=False)
            return

    # =========================================================================
    # 全工程一括実行 (Run All)
    # =========================================================================

    def merge_dataframes(self, df_lm: pd.DataFrame, df_rgb: pd.DataFrame) -> pd.DataFrame:
        """ランドマーク DataFrame と RGB生体指標 DataFrame を時間同期結合します。"""
        rppg_feature_cols = [
            c for c in df_rgb.columns 
            if c in [
                'bvp_selected', 'selected_roi', 'snr_db', 'bpm', 'bpm_raw', 'ppg_peaks', 
                'bvp_clean', 'bvp_clean_both', 'bpm_diff', 'bvp_selected_diff', 
                'bpm_base_10s', 'bvp_selected_base_10s',
                'rmssd', 'sdnn', 'sd2_sd1', 'mean_pp',
            ]
        ]
        df_out = df_lm.copy()
        if 'time' in df_out.columns and 'time' in df_rgb.columns:
            if len(df_out) == len(df_rgb) and np.allclose(df_out['time'].values, df_rgb['time'].values, atol=1e-3):
                for col in rppg_feature_cols:
                    df_out[col] = df_rgb[col].values
            else:
                merged = pd.merge_asof(
                    df_out.sort_values('time'),
                    df_rgb[['time'] + rppg_feature_cols].sort_values('time'),
                    on='time',
                    direction='nearest',
                    tolerance=0.05
                )
                df_out = merged
        else:
            min_len = min(len(df_out), len(df_rgb))
            df_out = df_out.iloc[:min_len].copy()
            for col in rppg_feature_cols:
                df_out[col] = df_rgb[col].iloc[:min_len].values
        return df_out

    def run_streamlined_pipeline(self):
        """
        【インメモリ被験者別ストリームライン処理】
        中間ディスクファイルへの保存を完全撤廃し、生データから被験者単位で
        メモリ上で切り出し ➔ LM処理 ➔ RGB処理 ➔ マージを行い、
        最終特徴量 (data/features/{stem}.csv) のみを保存します。
        その後、スライディングウィンドウ分割 (windowing) を実行します。
        """
        import time
        log.info("=" * 70)
        log.info("🚀 インメモリ被験者別ストリームライン前処理を開始します (中間ディスクI/Oゼロ)")
        log.info("=" * 70)

        raw_dir = Path(self.paths["raw_dir"])
        out_dir = Path(self.paths["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        calib_path = Path(self.paths.get("calib_master_path", "data/raw/calibration_master.csv"))
        master_df = None
        if calib_path.exists():
            try:
                master_df = load_calibration_master(calib_path)
                log.info(f"✓ キャリブレーションマスター読み込み成功 ({len(master_df)} 件)")
            except Exception as e:
                log.warning(f"キャリブレーションマスター読み込みエラー: {e}")

        raw_candidates = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("**/*.csv"))
        seen = set()
        raw_files = []
        for p in raw_candidates:
            if p.is_file() and "calibration_master" not in p.name and p.resolve() not in seen:
                seen.add(p.resolve())
                raw_files.append(p)
        raw_files = sorted(raw_files)

        if not raw_files:
            log.warning(f"処理対象の生データが見つかりません: {raw_dir}")
            return

        log.info(f"計 {len(raw_files)} 名の被験者データをメモリ上で順次処理します。")

        for idx, file_path in enumerate(raw_files, start=1):
            stem = file_path.stem
            t_sub = time.time()

            # 1. 読み込み & ユーザーID付与
            df_raw = pd.read_csv(file_path)
            df_raw["username"] = stem

            # 相対時間 time のリセット・巻き戻りを解消し、単調増加する連続時間軸を構築
            if "clientEpochMs" in df_raw.columns:
                df_raw["time"] = (df_raw["clientEpochMs"] - df_raw["clientEpochMs"].iloc[0]) / 1000.0
            elif "clientPerfMs" in df_raw.columns:
                df_raw["time"] = (df_raw["clientPerfMs"] - df_raw["clientPerfMs"].iloc[0]) / 1000.0

            # 2. メモリ上切り出し (LM & RGB)
            base_cols = [c for c in ['time', 'clientEpochMs', 'clientPerfMs', 'label', 'username'] if c in df_raw.columns]
            lm_cols = [c for c in df_raw.columns if (c.startswith('lm_') or c.startswith('centroid_')) and c not in base_cols]
            rgb_cols = [c for c in df_raw.columns if c.startswith('rppg_') and c not in base_cols]

            df_lm = df_raw[base_cols + lm_cols].copy()
            df_rgb = df_raw[base_cols + rgb_cols].copy()

            # 3. ランドマーク工程 (インメモリ: Step 3〜8)
            df_lm_norm, df_lm_clean = self.lm_pipeline.process_dataframe(
                df_lm=df_lm,
                subject_id=stem,
                master_df=master_df,
            )

            # 4. RGB工程 (インメモリ, クレンジングはLMクレンジングデータと同期)
            df_rgb_hr = self.rgb_pipeline.process_dataframe(
                df_rgb=df_rgb,
                df_lm_clean=df_lm_clean,
            )

            # 5. 時間同期マージ (インメモリ)
            df_merged = self.merge_dataframes(df_lm_norm, df_rgb_hr)

            # 6. 最終特徴量のみ保存
            out_file = out_dir / f"{stem}.csv"
            df_merged.to_csv(out_file, index=False)
            elapsed = time.time() - t_sub
            l0_pts = (df_merged["label"] == 0).sum() if "label" in df_merged.columns else 0
            l1_pts = (df_merged["label"] == 1).sum() if "label" in df_merged.columns else 0
            log.info(
                f"  [{idx:2d}/{len(raw_files):2d}] {stem:<14} ➔ 統合完了: {len(df_merged):,d}行 "
                f"(Label 1: {l1_pts:,d}, Label 0: {l0_pts:,d}) [{elapsed:.1f}s]"
            )

        log.info("\n" + "=" * 70)
        log.info("✓ 全被験者のインメモリ特徴量抽出が完了しました。")
        log.info("=" * 70)

        # 7. スライディングウィンドウ作成
        self.run_windowing()

    def run_all(self, keep_intermediates: bool = None):
        """全パイプライン（共通・LM・RGB・統合・ウィンドウ化）を順次実行する"""
        if keep_intermediates is None:
            keep_intermediates = not self.cfg.get("cleanup_intermediates", True)

        log.info("🚀 パイプライン全行程 (ランドマーク8工程 & RGB値7工程 & ウィンドウ分割) の一括実行を開始します")
        try:
            self.run_download()
            self.run_streamlined_pipeline()
        except Exception as e:
            log.error(f"❌ パイプライン実行中にエラーが発生しました: {e}")
            log.warning("⚠️ 途中の工程で失敗したため、成功した部分のファイルは保持されます。")
            raise e

        if not keep_intermediates:
            self.cleanup_all_intermediates()

        log.info("🎉 全工程の処理が正常に完了しました！")
