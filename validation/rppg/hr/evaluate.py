"""
evaluate.py
rppg_datasets の各フォルダに含まれる動画 (vid.avi) に対し、
1. config.yaml によるパラメータ管理 (サンプリングレート, ROI座標, フィルタ帯域等)
2. FaceLandmarker を適用して 25fps で指定ランドマークの RGB値を取得
3. preprocess の rppg 前処理 (クレンジング, 50Hzリサンプリング, POS脈波抽出, バンドパスフィルタ, 最適ROI選択, 心拍数抽出) を適用
4. 出力された心拍数 (BPM) と ground_truth.txt の真値 (HR) との誤差を計算
5. 比較グラフ画像 (Code_Generated_Image.png 形式) を output_dir (デフォルト: result) に保存
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import yaml

# ディレクトリパス設定 (rppg, validation)
_RPPG_DIR = Path(__file__).resolve().parent.parent
_VALIDATION_DIR = _RPPG_DIR.parent
for _p in [_VALIDATION_DIR, _RPPG_DIR]:
  if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

# 自ディレクトリ (hr) のパス設定
_HR_DIR = Path(__file__).resolve().parent
if str(_HR_DIR) not in sys.path:
  sys.path.insert(0, str(_HR_DIR))

from config_loader import load_config, log
from face_landmarker_extractor import FaceLandmarkerRGBExtractor
from metrics_evaluator import calculate_hr_errors, load_ground_truth
from plotter import plot_hr_comparison, plot_individual_subject
from rppg_preprocessor import RPPGPreprocessor


def run_evaluation(config_path: Path | str, cli_overrides: Optional[dict] = None):
  # 共通 config_loader による設定読み込み、CLI反映、パス解決
  cfg = load_config(config_path, cli_overrides=cli_overrides)
  paths_cfg = cfg.get("paths", {})
  dataset_dir = paths_cfg["dataset_dir"]
  output_dir = paths_cfg["output_dir"]
  model_path = paths_cfg.get("model_path")

  # Extraction 設定
  ext_cfg = cfg.get("extraction", {})
  target_fps = float(ext_cfg.get("target_fps", 25.0))
  region_size = int(ext_cfg.get("region_size", 20))
  lm_cfg = ext_cfg.get("landmarks", {})
  left_cheek = lm_cfg.get("left_cheek", [330, 347, 280, 266, 425])
  right_cheek = lm_cfg.get("right_cheek", [101, 118, 50, 36, 198])

  # rPPG 前処理設定
  rppg_cfg = cfg.get("rppg", {})
  fs = float(rppg_cfg.get("fs", 50.0))
  input_fps = int(rppg_cfg.get("input_fps", 25))
  pos_cfg = rppg_cfg.get("pos", {})
  pos_window_sec = float(pos_cfg.get("window_sec", 1.6))
  filt_cfg = rppg_cfg.get("filter", {})
  lowcut = float(filt_cfg.get("lowcut", 0.75))
  highcut = float(filt_cfg.get("highcut", 2.5))
  filter_order = int(filt_cfg.get("order", 2))
  hr_cfg = rppg_cfg.get("hr_extraction", {})
  roi_cols = hr_cfg.get(
      "roi_cols",
      {
          "both": "bvp_clean_both",
          "left": "bvp_clean_left",
          "right": "bvp_clean_right",
      },
  )

  # 移動平均設定
  ma_cfg = cfg.get("moving_average", {})
  ma_enabled = ma_cfg.get("enabled", True)
  ma_window_sec = float(ma_cfg.get("window_sec", 5.0))
  ma_center = ma_cfg.get("center", False)

  # 可視化設定
  vis_cfg = cfg.get("visualization", {})
  save_plots = vis_cfg.get("save_plots", True)
  combined_plot_filename = vis_cfg.get(
      "combined_plot_filename", "hr_comparison.png"
  )
  save_individual_plots = vis_cfg.get("save_individual_plots", True)
  dpi = int(vis_cfg.get("dpi", 300))

  # 被験者フォルダの決定
  specified_subjects = cfg.get("subjects")
  if specified_subjects:
    subject_dirs = [dataset_dir / s for s in specified_subjects]
  else:
    subject_dirs = sorted(
        [d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )

  if not subject_dirs:
    print(f"❌ 被験者フォルダが見つかりませんでした: {dataset_dir}")
    return

  print("=" * 75)
  print(" remote-PPG (rPPG) 心拍数推定 & 精度評価パイプライン")
  print("=" * 75)
  print(f"📄 設定ファイル: {config_path}")
  print(f"📁 データセットディレクトリ: {dataset_dir}")
  print(f"💾 結果出力先 (result): {output_dir}")
  print(f"🎯 抽出サンプリングレート: {target_fps} fps (ROIサイズ: {region_size}px)")
  print(f"📍 左頬ランドマーク: {left_cheek}")
  print(f"📍 右頬ランドマーク: {right_cheek}")
  print(
      f"⚙️  前処理パラメータ: fs={fs}Hz, POS窓長={pos_window_sec}s,"
      f" 帯域=[{lowcut}, {highcut}]Hz (次数: {filter_order})"
  )
  if ma_enabled and ma_window_sec > 0:
    print(
        f"📊 BPM移動平均: 有効 (窓長: {ma_window_sec}秒,"
        f" center={'True' if ma_center else 'False'})"
    )
  else:
    print("📊 BPM移動平均: 無効 (生BPMを出力)")
  print(f"👥 対象被験者数: {len(subject_dirs)} 件")
  print("=" * 75)

  results = []
  comparison_records = []

  preprocessor = RPPGPreprocessor(
      fs=fs,
      input_fps=input_fps,
      pos_window_sec=pos_window_sec,
      lowcut=lowcut,
      highcut=highcut,
      filter_order=filter_order,
      roi_cols=roi_cols,
      ma_window_sec=ma_window_sec,
      ma_enabled=ma_enabled,
      ma_center=ma_center,
  )

  with FaceLandmarkerRGBExtractor(
      model_path=model_path,
      region_size=region_size,
      target_fps=target_fps,
      left_cheek_idxs=left_cheek,
      right_cheek_idxs=right_cheek,
  ) as extractor:
    for idx, s_dir in enumerate(subject_dirs, 1):
      s_name = s_dir.name
      avi_path = s_dir / "vid.avi"
      gt_path = s_dir / "ground_truth.txt"

      print(f"\n[{idx}/{len(subject_dirs)}] 被験者: {s_name} の処理を開始...")

      if not avi_path.exists():
        print(f"  ⚠️ vid.avi が存在しないためスキップします: {avi_path}")
        continue
      if not gt_path.exists():
        print(
            f"  ⚠️ ground_truth.txt が存在しないためスキップします: {gt_path}"
        )
        continue

      t_start = time.time()

      # 1. Ground Truth の読み込み
      try:
        gt_ppg, gt_hr, gt_timesteps = load_ground_truth(gt_path)
        print(
            f"  ✓ GT読込完了: 計 {len(gt_hr)} フレーム (録画時間:"
            f" {gt_timesteps[-1] - gt_timesteps[0]:.2f} 秒, 平均HR:"
            f" {np.mean(gt_hr):.1f} BPM)"
        )
      except Exception as e:
        print(f"  ❌ GT読込失敗: {e}")
        continue

      # 2. FaceLandmarker による 25fps RGB値抽出
      try:
        print(f"  ▶ Step 1: FaceLandmarker 適用 & {target_fps}fps RGB抽出中...")
        df_rgb = extractor.extract_rgb_from_video(
            avi_path, gt_timestamps=gt_timesteps
        )
        print(
            f"    └─ 抽出完了: {len(df_rgb)} サンプル ({target_fps}fps時系列,"
            f" ランドマーク{len(left_cheek)+len(right_cheek)}点 +"
            " 左右/両頬平均)"
        )
      except Exception as e:
        print(f"  ❌ RGB抽出失敗: {e}")
        continue

      # 3. preprocess と同一の前処理パイプライン適用
      try:
        print(f"  ▶ Step 2: preprocess rPPG 前処理パイプライン適用中...")
        df_hr, selected_roi, snr_db = preprocessor.process(df_rgb)
        print(
            f"    └─ 前処理完了 (最適ROI: {selected_roi}, SNR: {snr_db:.2f} dB,"
            f" 推定平均HR: {df_hr['bpm'].mean():.1f} BPM)"
        )
      except Exception as e:
        print(f"  ❌ 前処理失敗: {e}")
        continue

      # 4. 誤差計算
      try:
        print(f"  ▶ Step 3: GT心拍数との誤差計算中...")
        metrics = calculate_hr_errors(
            df_hr=df_hr,
            gt_hr=gt_hr,
            gt_timesteps=gt_timesteps,
            subject_name=s_name,
            selected_roi=selected_roi,
            snr_db=snr_db,
        )

        elapsed = time.time() - t_start
        print(f"  ✨ {s_name} 完了 ({elapsed:.1f} 秒)")
        print(
            f"    ├─ MAE (平均絶対誤差): {metrics['mae']} BPM"
            f" / RMSE: {metrics['rmse']} BPM"
        )
        print(
            f"    ├─ Mean Error (バイアス): {metrics['mean_error']} BPM"
            f" (SD: {metrics['std_error']})"
        )
        print(
            f"    ├─ 相関係数 (r): {metrics['pearson_r']} (GT平均:"
            f" {metrics['gt_hr_mean']} BPM vs 推定平均:"
            f" {metrics['est_bpm_mean']} BPM)"
        )
        print(f"    └─ 採択領域: {selected_roi} (SNR: {metrics['snr_db']} dB)")

        results.append(metrics)

        # 被験者ごとの時系列予測結果 CSV 保存
        est_times = df_hr["time"].values
        gt_hr_synced = np.interp(est_times, gt_timesteps, gt_hr)
        df_comparison = pd.DataFrame({
            "time": est_times,
            "bvp_selected": df_hr["bvp_selected"],
            "bpm_raw": df_hr["bpm_raw"] if "bpm_raw" in df_hr else df_hr["bpm"],
            "bpm_pred": df_hr["bpm"],
            "ppg_peaks": df_hr["ppg_peaks"],
            "hr_true": gt_hr_synced,
            "error": df_hr["bpm"] - gt_hr_synced,
            "abs_error": np.abs(df_hr["bpm"] - gt_hr_synced),
        })
        comp_path = output_dir / f"pred_{s_name}.csv"
        df_comparison.to_csv(comp_path, index=False)

        comp_item = {
            "subject": s_name,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "df_comparison": df_comparison,
        }
        comparison_records.append(comp_item)

        # 個別プロット
        if save_plots and save_individual_plots:
          ind_plot_path = output_dir / f"plot_{s_name}.png"
          plot_individual_subject(comp_item, ind_plot_path, dpi=dpi)

      except Exception as e:
        print(f"  ❌ 誤差計算失敗: {e}")
        continue

  if not results:
    print("\n❌ 有効な評価結果が得られませんでした。")
    return

  # 結果の集計とサマリー CSV 出力
  df_summary = pd.DataFrame(results)
  summary_csv_path = output_dir / "evaluation_summary.csv"
  df_summary.to_csv(summary_csv_path, index=False)

  # 全体統合比較グラフの作成・保存 (Code_Generated_Image.png 形式)
  if save_plots and comparison_records:
    combined_plot_path = output_dir / combined_plot_filename
    plot_hr_comparison(comparison_records, combined_plot_path, dpi=dpi)

  print("\n" + "=" * 90)
  print("                     rPPG 心拍数推定 総合評価結果サマリー")
  print("=" * 90)
  print(
      f"{'Subject':<10} | {'ROI':<6} | {'SNR(dB)':<7} | {'GT HR':<7} |"
      f" {'Est HR':<7} | {'MAE(BPM)':<8} | {'RMSE':<6} | {'Error(Bias)':<11} |"
      f" {'Corr(r)':<7}"
  )
  print("-" * 90)
  for _, row in df_summary.iterrows():
    print(
        f"{row['subject']:<10} | {row['selected_roi']:<6} |"
        f" {row['snr_db']:>7.2f} | {row['gt_hr_mean']:>7.2f} |"
        f" {row['est_bpm_mean']:>7.2f} | {row['mae']:>8.2f} |"
        f" {row['rmse']:>6.2f} | {row['mean_error']:>11.2f} |"
        f" {row['pearson_r']:>7.3f}"
    )
  print("-" * 90)

  mean_mae = df_summary["mae"].mean()
  mean_rmse = df_summary["rmse"].mean()
  mean_bias = df_summary["mean_error"].mean()
  mean_r = df_summary["pearson_r"].mean()
  mean_snr = df_summary["snr_db"].mean()

  print(
      f"{'MEAN':<10} | {'-':<6} | {mean_snr:>7.2f} |"
      f" {df_summary['gt_hr_mean'].mean():>7.2f} |"
      f" {df_summary['est_bpm_mean'].mean():>7.2f} | {mean_mae:>8.2f} |"
      f" {mean_rmse:>6.2f} | {mean_bias:>11.2f} | {mean_r:>7.3f}"
  )
  print("=" * 90)
  print(f"\n📊 評価結果サマリーCSV: {summary_csv_path}")
  if save_plots:
    print(f"📈 比較グラフ画像 (統合): {output_dir / combined_plot_filename}")
  print(f"📁 各種データ保存先: {output_dir}")


def main():
  parser = argparse.ArgumentParser(
      description=(
          "rppg_datasets に対する FaceLandmarker + preprocess パイプライン心拍数評価"
      )
  )
  parser.add_argument(
      "--config",
      type=str,
      default=str(Path(__file__).resolve().parent / "config.yaml"),
      help="設定ファイル (config.yaml) のパス",
  )
  parser.add_argument(
      "--dataset_dir",
      type=str,
      default=None,
      help="rppg_datasets ディレクトリのパス (config.yaml を上書き)",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default=None,
      help="結果出力ディレクトリ (config.yaml を上書き)",
  )
  parser.add_argument(
      "--subject",
      nargs="+",
      default=None,
      help="特定の被験者のみ実行する場合 (例: --subject subject1 subject5)",
  )
  parser.add_argument(
      "--target_fps",
      type=float,
      default=None,
      help="サンプリングレート (config.yaml を上書き)",
  )
  parser.add_argument(
      "--ma_window_sec",
      type=float,
      default=None,
      help="移動平均の窓長 (秒) (config.yaml を上書き)",
  )
  parser.add_argument(
      "--no_ma",
      action="store_true",
      default=None,
      help="移動平均を無効化する",
  )
  parser.add_argument(
      "--model_path",
      type=str,
      default=None,
      help="face_landmarker.task のパス (config.yaml を上書き)",
  )
  args = parser.parse_args()

  cli_overrides = {
      "dataset_dir": args.dataset_dir,
      "output_dir": args.output_dir,
      "subjects": args.subject,
      "target_fps": args.target_fps,
      "ma_window_sec": args.ma_window_sec,
      "no_ma": args.no_ma,
      "model_path": args.model_path,
  }

  run_evaluation(config_path=Path(args.config), cli_overrides=cli_overrides)


if __name__ == "__main__":
  main()
