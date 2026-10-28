"""
evaluate_hrv.py
rppg_datasets の各被験者データに対し、
1. FaceLandmarker による 25/30fps RGB値抽出
2. preprocess と全く同じコードによる前処理 (POS脈波抽出、バンドパスフィルタ、最適ROI選択)
3. 【HRV 特徴量専用ライン】(放物線補間 ➔ Kubios外れ値補正 ➔ スライディング窓4大指標算出)
4. 結果を hrv/result/ 配下に CSV および グラフ画像として保存
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

# ディレクトリパス設定 (rppg, validation)
_RPPG_EVAL_DIR = Path(__file__).resolve().parent.parent
_VALIDATION_DIR = _RPPG_EVAL_DIR.parent
for _p in [_VALIDATION_DIR, _RPPG_EVAL_DIR]:
  if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

# hrv ディレクトリ自身のパス
_HRV_DIR = Path(__file__).resolve().parent
if str(_HRV_DIR) not in sys.path:
  sys.path.insert(0, str(_HRV_DIR))

from config_loader import load_config, log
from face_landmarker_extractor import FaceLandmarkerRGBExtractor
from hrv_preprocessor import HRVPreprocessor
from plotter_hrv import plot_combined_hrv, plot_individual_hrv


def load_ground_truth(
    gt_path: Path | str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """ground_truth.txt を読み込み、(ppg_signal, hr, timesteps) を返します。"""
  gt_path = Path(gt_path)
  if not gt_path.exists():
    raise FileNotFoundError(f"ground_truth.txt が見つかりません: {gt_path}")
  with open(gt_path, "r", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]
  if len(lines) < 3:
    raise ValueError(f"ground_truth.txt の行数が不足しています: {len(lines)}")
  ppg = np.array([float(x) for x in lines[0].split()], dtype=float)
  hr = np.array([float(x) for x in lines[1].split()], dtype=float)
  timesteps = np.array([float(x) for x in lines[2].split()], dtype=float)
  return ppg, hr, timesteps


def run_hrv_evaluation(
    config_path: Path | str, cli_overrides: Optional[dict] = None
):
  # 共通 config_loader による設定読み込み、CLI反映、パス解決
  cfg = load_config(config_path, cli_overrides=cli_overrides)
  paths_cfg = cfg.get("paths", {})
  dataset_dir = paths_cfg["dataset_dir"]
  output_dir = paths_cfg["output_dir"]
  model_path = paths_cfg.get("model_path")

  # Extraction 設定
  ext_cfg = cfg.get("extraction", {})
  target_fps = float(ext_cfg.get("target_fps", 30.0))
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

  # HRV 設定
  hrv_cfg = cfg.get("hrv", {})
  hrv_enabled = hrv_cfg.get("enabled", True)
  hrv_window_sec = float(hrv_cfg.get("window_sec", 45.0))
  hrv_stride_sec = float(hrv_cfg.get("stride_sec", 1.0))
  save_csv = hrv_cfg.get("save_csv", True)

  # 可視化設定
  vis_cfg = cfg.get("visualization", {})
  save_plots = vis_cfg.get("save_plots", True)
  combined_plot_filename = vis_cfg.get(
      "combined_plot_filename", "hrv_comparison.png"
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

  print("=" * 80)
  print(" remote-PPG (rPPG) 【HRV 特徴量専用ライン】評価・抽出パイプライン")
  print("=" * 80)
  print(f"📄 設定ファイル: {config_path}")
  print(f"📁 データセット: {dataset_dir}")
  print(f"💾 結果出力先: {output_dir}")
  print(f"🎯 抽出サンプリングレート: {target_fps} fps (ROIサイズ: {region_size}px)")
  print(f"⚙️  前処理: fs={fs}Hz, POS窓長={pos_window_sec}s, 帯域=[{lowcut}, {highcut}]Hz")
  print(
      f"🫀 HRV スライディング窓: {hrv_window_sec}秒 (ストライド: {hrv_stride_sec}秒, 補正: Kubios)"
  )
  print(f"👥 対象被験者数: {len(subject_dirs)} 件")
  print("=" * 80)

  preprocessor = HRVPreprocessor(
      fs=fs,
      input_fps=input_fps,
      pos_window_sec=pos_window_sec,
      lowcut=lowcut,
      highcut=highcut,
      filter_order=filter_order,
      roi_cols=roi_cols,
      hrv_window_sec=hrv_window_sec,
      hrv_stride_sec=hrv_stride_sec,
  )

  hrv_summary_records = []
  all_hrv_data = []

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

      t_start = time.time()

      # 1. Ground Truth のタイムスタンプ取得 (同期用)
      gt_timesteps = None
      if gt_path.exists():
        try:
          _, _, gt_timesteps = load_ground_truth(gt_path)
        except Exception:
          gt_timesteps = None

      # 2. FaceLandmarker による RGB 抽出
      try:
        print(f"  ▶ Step 1: FaceLandmarker 適用 & {target_fps}fps RGB抽出中...")
        df_rgb = extractor.extract_rgb_from_video(
            avi_path, gt_timestamps=gt_timesteps
        )
        print(f"    └─ 抽出完了: {len(df_rgb)} サンプル")
      except Exception as e:
        print(f"  ❌ RGB抽出失敗: {e}")
        continue

      # 3. preprocess と全く同じコードによる前処理 & HRV 特徴量抽出
      try:
        print(f"  ▶ Step 2: preprocess 準拠 HRV 特徴量専用ライン実行中...")
        df_hr, selected_roi, snr_db, df_hrv = preprocessor.process(df_rgb)
        if df_hrv is None or df_hrv.empty:
          print("  ⚠️ HRV 特徴量テーブルが空です。スキップします。")
          continue

        mean_rmssd = float(df_hrv["rmssd"].dropna().mean())
        mean_sdnn = float(df_hrv["sdnn"].dropna().mean())
        mean_sd2_sd1 = float(df_hrv["sd2_sd1"].dropna().mean())
        mean_pp = float(df_hrv["mean_pp"].dropna().mean())
        n_windows = len(df_hrv)

        elapsed = time.time() - t_start
        print(f"  ✨ {s_name} 完了 ({elapsed:.1f} 秒)")
        print(f"    ├─ 採択ROI: {selected_roi} (SNR: {snr_db:.2f} dB)")
        print(f"    ├─ スライディング窓数: {n_windows} 窓")
        print(f"    ├─ 平均 RMSSD: {mean_rmssd:.2f} ms")
        print(f"    ├─ 平均 SDNN: {mean_sdnn:.2f} ms")
        print(f"    ├─ 平均 SD2/SD1: {mean_sd2_sd1:.2f}")
        print(f"    └─ 平均 Mean PP: {mean_pp:.2f} ms")

        # CSV 保存 (hrv_{subject}.csv)
        if save_csv:
          hrv_csv_path = output_dir / f"hrv_{s_name}.csv"
          df_hrv.to_csv(hrv_csv_path, index=False)
          print(f"    💾 HRVテーブル保存: {hrv_csv_path.name}")

        # 個別プロット画像保存
        if save_plots and save_individual_plots:
          plot_path = output_dir / f"plot_hrv_{s_name}.png"
          plot_individual_hrv(s_name, df_hrv, plot_path, dpi=dpi)
          print(f"    📈 個別グラフ保存: {plot_path.name}")

        hrv_summary_records.append({
            "subject": s_name,
            "selected_roi": selected_roi,
            "snr_db": round(snr_db, 2),
            "n_windows": n_windows,
            "mean_rmssd": round(mean_rmssd, 2),
            "mean_sdnn": round(mean_sdnn, 2),
            "mean_sd2_sd1": round(mean_sd2_sd1, 2),
            "mean_pp": round(mean_pp, 2),
        })

        all_hrv_data.append({
            "subject": s_name,
            "df_hrv": df_hrv,
        })

      except Exception as e:
        print(f"  ❌ HRV抽出失敗: {e}")
        continue

  if not hrv_summary_records:
    print("\n❌ 有効な HRV 解析結果が得られませんでした。")
    return

  # サマリー CSV 保存
  df_summary = pd.DataFrame(hrv_summary_records)
  summary_csv_path = output_dir / "hrv_summary.csv"
  df_summary.to_csv(summary_csv_path, index=False)

  # 全体統合比較プロット保存
  if save_plots and all_hrv_data:
    combined_plot_path = output_dir / combined_plot_filename
    plot_combined_hrv(all_hrv_data, combined_plot_path, dpi=dpi)

  print("\n" + "=" * 80)
  print("                     rPPG HRV 特徴量解析 総合サマリー")
  print("=" * 80)
  print(
      f"{'Subject':<10} | {'ROI':<6} | {'SNR(dB)':<7} | {'Windows':<7} |"
      f" {'RMSSD(ms)':<10} | {'SDNN(ms)':<9} | {'SD2/SD1':<7} | {'Mean PP(ms)':<10}"
  )
  print("-" * 80)
  for _, row in df_summary.iterrows():
    print(
        f"{row['subject']:<10} | {row['selected_roi']:<6} |"
        f" {row['snr_db']:>7.2f} | {row['n_windows']:>7} |"
        f" {row['mean_rmssd']:>10.2f} | {row['mean_sdnn']:>9.2f} |"
        f" {row['mean_sd2_sd1']:>7.2f} | {row['mean_pp']:>10.2f}"
    )
  print("-" * 80)
  print(
      f"{'MEAN':<10} | {'-':<6} | {df_summary['snr_db'].mean():>7.2f} |"
      f" {int(round(df_summary['n_windows'].mean())):>7} |"
      f" {df_summary['mean_rmssd'].mean():>10.2f} |"
      f" {df_summary['mean_sdnn'].mean():>9.2f} |"
      f" {df_summary['mean_sd2_sd1'].mean():>7.2f} |"
      f" {df_summary['mean_pp'].mean():>10.2f}"
  )
  print("=" * 80)
  print(f"\n📊 HRVサマリーCSV: {summary_csv_path}")
  if save_plots:
    print(f"📈 統合比較グラフ: {output_dir / combined_plot_filename}")
  print(f"📁 結果保存ディレクトリ: {output_dir}")


def main():
  parser = argparse.ArgumentParser(
      description="rppg_datasets に対する preprocess 準拠 HRV 特徴量専用ライン評価・抽出"
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
      "--hrv_window_sec",
      type=float,
      default=None,
      help="HRVスライディング窓長 (秒) (config.yaml を上書き)",
  )
  parser.add_argument(
      "--hrv_stride_sec",
      type=float,
      default=None,
      help="HRVスライディング窓ストライド (秒) (config.yaml を上書き)",
  )
  args = parser.parse_args()

  cli_overrides = {
      "dataset_dir": args.dataset_dir,
      "output_dir": args.output_dir,
      "subjects": args.subject,
      "hrv_window_sec": args.hrv_window_sec,
      "hrv_stride_sec": args.hrv_stride_sec,
  }

  run_hrv_evaluation(config_path=Path(args.config), cli_overrides=cli_overrides)


if __name__ == "__main__":
  main()
