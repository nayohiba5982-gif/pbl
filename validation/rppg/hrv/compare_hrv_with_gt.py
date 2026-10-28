"""
compare_hrv_with_gt.py
各 subject の ground_truth.txt に記録された接触型 PPG 真値データから
NeuroKit2 を用いて算出した HRV 指標 (GT HRV) と、
preprocess パイプラインで採用している rPPG 由来の HRV 指標 (rPPG HRV) との
差異 (MAE, RMSE, Bias, 相関係数 r, MAPE) を詳細に定量評価・比較するモジュール
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import yaml

# パス設定
_RPPG_EVAL_DIR = Path(__file__).resolve().parent.parent
_VALIDATION_DIR = _RPPG_EVAL_DIR.parent
for _p in [_VALIDATION_DIR, _RPPG_EVAL_DIR]:
  if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

from config_loader import load_config, log

_PREPROCESS_DIR = _RPPG_EVAL_DIR.parent / "preprocess"
if str(_PREPROCESS_DIR) not in sys.path:
  sys.path.insert(0, str(_PREPROCESS_DIR))

try:
  import neurokit2 as nk
  from features.rppg.hrv_extraction import extract_hrv_features_sliding_window
except ImportError as e:
  raise ImportError(f"必要なモジュールの読み込みに失敗しました: {e}")

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


def extract_gt_hrv_sliding_window(
    gt_ppg: np.ndarray,
    gt_timesteps: np.ndarray,
    fs: float = 50.0,
    window_sec: float = 45.0,
    stride_sec: float = 1.0,
) -> Tuple[pd.DataFrame, np.ndarray]:
  """接触型 PPG 真値信号から NeuroKit2 でピークを検出し、

  preprocess と同一のスライディング窓設定で HRV 4大指標テーブルを算出します。
  """
  # 1. 重複タイムスタンプの除去
  _, unique_idx = np.unique(gt_timesteps, return_index=True)
  t_clean = gt_timesteps[unique_idx]
  ppg_clean = gt_ppg[unique_idx]

  # 2. 50Hz PCHIP リサンプリング
  total_duration = t_clean[-1] - t_clean[0]
  t_uniform = np.arange(t_clean[0], t_clean[-1], 1.0 / fs)
  pchip = PchipInterpolator(t_clean, ppg_clean)
  ppg_uniform = pchip(t_uniform)

  # 3. NeuroKit2 によるピーク検出
  signals, info = nk.ppg_process(ppg_uniform, sampling_rate=int(round(fs)))
  gt_peaks_idx = np.asarray(info["PPG_Peaks"], dtype=int)
  gt_peaks_sec = t_uniform[gt_peaks_idx]

  # 4. preprocess と同一の extract_hrv_features_sliding_window で指標算出
  df_gt_hrv = extract_hrv_features_sliding_window(
      peaks_sec=gt_peaks_sec,
      total_duration_sec=total_duration,
      fs=fs,
      window_sec=window_sec,
      stride_sec=stride_sec,
  )

  return df_gt_hrv, gt_peaks_sec


def compute_hrv_comparison_metrics(
    df_pred_hrv: pd.DataFrame,
    df_gt_hrv: pd.DataFrame,
    metrics_keys: List[str] = ["mean_pp", "rmssd", "sdnn", "sd2_sd1"],
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
  """予測 HRV と GT HRV を window_center で同期し、各指標の誤差を計算します。"""
  # window_center でマージ
  df_merged = pd.merge(
      df_pred_hrv,
      df_gt_hrv,
      on="window_center",
      suffixes=("_pred", "_gt"),
      how="inner",
  )

  results = {}

  for m in metrics_keys:
    col_pred = f"{m}_pred"
    col_gt = f"{m}_gt"

    if col_pred not in df_merged or col_gt not in df_merged:
      continue

    # 有効ペア抽出
    valid_mask = ~df_merged[col_pred].isna() & ~df_merged[col_gt].isna()
    y_pred = df_merged.loc[valid_mask, col_pred].values
    y_gt = df_merged.loc[valid_mask, col_gt].values

    if len(y_pred) == 0:
      results[m] = {
          "mae": np.nan,
          "rmse": np.nan,
          "mean_error": np.nan,
          "std_error": np.nan,
          "pearson_r": np.nan,
          "mape": np.nan,
          "gt_mean": np.nan,
          "pred_mean": np.nan,
          "n_samples": 0,
      }
      continue

    errors = y_pred - y_gt
    abs_errors = np.abs(errors)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    mean_err = float(np.mean(errors))
    std_err = float(np.std(errors))

    # 相関係数
    if len(y_pred) >= 2 and np.std(y_pred) > 1e-6 and np.std(y_gt) > 1e-6:
      r = float(np.corrcoef(y_pred, y_gt)[0, 1])
    else:
      r = np.nan

    # MAPE (%)
    gt_nonzero = y_gt != 0
    if np.any(gt_nonzero):
      mape = float(
          np.mean(abs_errors[gt_nonzero] / np.abs(y_gt[gt_nonzero])) * 100.0
      )
    else:
      mape = np.nan

    results[m] = {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mean_error": round(mean_err, 2),
        "std_error": round(std_err, 2),
        "pearson_r": round(r, 3) if not np.isnan(r) else np.nan,
        "mape": round(mape, 2) if not np.isnan(mape) else np.nan,
        "gt_mean": round(float(np.mean(y_gt)), 2),
        "pred_mean": round(float(np.mean(y_pred)), 2),
        "n_samples": int(len(y_pred)),
    }

  return results, df_merged


def plot_hrv_gt_comparison_subject(
    subject: str,
    df_merged: pd.DataFrame,
    metrics_summary: Dict[str, Dict[str, float]],
    save_path: Path,
    dpi: int = 300,
):
  """GT vs rPPG の HRV 4大指標時系列比較プロットを描画・保存します。"""
  save_path.parent.mkdir(parents=True, exist_ok=True)
  fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
  fig.suptitle(
      f"HRV Ground Truth vs rPPG Estimation Comparison - {subject}",
      fontsize=14,
      fontweight="bold",
  )

  times = df_merged["window_center"].values
  metrics_info = [
      ("mean_pp", "Mean PP Interval (ms)", "#d62728"),
      ("rmssd", "RMSSD (ms)", "#1f77b4"),
      ("sdnn", "SDNN (ms)", "#ff7f0e"),
      ("sd2_sd1", "SD2 / SD1 Ratio", "#2ca02c"),
  ]

  for ax, (m, label, color) in zip(axes, metrics_info):
    pred_col = f"{m}_pred"
    gt_col = f"{m}_gt"

    m_stats = metrics_summary.get(m, {})
    mae = m_stats.get("mae", np.nan)
    r = m_stats.get("pearson_r", np.nan)

    ax.plot(
        times,
        df_merged[gt_col],
        "k--",
        label=f"Ground Truth (Contact PPG, mean={m_stats.get('gt_mean', np.nan)})",
        lw=2,
        alpha=0.8,
    )
    ax.plot(
        times,
        df_merged[pred_col],
        "-o",
        color=color,
        label=(
            f"rPPG Estimated (preprocess, mean={m_stats.get('pred_mean', np.nan)})"
        ),
        lw=2,
        markersize=4,
    )

    r_text = f", r: {r:.2f}" if not np.isnan(r) else ""
    ax.set_title(
        f"{label} [MAE: {mae:.2f}{r_text}]",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylabel(label, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

  axes[-1].set_xlabel("Time (seconds) [Window Center]", fontsize=11)
  plt.tight_layout()
  plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
  plt.close(fig)


def plot_overall_hrv_error_bars(
    df_overall: pd.DataFrame,
    save_path: Path,
    dpi: int = 300,
):
  """全被験者の指標別 MAE & MAPE を比較する総合グラフ"""
  save_path.parent.mkdir(parents=True, exist_ok=True)
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
  fig.suptitle(
      "Overall HRV Estimation Errors (rPPG vs Ground Truth)",
      fontsize=14,
      fontweight="bold",
  )

  # 指標ごとの MAE バープロット (mean_pp, rmssd, sdnn)
  time_metrics = ["mean_pp", "rmssd", "sdnn"]
  subjects = df_overall["subject"].unique()
  x = np.arange(len(subjects))
  width = 0.25

  for i, m in enumerate(time_metrics):
    maes = [
        df_overall.loc[
            (df_overall["subject"] == s) & (df_overall["metric"] == m), "mae"
        ].values[0]
        for s in subjects
    ]
    ax1.bar(x + (i - 1) * width, maes, width, label=m.upper())

  ax1.set_title("Absolute Error (MAE in ms) by Metric", fontweight="bold")
  ax1.set_ylabel("MAE (ms)")
  ax1.set_xticks(x)
  ax1.set_xticklabels(subjects)
  ax1.grid(True, axis="y", alpha=0.3)
  ax1.legend()

  # MAPE (%) バープロット (全4指標)
  all_metrics = ["mean_pp", "rmssd", "sdnn", "sd2_sd1"]
  width_4 = 0.2
  for i, m in enumerate(all_metrics):
    mapes = [
        df_overall.loc[
            (df_overall["subject"] == s) & (df_overall["metric"] == m), "mape"
        ].values[0]
        for s in subjects
    ]
    ax2.bar(x + (i - 1.5) * width_4, mapes, width_4, label=m.upper())

  ax2.set_title("Relative Error (MAPE in %) by Metric", fontweight="bold")
  ax2.set_ylabel("MAPE (%)")
  ax2.set_xticks(x)
  ax2.set_xticklabels(subjects)
  ax2.grid(True, axis="y", alpha=0.3)
  ax2.legend()

  plt.tight_layout()
  plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
  plt.close(fig)


def run_comparison(config_path: Path | str):
  # 共通 config_loader による設定読み込み & パス解決
  cfg = load_config(config_path)
  paths_cfg = cfg.get("paths", {})
  dataset_dir = paths_cfg["dataset_dir"]
  result_dir = paths_cfg["output_dir"]

  hrv_cfg = cfg.get("hrv", {})
  window_sec = float(hrv_cfg.get("window_sec", 45.0))
  stride_sec = float(hrv_cfg.get("stride_sec", 1.0))

  # 出力先
  comp_out_dir = result_dir / "gt_comparison"
  comp_out_dir.mkdir(parents=True, exist_ok=True)

  subject_dirs = sorted(
      [d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
  )

  print("=" * 85)
  print(" remote-PPG (preprocess) vs Ground Truth (NeuroKit2) HRV 指標 精度・差異検証")
  print("=" * 85)
  print(f"📁 データセット: {dataset_dir}")
  print(f"📁 既存rPPG結果: {result_dir}")
  print(f"💾 比較出力先: {comp_out_dir}")
  print(f"⏱️ スライディング窓長: {window_sec} 秒 (ストライド: {stride_sec} 秒)")
  print("=" * 85)

  overall_records = []

  for s_dir in subject_dirs:
    s_name = s_dir.name
    gt_path = s_dir / "ground_truth.txt"
    rppg_hrv_path = result_dir / f"hrv_{s_name}.csv"

    print(f"\n▶ 被験者: {s_name} の検証中...")

    if not gt_path.exists():
      print(f"  ⚠️ ground_truth.txt が見つかりません: {gt_path}")
      continue
    if not rppg_hrv_path.exists():
      print(f"  ⚠️ rPPG HRV ファイルが見つかりません: {rppg_hrv_path}")
      continue

    # 1. GT 読み込み
    try:
      gt_ppg, _, gt_timesteps = load_ground_truth(gt_path)
    except Exception as e:
      print(f"  ❌ GT読み込み失敗: {e}")
      continue

    # 2. GT から NeuroKit2 でスライディング窓 HRV 算出
    try:
      df_gt_hrv, gt_peaks_sec = extract_gt_hrv_sliding_window(
          gt_ppg,
          gt_timesteps,
          fs=50.0,
          window_sec=window_sec,
          stride_sec=stride_sec,
      )
      # GT HRV CSV も保存
      gt_hrv_csv = comp_out_dir / f"gt_hrv_{s_name}.csv"
      df_gt_hrv.to_csv(gt_hrv_csv, index=False)
      print(
          f"  ✓ GT ピーク検出完了 ({len(gt_peaks_sec)} 拍, {len(df_gt_hrv)} 窓)"
      )
    except Exception as e:
      print(f"  ❌ GT HRV 算出失敗: {e}")
      continue

    # 3. rPPG HRV 読み込み
    df_rppg_hrv = pd.read_csv(rppg_hrv_path)

    # 4. 誤差・相関の計算
    metrics_summary, df_merged = compute_hrv_comparison_metrics(
        df_pred_hrv=df_rppg_hrv,
        df_gt_hrv=df_gt_hrv,
    )

    # 同期比較 CSV 保存
    comp_csv = comp_out_dir / f"comparison_{s_name}.csv"
    df_merged.to_csv(comp_csv, index=False)

    # 個別比較プロット画像保存
    plot_path = comp_out_dir / f"plot_gt_comp_{s_name}.png"
    plot_hrv_gt_comparison_subject(
        s_name, df_merged, metrics_summary, plot_path
    )
    print(f"  📈 比較プロット保存: {plot_path.name}")

    # コンソール表示
    print(f"  ┌─ {'Metric':<10} | {'GT Mean':<9} | {'rPPG':<9} | {'MAE':<7} | {'RMSE':<7} | {'Bias':<7} | {'MAPE(%)':<8} | {'r':<6}")
    print(f"  ├─" + "-" * 75)
    for m in ["mean_pp", "rmssd", "sdnn", "sd2_sd1"]:
      s = metrics_summary.get(m, {})
      print(
          f"  │  {m:<10} | {s.get('gt_mean', 0):>9.2f} |"
          f" {s.get('pred_mean', 0):>9.2f} | {s.get('mae', 0):>7.2f} |"
          f" {s.get('rmse', 0):>7.2f} | {s.get('mean_error', 0):>7.2f} |"
          f" {s.get('mape', 0):>7.1f}% | {s.get('pearson_r', 0):>6.3f}"
      )
      overall_records.append({
          "subject": s_name,
          "metric": m,
          **s,
      })
    print(f"  └─" + "-" * 75)

  if not overall_records:
    print("❌ 有効な比較結果が得られませんでした。")
    return

  df_all = pd.DataFrame(overall_records)
  summary_csv = comp_out_dir / "hrv_gt_comparison_summary.csv"
  df_all.to_csv(summary_csv, index=False)

  # 全体誤差グラフ
  overall_plot = comp_out_dir / "hrv_gt_overall_errors.png"
  plot_overall_hrv_error_bars(df_all, overall_plot)

  print("\n" + "=" * 90)
  print("                  全被験者 HRV 精度・誤差総合サマリー (rPPG vs GT)")
  print("=" * 90)
  print(
      f"{'Metric':<12} | {'GT Mean':<10} | {'rPPG Mean':<10} | {'Mean MAE':<10} |"
      f" {'Mean RMSE':<10} | {'Mean MAPE':<11} | {'Mean Corr(r)':<12}"
  )
  print("-" * 90)
  for m in ["mean_pp", "rmssd", "sdnn", "sd2_sd1"]:
    sub_df = df_all[df_all["metric"] == m]
    print(
        f"{m:<12} | {sub_df['gt_mean'].mean():>10.2f} |"
        f" {sub_df['pred_mean'].mean():>10.2f} | {sub_df['mae'].mean():>10.2f}"
        f" | {sub_df['rmse'].mean():>10.2f} |"
        f" {sub_df['mape'].mean():>10.1f}% | {sub_df['pearson_r'].mean():>12.3f}"
    )
  print("=" * 90)
  print(f"\n📊 比較サマリーCSV: {summary_csv}")
  print(f"📈 総合誤差プロット: {overall_plot}")
  print(f"📁 各種データ保存先: {comp_out_dir}")


def main():
  parser = argparse.ArgumentParser(
      description="ground_truth.txt から算出した真値 HRV と rPPG HRV の差異検証"
  )
  parser.add_argument(
      "--config",
      type=str,
      default=str(Path(__file__).resolve().parent / "config.yaml"),
      help="hrv/config.yaml のパス",
  )
  args = parser.parse_args()
  run_comparison(config_path=args.config)


if __name__ == "__main__":
  main()
