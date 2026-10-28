"""
plotter_hrv.py
HRV (心拍変動) 解析結果の可視化モジュール
- 個別被験者の HRV 4大指標推移プロット
- ポアンカレプロット (Poincaré Plot: RR_n vs RR_{n+1})
- 全被験者統合 HRV 比較プロット
"""

from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_individual_hrv(
    subject_name: str,
    df_hrv: pd.DataFrame,
    save_path: Path | str,
    dpi: int = 300,
):
  """個別被験者の HRV 4大指標 (RMSSD, SDNN, SD2/SD1, Mean PP) の時系列推移をプロット"""
  save_path = Path(save_path)
  save_path.parent.mkdir(parents=True, exist_ok=True)

  fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
  fig.suptitle(
      f"HRV 4-Metrics Sliding Window Analysis - {subject_name}",
      fontsize=14,
      fontweight="bold",
  )

  times = df_hrv["window_center"].values

  # 1. RMSSD
  axes[0].plot(
      times, df_hrv["rmssd"], color="#1f77b4", lw=2, marker="o", markersize=3
  )
  axes[0].set_ylabel("RMSSD (ms)", fontsize=11, fontweight="semibold")
  axes[0].grid(True, alpha=0.3)
  axes[0].set_title(
      f"Mean RMSSD: {df_hrv['rmssd'].mean():.2f} ms",
      loc="right",
      fontsize=10,
      color="#1f77b4",
  )

  # 2. SDNN
  axes[1].plot(
      times, df_hrv["sdnn"], color="#ff7f0e", lw=2, marker="s", markersize=3
  )
  axes[1].set_ylabel("SDNN (ms)", fontsize=11, fontweight="semibold")
  axes[1].grid(True, alpha=0.3)
  axes[1].set_title(
      f"Mean SDNN: {df_hrv['sdnn'].mean():.2f} ms",
      loc="right",
      fontsize=10,
      color="#ff7f0e",
  )

  # 3. SD2 / SD1
  axes[2].plot(
      times, df_hrv["sd2_sd1"], color="#2ca02c", lw=2, marker="^", markersize=3
  )
  axes[2].set_ylabel("SD2 / SD1", fontsize=11, fontweight="semibold")
  axes[2].grid(True, alpha=0.3)
  axes[2].set_title(
      f"Mean SD2/SD1: {df_hrv['sd2_sd1'].mean():.2f}",
      loc="right",
      fontsize=10,
      color="#2ca02c",
  )

  # 4. Mean PP
  axes[3].plot(
      times, df_hrv["mean_pp"], color="#d62728", lw=2, marker="d", markersize=3
  )
  axes[3].set_ylabel("Mean PP (ms)", fontsize=11, fontweight="semibold")
  axes[3].set_xlabel("Time (s) [Window Center]", fontsize=11, fontweight="semibold")
  axes[3].grid(True, alpha=0.3)
  axes[3].set_title(
      f"Mean PP: {df_hrv['mean_pp'].mean():.2f} ms",
      loc="right",
      fontsize=10,
      color="#d62728",
  )

  plt.tight_layout()
  plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
  plt.close(fig)


def plot_combined_hrv(
    hrv_records: List[Dict],
    save_path: Path | str,
    dpi: int = 300,
):
  """全被験者の HRV 比較プロット (バープロット & 推移)"""
  save_path = Path(save_path)
  save_path.parent.mkdir(parents=True, exist_ok=True)

  subjects = [r["subject"] for r in hrv_records]
  mean_rmssd = [r["df_hrv"]["rmssd"].mean() for r in hrv_records]
  mean_sdnn = [r["df_hrv"]["sdnn"].mean() for r in hrv_records]
  mean_sd2_sd1 = [r["df_hrv"]["sd2_sd1"].mean() for r in hrv_records]
  mean_pp = [r["df_hrv"]["mean_pp"].mean() for r in hrv_records]

  x = np.arange(len(subjects))
  width = 0.35

  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
  fig.suptitle(
      "remote-PPG HRV 4-Metrics Overall Comparison Across Subjects",
      fontsize=14,
      fontweight="bold",
  )

  # 1. RMSSD
  bars1 = ax1.bar(x, mean_rmssd, width, color="#1f77b4", edgecolor="black")
  ax1.set_title("Mean RMSSD (Parasympathetic Activity)", fontweight="bold")
  ax1.set_ylabel("RMSSD (ms)")
  ax1.set_xticks(x)
  ax1.set_xticklabels(subjects, rotation=15)
  ax1.grid(True, axis="y", alpha=0.3)
  ax1.bar_label(bars1, fmt="%.1f", padding=3)

  # 2. SDNN
  bars2 = ax2.bar(x, mean_sdnn, width, color="#ff7f0e", edgecolor="black")
  ax2.set_title("Mean SDNN (Total Autonomic Variability)", fontweight="bold")
  ax2.set_ylabel("SDNN (ms)")
  ax2.set_xticks(x)
  ax2.set_xticklabels(subjects, rotation=15)
  ax2.grid(True, axis="y", alpha=0.3)
  ax2.bar_label(bars2, fmt="%.1f", padding=3)

  # 3. SD2 / SD1
  bars3 = ax3.bar(x, mean_sd2_sd1, width, color="#2ca02c", edgecolor="black")
  ax3.set_title("Mean SD2 / SD1 (Sympatho-Vagal Balance)", fontweight="bold")
  ax3.set_ylabel("SD2 / SD1 Ratio")
  ax3.set_xticks(x)
  ax3.set_xticklabels(subjects, rotation=15)
  ax3.grid(True, axis="y", alpha=0.3)
  ax3.bar_label(bars3, fmt="%.2f", padding=3)

  # 4. Mean PP
  bars4 = ax4.bar(x, mean_pp, width, color="#d62728", edgecolor="black")
  ax4.set_title("Mean PP Interval (NN Interval)", fontweight="bold")
  ax4.set_ylabel("Mean PP (ms)")
  ax4.set_xticks(x)
  ax4.set_xticklabels(subjects, rotation=15)
  ax4.grid(True, axis="y", alpha=0.3)
  ax4.bar_label(bars4, fmt="%.1f", padding=3)

  plt.tight_layout()
  plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
  plt.close(fig)
