"""
plotter.py
推定心拍数 (POS Pred) と真値心拍数 (Ground Truth) の比較グラフを描画・保存するモジュール
"""

from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_hr_comparison(
    results_data: List[Dict],
    output_path: Path | str,
    dpi: int = 300,
):
  """全被験者の Ground Truth と推定心拍数 (BPM) の時系列比較グラフ (縦並びサブプロット) を作成・保存します。

  Code_Generated_Image.png のスタイルを忠実に再現します。
  """
  n_subjects = len(results_data)
  if n_subjects == 0:
    return

  fig, axes = plt.subplots(
      n_subjects, 1, figsize=(10, 2.4 * n_subjects), sharex=False
  )
  if n_subjects == 1:
    axes = [axes]

  for ax, item in zip(axes, results_data):
    s_name = item["subject"]
    mae = item.get("mae", 0.0)
    df_comp = item["df_comparison"]

    time_sec = df_comp["time"].values
    bpm_pred = df_comp["bpm_pred"].values
    hr_true = df_comp["hr_true"].values

    # Subject 番号の表示名整形 (subject1 -> Subject 1)
    display_name = s_name.replace("subject", "Subject ")

    # プロット (黒: GT, 赤/マゼンタ系: POS Pred)
    ax.plot(
        time_sec,
        hr_true,
        color="#111111",
        label="Ground Truth (BPM)",
        linewidth=1.3,
    )
    ax.plot(
        time_sec,
        bpm_pred,
        color="#e6194b",
        label="POS Pred (BPM)",
        linewidth=1.0,
        alpha=0.9,
    )

    ax.set_title(f"{display_name} (MAE: {mae:.2f})", fontsize=10, pad=4)
    ax.set_ylabel("BPM", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    ax.tick_params(labelsize=9)

  # 最下段のプロットにのみ X 軸ラベルを表示
  axes[-1].set_xlabel("Time (s)", fontsize=10)

  plt.tight_layout()
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
  plt.close(fig)
  print(f"  📈 比較グラフを保存しました: {output_path}")


def plot_individual_subject(
    item: Dict,
    output_path: Path | str,
    dpi: int = 300,
):
  """個別の被験者の比較グラフを作成・保存します。"""
  s_name = item["subject"]
  mae = item.get("mae", 0.0)
  rmse = item.get("rmse", 0.0)
  df_comp = item["df_comparison"]

  time_sec = df_comp["time"].values
  bpm_pred = df_comp["bpm_pred"].values
  hr_true = df_comp["hr_true"].values

  fig, ax = plt.subplots(figsize=(10, 3.2))
  display_name = s_name.replace("subject", "Subject ")

  ax.plot(
      time_sec,
      hr_true,
      color="#111111",
      label="Ground Truth (BPM)",
      linewidth=1.4,
  )
  ax.plot(
      time_sec,
      bpm_pred,
      color="#e6194b",
      label="POS Pred (BPM)",
      linewidth=1.1,
      alpha=0.9,
  )

  ax.set_title(
      f"{display_name} - Heart Rate Estimation (MAE: {mae:.2f} BPM, RMSE:"
      f" {rmse:.2f} BPM)",
      fontsize=11,
  )
  ax.set_xlabel("Time (s)", fontsize=10)
  ax.set_ylabel("Heart Rate (BPM)", fontsize=10)
  ax.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
  ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

  plt.tight_layout()
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
  plt.close(fig)
