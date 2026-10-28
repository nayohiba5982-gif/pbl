"""
metrics_evaluator.py
ground_truth.txt のパースおよび心拍数 (HR/BPM) の誤差計算・評価指標算出モジュール
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


def load_ground_truth(gt_path: Path | str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """ground_truth.txt を読み込み、(ppg_signal, hr, timesteps) を返します。

  Line 1: PPG signal
  Line 2: Heart rate (HR)
  Line 3: Timestep (seconds)
  """
  gt_path = Path(gt_path)
  if not gt_path.exists():
    raise FileNotFoundError(f"ground_truth.txt が見つかりません: {gt_path}")

  with open(gt_path, "r", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]

  if len(lines) < 3:
    raise ValueError(
        f"ground_truth.txt に必要な3行のデータが含まれていません (行数: {len(lines)})"
    )

  ppg = np.array([float(x) for x in lines[0].split()], dtype=float)
  hr = np.array([float(x) for x in lines[1].split()], dtype=float)
  timesteps = np.array([float(x) for x in lines[2].split()], dtype=float)

  return ppg, hr, timesteps


def calculate_hr_errors(
    df_hr: pd.DataFrame,
    gt_hr: np.ndarray,
    gt_timesteps: np.ndarray,
    subject_name: str = "",
    selected_roi: str = "",
    snr_db: float = 0.0,
) -> Dict[str, float | str]:
  """推定心拍数 (df_hr['bpm']) と GT (gt_hr) を時刻同期し、各種評価指標を算出します。"""
  est_time = df_hr["time"].values
  est_bpm = df_hr["bpm"].values

  # GT を推定時刻グリッドに線形補間
  gt_hr_synced = np.interp(est_time, gt_timesteps, gt_hr)

  # 有効サンプル（NaN除外）
  valid_mask = (
      ~np.isnan(est_bpm)
      & ~np.isnan(gt_hr_synced)
      & (est_time >= gt_timesteps[0])
      & (est_time <= gt_timesteps[-1])
  )
  n_valid = int(np.sum(valid_mask))

  if n_valid < 10:
    return {
        "subject": subject_name,
        "selected_roi": selected_roi,
        "snr_db": round(snr_db, 2),
        "gt_hr_mean": (
            round(float(np.nanmean(gt_hr)), 2)
            if len(gt_hr) > 0
            else float("nan")
        ),
        "est_bpm_mean": (
            round(float(np.nanmean(est_bpm)), 2)
            if len(est_bpm) > 0
            else float("nan")
        ),
        "mae": float("nan"),
        "rmse": float("nan"),
        "mean_error": float("nan"),
        "std_error": float("nan"),
        "pearson_r": float("nan"),
        "valid_samples": n_valid,
    }

  y_true = gt_hr_synced[valid_mask]
  y_pred = est_bpm[valid_mask]

  errors = y_pred - y_true
  mae = float(np.mean(np.abs(errors)))
  rmse = float(np.sqrt(np.mean(errors**2)))
  mean_error = float(np.mean(errors))
  std_error = float(np.std(errors))

  # Pearson 相関係数
  if np.std(y_pred) > 1e-6 and np.std(y_true) > 1e-6:
    corr = float(np.corrcoef(y_pred, y_true)[0, 1])
  else:
    corr = 0.0

  return {
      "subject": subject_name,
      "selected_roi": selected_roi,
      "snr_db": round(snr_db, 2),
      "gt_hr_mean": round(float(np.mean(y_true)), 2),
      "est_bpm_mean": round(float(np.mean(y_pred)), 2),
      "mae": round(mae, 2),
      "rmse": round(rmse, 2),
      "mean_error": round(mean_error, 2),
      "std_error": round(std_error, 2),
      "pearson_r": round(corr, 3),
      "valid_samples": n_valid,
  }
