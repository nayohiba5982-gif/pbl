"""
hr_extraction.py
[ 最適ROIのBVP波形 (bvp_clean) ]
                       │
                       ▼
            [ 共通: ピーク粗検出 ]
       （NeuroKit2 等でピークインデックスを取得）
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
  【 HR 推定ライン 】         【 HRV 特徴量専用ライン 】
  ① ピーク間隔から瞬時BPM      (hrv_extraction.py に委譲)
  ② 5秒移動平均（平滑化）      ① サブサンプル補間 (放物線近似)
  ③ 出力: 時系列 BPM データ    ② IBI外れ値の検出と補正
                              ③ スライディング窓 (45〜60秒)
                              ④ 4大指標算出 (rmssd, sdnn, sd2/sd1, mean_pp)
                              ⑤ 出力: 機械学習用 特徴量テーブル
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import neurokit2 as nk
import numpy as np
import pandas as pd

# filtering ディレクトリからの moving_average インポート
_PREPROCESS_DIR = Path(__file__).resolve().parent.parent.parent
if str(_PREPROCESS_DIR) not in sys.path:
  sys.path.insert(0, str(_PREPROCESS_DIR))

try:
  from features.rppg.roi_selector import select_best_bvp
except ImportError:
  from roi_selector import select_best_bvp

try:
  from filtering.moving_average import apply_bpm_moving_average
except ImportError:
  from moving_average import apply_bpm_moving_average

try:
  from features.rppg.hrv_extraction import process_hrv_pipeline
except ImportError:
  from hrv_extraction import process_hrv_pipeline


def process_hr_extraction(
    df: pd.DataFrame,
    roi_cols: dict,
    fs: float = 50.0,
    ma_window_sec: float = 5.0,
    ma_enabled: bool = True,
    ma_center: bool = False,
    compute_hrv: bool = False,
    hrv_window_sec: float = 60.0,
    hrv_stride_sec: float = 1.0,
) -> Union[
    Tuple[pd.DataFrame, str, float],
    Tuple[pd.DataFrame, str, float, Optional[pd.DataFrame]],
]:
  """最適BVPの採択、共通ピーク粗検出、および

  【HR推定ライン】(瞬時BPM ➔ 5秒移動平均)
  【HRV特徴量専用ライン】(放物線補間 ➔ IBI外れ値除去 ➔ 4大指標算出) を実行します。

  Parameters:
      df (pd.DataFrame): フィルタ後BVPを含むDataFrame
      roi_cols (dict): ROI列マッピング
      fs (float): サンプリング周波数 (Hz)
      ma_window_sec (float): 移動平均窓長 (秒, デフォルト: 5.0)
      ma_enabled (bool): 移動平均を適用するか (デフォルト: True)
      ma_center (bool): 中心窓フラグ (デフォルト: False)
      compute_hrv (bool): HRV 4大指標テーブルも同時に算出するか

  Returns:
      compute_hrv=False の場合: (df_hr, selected_region, selected_snr)
      compute_hrv=True の場合: (df_hr, selected_region, selected_snr,
      df_hrv_table)
  """
  df_out = df.copy()

  # 1. 最適な BVP の選択
  best_bvp, selected_region, selected_snr = select_best_bvp(
      df_out, roi_cols, fs=fs
  )

  df_out["bvp_selected"] = best_bvp
  df_out["selected_roi"] = selected_region
  df_out["snr_db"] = selected_snr

  # 欠損値補間
  bvp_clean = (
      pd.Series(best_bvp).interpolate(method="linear").bfill().ffill().values
  )

  # 2. [ 共通: ピーク粗検出 ]
  rough_peak_indices = np.array([], dtype=int)
  try:
    signals, info = nk.ppg_process(bvp_clean, sampling_rate=int(round(fs)))
    raw_bpm = signals["PPG_Rate"].values
    ppg_peaks = signals["PPG_Peaks"].values
    if "PPG_Peaks" in info and len(info["PPG_Peaks"]) > 0:
      rough_peak_indices = np.asarray(info["PPG_Peaks"], dtype=int)
    else:
      rough_peak_indices = np.where(ppg_peaks == 1)[0]
  except Exception:
    raw_bpm = np.full(len(df_out), np.nan)
    ppg_peaks = np.zeros(len(df_out), dtype=int)

  # 3. 【 HR 推定ライン 】
  # ① ピーク間隔から瞬時BPM
  df_out["bpm_raw"] = raw_bpm
  df_out["bpm"] = raw_bpm
  df_out["ppg_peaks"] = ppg_peaks

  # ② 5秒移動平均（平滑化）
  if ma_enabled and ma_window_sec > 0:
    df_out = apply_bpm_moving_average(
        df_out,
        target_col="bpm",
        fs=fs,
        window_sec=ma_window_sec,
        center=ma_center,
        backup_raw=True,
    )

  # 4. 【 HRV 特徴量専用ライン 】
  df_hrv_table = None
  if compute_hrv:
    if len(rough_peak_indices) >= 4:
      try:
        df_hrv_table, _, _ = process_hrv_pipeline(
            bvp_signal=bvp_clean,
            rough_peak_indices=rough_peak_indices,
            fs=fs,
            window_sec=hrv_window_sec,
            stride_sec=hrv_stride_sec,
        )
      except Exception as e:
        print(f"  ⚠️ HRV特徴量算出失敗: {e}")
        df_hrv_table = None

    return df_out, selected_region, selected_snr, df_hrv_table

  return df_out, selected_region, selected_snr
