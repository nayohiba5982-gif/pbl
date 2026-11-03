"""
hrv_extraction.py
【HRV 特徴量専用ライン】
1. サブサンプル補間 (放物線近似 / Parabolic Interpolation) によるピーク時刻のミリ秒精度補正
2. IBI 外れ値の検出と補正 (NeuroKit2 signal_fixpeaks 等)
3. スライディング窓 (45〜60秒) による時系列分割
4. 4大指標の算出 (rmssd, sdnn, sd2/sd1, mean_pp) by NeuroKit2
5. 機械学習用 特徴量テーブルの生成
"""

from typing import Dict, List, Optional, Tuple, Union
import warnings
import neurokit2 as nk
import numpy as np
import pandas as pd

try:
  from neurokit2.hrv.hrv_nonlinear import _hrv_nonlinear_poincare
except ImportError:
  _hrv_nonlinear_poincare = None

HRV_METRIC_COLS = ["rmssd", "sdnn", "sd2_sd1", "mean_pp"]


def refine_peaks_parabolic(
    peaks: np.ndarray,
    signal: np.ndarray,
    fs: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
  """ピーク粗検出で得られた離散インデックスに対し、周辺3点を用いた放物線フィッティング (二次関数補間) を行い、

  サブサンプル精度のピーク位置 (サンプル単位) およびピーク時刻 (秒・ミリ秒) を算出します。

  頂点オフセット:
      delta = - (y[p+1] - y[p-1]) / (2 * (y[p+1] - 2*y[p] + y[p-1]))
  """
  peaks = np.asarray(peaks, dtype=int)
  N = len(signal)
  peaks_refined_samples = []

  for p in peaks:
    if 1 <= p < N - 1:
      y_prev = signal[p - 1]
      y_curr = signal[p]
      y_next = signal[p + 1]

      denom = 2.0 * (y_next - 2.0 * y_curr + y_prev)
      # 上に凸 (denom < 0) かつ ゼロ除算回避
      if abs(denom) > 1e-8 and denom < 0:
        delta = -(y_next - y_prev) / denom
        # 物理的に隣接サンプル幅 [-1, 1] に制限
        delta = np.clip(delta, -1.0, 1.0)
        p_sub = p + delta
      else:
        p_sub = float(p)
    else:
      p_sub = float(p)

    peaks_refined_samples.append(p_sub)

  peaks_refined_samples = np.array(peaks_refined_samples, dtype=float)
  peaks_refined_sec = peaks_refined_samples / fs

  return peaks_refined_samples, peaks_refined_sec


def correct_ibi_outliers(
    peaks_samples: np.ndarray,
    fs: float = 50.0,
    method: str = "Kubios",
) -> np.ndarray:
  """サブサンプル精度ピーク位置列に対し、NeuroKit2 の signal_fixpeaks を用いて

  異所性拍や誤検出ピーク (IBI外れ値) を検出・補正します。
  """
  if len(peaks_samples) < 5:
    return peaks_samples

  try:
    res = nk.signal_fixpeaks(
        peaks=peaks_samples, sampling_rate=int(round(fs)), method=method
    )
    if isinstance(res, tuple):
      if isinstance(res[0], np.ndarray):
        fixed_peaks = res[0]
      elif len(res) > 1 and isinstance(res[1], np.ndarray):
        fixed_peaks = res[1]
      else:
        fixed_peaks = peaks_samples
    elif isinstance(res, np.ndarray):
      fixed_peaks = res
    else:
      fixed_peaks = peaks_samples
  except Exception:
    fixed_peaks = peaks_samples

  # 生理学的に不自然な間隔 (< 0.25s ≒ > 240 BPM, > 2.5s ≒ < 24 BPM) を最小限ガード
  diffs = np.diff(fixed_peaks) / fs
  invalid = (diffs < 0.25) | (diffs > 2.5)
  if np.any(invalid):
    # 急激に近すぎる多重ピークは後ろ側を除外
    valid_idx = [0]
    for i in range(1, len(fixed_peaks)):
      dt = (fixed_peaks[i] - fixed_peaks[valid_idx[-1]]) / fs
      if dt >= 0.25:
        valid_idx.append(i)
    fixed_peaks = fixed_peaks[valid_idx]

  return fixed_peaks


def compute_hrv_4metrics(
    peaks_sec: np.ndarray,
    fs: float = 50.0,
) -> Dict[str, float]:
  """指定されたピーク時刻系列 (秒) から NeuroKit2 を用いて 4大 HRV 指標

  (mean_pp, rmssd, sdnn, sd2/sd1) を算出します。
  """
  if len(peaks_sec) < 4:
    return {
        "mean_pp": np.nan,
        "rmssd": np.nan,
        "sdnn": np.nan,
        "sd2_sd1": np.nan,
        "n_peaks": len(peaks_sec),
    }

  # ピーク位置 (サンプル単位)
  peaks_samples = peaks_sec * fs

  try:
    hrv_time = nk.hrv_time(peaks_samples, sampling_rate=int(round(fs)))
    mean_pp = (
        float(hrv_time["HRV_MeanNN"].iloc[0])
        if "HRV_MeanNN" in hrv_time
        else np.nan
    )
    rmssd = (
        float(hrv_time["HRV_RMSSD"].iloc[0])
        if "HRV_RMSSD" in hrv_time
        else np.nan
    )
    sdnn = (
        float(hrv_time["HRV_SDNN"].iloc[0])
        if "HRV_SDNN" in hrv_time
        else np.nan
    )
  except Exception:
    ibi_ms = np.diff(peaks_sec) * 1000.0
    mean_pp = float(np.mean(ibi_ms)) if len(ibi_ms) > 0 else np.nan
    rmssd = (
        float(np.sqrt(np.mean(np.diff(ibi_ms) ** 2)))
        if len(ibi_ms) > 1
        else np.nan
    )
    sdnn = float(np.std(ibi_ms, ddof=1)) if len(ibi_ms) > 1 else np.nan

  # 非線形解析 (SD2 / SD1: Poincaré plot)
  sd2_sd1 = np.nan
  ibi_ms = np.diff(peaks_sec) * 1000.0
  if len(ibi_ms) >= 3:
    try:
      if _hrv_nonlinear_poincare is not None:
        poincare_out = _hrv_nonlinear_poincare(ibi_ms, out={})
        sd1 = float(poincare_out.get("SD1", np.nan))
        sd2 = float(poincare_out.get("SD2", np.nan))
      else:
        rri_n = ibi_ms[:-1]
        rri_plus = ibi_ms[1:]
        sd1 = float(np.std((rri_n - rri_plus) / np.sqrt(2.0), ddof=1))
        sd2 = float(np.std((rri_n + rri_plus) / np.sqrt(2.0), ddof=1))
      if sd1 > 1e-6 and not np.isnan(sd2):
        sd2_sd1 = float(sd2 / sd1)
    except Exception:
      pass

  return {
      "mean_pp": round(mean_pp, 2) if not np.isnan(mean_pp) else np.nan,
      "rmssd": round(rmssd, 2) if not np.isnan(rmssd) else np.nan,
      "sdnn": round(sdnn, 2) if not np.isnan(sdnn) else np.nan,
      "sd2_sd1": round(sd2_sd1, 3) if not np.isnan(sd2_sd1) else np.nan,
      "n_peaks": int(len(peaks_sec)),
  }


def extract_hrv_features_sliding_window(
    peaks_sec: np.ndarray,
    total_duration_sec: float,
    fs: float = 50.0,
    window_sec: float = 60.0,
    stride_sec: float = 1.0,
    min_peaks_per_window: int = 10,
) -> pd.DataFrame:
  """45〜60秒のスライディング窓を回し、各窓で4大指標を算出して特徴量テーブルを生成します。

  Parameters:
      peaks_sec (np.ndarray): サブサンプル補正・外れ値除去済みのピーク時刻 (秒)
      total_duration_sec (float): 信号の総秒数
      fs (float): サンプリング周波数 (Hz)
      window_sec (float): ウィンドウ幅 (秒, 例: 45.0 〜 60.0)
      stride_sec (float): スライド幅 (秒, 例: 1.0)
      min_peaks_per_window (int): ウィンドウ内の最小必要ピーク数

  Returns:
      pd.DataFrame: ウィンドウごとの HRV 4大指標テーブル
  """
  peaks_sec = np.sort(np.asarray(peaks_sec, dtype=float))
  records = []

  if total_duration_sec < window_sec:
    # 録画全体が窓幅より短い場合は録画全体で1レコード算出
    metrics = compute_hrv_4metrics(peaks_sec, fs=fs)
    records.append({
        "window_start": 0.0,
        "window_end": round(total_duration_sec, 2),
        "window_center": round(total_duration_sec / 2.0, 2),
        **metrics,
    })
    return pd.DataFrame(records)

  # スライディング窓ループ
  start_times = np.arange(0.0, total_duration_sec - window_sec + 1e-5, stride_sec)

  for w_start in start_times:
    w_end = w_start + window_sec
    w_center = w_start + window_sec / 2.0

    # 窓内のピークを抽出
    mask = (peaks_sec >= w_start) & (peaks_sec <= w_end)
    w_peaks = peaks_sec[mask]

    if len(w_peaks) >= min_peaks_per_window:
      # 相対時刻にして指標計算
      metrics = compute_hrv_4metrics(w_peaks - w_start, fs=fs)
    else:
      metrics = {
          "mean_pp": np.nan,
          "rmssd": np.nan,
          "sdnn": np.nan,
          "sd2_sd1": np.nan,
          "n_peaks": int(len(w_peaks)),
      }

    records.append({
        "window_start": round(float(w_start), 2),
        "window_end": round(float(w_end), 2),
        "window_center": round(float(w_center), 2),
        **metrics,
    })

  return pd.DataFrame(records)


def process_hrv_pipeline(
    bvp_signal: np.ndarray,
    rough_peak_indices: np.ndarray,
    fs: float = 50.0,
    window_sec: float = 60.0,
    stride_sec: float = 1.0,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
  """【HRV 特徴量専用ライン】全体統括関数

  1. サブサンプル補間 (放物線近似)
  2. IBI外れ値の検出と補正
  3. スライディング窓 (45〜60秒) による4大指標の算出
  """
  total_duration_sec = len(bvp_signal) / fs

  # 1. サブサンプル補間
  peaks_sub_samples, peaks_sub_sec = refine_peaks_parabolic(
      rough_peak_indices, bvp_signal, fs=fs
  )

  # 2. IBI外れ値の検出と補正
  peaks_clean_samples = correct_ibi_outliers(
      peaks_sub_samples, fs=fs, method="Kubios"
  )
  peaks_clean_sec = peaks_clean_samples / fs

  # 3. スライディング窓による4大指標テーブル生成
  df_hrv_windows = extract_hrv_features_sliding_window(
      peaks_clean_sec,
      total_duration_sec=total_duration_sec,
      fs=fs,
      window_sec=window_sec,
      stride_sec=stride_sec,
  )

  return df_hrv_windows, peaks_clean_samples, peaks_clean_sec


def align_hrv_to_timeseries(
    df: pd.DataFrame,
    df_hrv_windows: Optional[pd.DataFrame],
    fs: float = 50.0,
    time_col: str = "time",
) -> pd.DataFrame:
  """スライディング窓で算出された HRV 4大指標テーブル (df_hrv_windows) を、
  window_center を基準として元の 50Hz 時系列 DataFrame (df) にタイムスタンプ同期・補間して結合します。
  """
  df_out = df.copy()
  n_rows = len(df_out)
  if n_rows == 0:
    for col in HRV_METRIC_COLS:
      df_out[col] = np.array([], dtype=np.float32)
    return df_out

  if time_col in df_out.columns:
    t_vals = df_out[time_col].values.astype(float)
    rel_time = t_vals - t_vals[0]
  else:
    rel_time = np.arange(n_rows, dtype=float) / fs

  if df_hrv_windows is None or df_hrv_windows.empty or "window_center" not in df_hrv_windows.columns:
    for col in HRV_METRIC_COLS:
      df_out[col] = 0.0
    return df_out

  centers = df_hrv_windows["window_center"].values.astype(float)

  for col in HRV_METRIC_COLS:
    if col not in df_hrv_windows.columns:
      df_out[col] = 0.0
      continue
    s = (
        pd.Series(df_hrv_windows[col].values, dtype=float)
        .interpolate(method="linear")
        .bfill()
        .ffill()
        .fillna(0.0)
        .values
    )
    if len(centers) == 1:
      df_out[col] = float(s[0])
    else:
      df_out[col] = np.interp(rel_time, centers, s).astype(np.float32)

  return df_out


def ensure_hrv_in_dataframe(
    df: pd.DataFrame,
    fs: float = 50.0,
    window_sec: float = 45.0,
    stride_sec: float = 1.0,
    time_col: str = "time",
) -> pd.DataFrame:
  """DataFrame に HRV 4大指標 (rmssd, sdnn, sd2_sd1, mean_pp) が含まれていない場合、
  bvp_selected および ppg_peaks から 45秒窓で算出してタイムスタンプ同期結合します。
  """
  if all(col in df.columns and not df[col].isna().all() for col in HRV_METRIC_COLS):
    return df

  if "bvp_selected" not in df.columns or len(df) == 0:
    df_out = df.copy()
    for col in HRV_METRIC_COLS:
      if col not in df_out.columns:
        df_out[col] = 0.0
    return df_out

  bvp_clean = (
      pd.Series(df["bvp_selected"].values, dtype=float)
      .interpolate(method="linear")
      .bfill()
      .ffill()
      .fillna(0.0)
      .values
  )

  rough_peak_indices = np.array([], dtype=int)
  if "ppg_peaks" in df.columns:
    rough_peak_indices = np.where(df["ppg_peaks"].values == 1)[0]

  if len(rough_peak_indices) < 4:
    try:
      signals, info = nk.ppg_process(bvp_clean, sampling_rate=int(round(fs)))
      if "PPG_Peaks" in info and len(info["PPG_Peaks"]) > 0:
        rough_peak_indices = np.asarray(info["PPG_Peaks"], dtype=int)
      else:
        rough_peak_indices = np.where(signals["PPG_Peaks"].values == 1)[0]
    except Exception:
      rough_peak_indices = np.array([], dtype=int)

  df_hrv_table = None
  if len(rough_peak_indices) >= 4:
    try:
      df_hrv_table, _, _ = process_hrv_pipeline(
          bvp_signal=bvp_clean,
          rough_peak_indices=rough_peak_indices,
          fs=fs,
          window_sec=window_sec,
          stride_sec=stride_sec,
      )
    except Exception:
      df_hrv_table = None

  return align_hrv_to_timeseries(df, df_hrv_table, fs=fs, time_col=time_col)

