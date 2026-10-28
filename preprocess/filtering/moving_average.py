"""
moving_average.py
rPPG前処理により出力された心拍数 (BPM) 系列に対して移動平均 (Moving Average) を適用するモジュール
"""

from typing import Optional, Union
import numpy as np
import pandas as pd


def apply_moving_average(
    signal: Union[pd.Series, np.ndarray],
    fs: float = 50.0,
    window_sec: float = 5.0,
    min_periods: int = 1,
    center: bool = False,
) -> np.ndarray:
  """1次元信号 (BPM時系列等) に対し、指定した秒数の移動平均を適用します。

  Parameters:
      signal (pd.Series | np.ndarray): 入力信号
      fs (float): サンプリング周波数 (Hz, デフォルト: 50.0)
      window_sec (float): 移動平均のウィンドウ長 (秒, デフォルト: 5.0)
      min_periods (int): 窓内の最小必要サンプル数 (デフォルト: 1)
      center (bool): 窓を中心に置くか (False: 過去窓, True: 前後対称)

  Returns:
      np.ndarray: 移動平均適用後の信号配列
  """
  if window_sec <= 0:
    return np.asarray(signal, dtype=float)

  window_samples = max(1, int(round(window_sec * fs)))
  s = pd.Series(signal)

  smoothed = s.rolling(
      window=window_samples,
      min_periods=min_periods,
      center=center,
  ).mean()

  return smoothed.values


def apply_bpm_moving_average(
    df: pd.DataFrame,
    target_col: str = "bpm",
    fs: float = 50.0,
    window_sec: float = 5.0,
    center: bool = False,
    backup_raw: bool = True,
) -> pd.DataFrame:
  """DataFrame 内の心拍数列 (bpm) に対し移動平均を適用します。

  Parameters:
      df (pd.DataFrame): 対象 DataFrame
      target_col (str): 心拍数列名 (デフォルト: "bpm")
      fs (float): サンプリング周波数 (Hz)
      window_sec (float): ウィンドウ幅 (秒)
      center (bool): 中心窓フラグ
      backup_raw (bool): 元の生BPMを 'bpm_raw' 列に退避するか (デフォルト: True)

  Returns:
      pd.DataFrame: 平滑化された心拍数列を含む DataFrame
  """
  df_out = df.copy()
  if target_col not in df_out.columns:
    return df_out

  if backup_raw and "bpm_raw" not in df_out.columns:
    df_out["bpm_raw"] = df_out[target_col].copy()

  if window_sec > 0:
    df_out[target_col] = apply_moving_average(
        df_out[target_col].values,
        fs=fs,
        window_sec=window_sec,
        min_periods=1,
        center=center,
    )

  return df_out
