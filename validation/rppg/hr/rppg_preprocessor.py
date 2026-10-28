"""
rppg_preprocessor.py
preprocess ディレクトリの rPPG 前処理と全く同じアルゴリズム・パラメータを適用し、
BPM出力後に指定秒数 (デフォルト: 5.0秒) の移動平均を適用するモジュール
  Step 3: 欠損値・外れ値除去 (clean_rppg_outliers, fps=25)
  Step 4: 50Hz PCHIP リサンプリング (resample_rgb_dataframe, target_fps=50.0)
  Step 5: POS法による脈波抽出 (extract_pos_bvp_dataframe, window_sec=1.6)
  Step 6: バンドパスフィルタ (process_filtering_dataframe, 0.75-2.5Hz, Butterworth order=2)
  Step 7: 最適ROI選択 & 心拍数・ピーク抽出 (process_hr_extraction, NeuroKit2 ppg_process)
  Post-Process: BPM出力後の移動平均 (moving_average.py)
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

# preprocess パッケージからのインポート
_PREPROCESS_DIR = Path(__file__).resolve().parent.parent / "preprocess"
if not _PREPROCESS_DIR.exists():
  _PREPROCESS_DIR = Path(__file__).resolve().parent.parent.parent / "preprocess"
if str(_PREPROCESS_DIR) not in sys.path:
  sys.path.insert(0, str(_PREPROCESS_DIR))

try:
  from cleaning.rgb_cleaner import clean_rppg_outliers
  from features.rppg.hr_extraction import process_hr_extraction
  from features.rppg.pos_algorithm import extract_pos_bvp_dataframe
  from filtering.bandpass_filter import process_filtering_dataframe
  from resampling.rgb_resampler import resample_rgb_dataframe
except ImportError as e:
  raise ImportError(f"preprocess モジュールの読み込みに失敗しました: {e}")

# 移動平均モジュール (rppg_evaluation 配下)
from moving_average import apply_bpm_moving_average


class RPPGPreprocessor:
  """preprocess のパイプライン (Step 3〜7) と全く同じ処理を実行し、

  BPM出力後に移動平均を適用するクラス
  """

  def __init__(
      self,
      fs: float = 50.0,
      input_fps: int = 25,
      pos_window_sec: float = 1.6,
      lowcut: float = 0.75,
      highcut: float = 2.5,
      filter_order: int = 2,
      roi_regions: Optional[Dict[str, str]] = None,
      roi_cols: Optional[Dict[str, str]] = None,
      ma_window_sec: float = 5.0,
      ma_enabled: bool = True,
      ma_center: bool = False,
  ):
    self.fs = fs
    self.input_fps = input_fps
    self.pos_window_sec = pos_window_sec
    self.lowcut = lowcut
    self.highcut = highcut
    self.filter_order = filter_order

    self.roi_regions = roi_regions or {
        "both": "rppg_bothCheeks",
        "left": "rppg_leftCheek",
        "right": "rppg_rightCheek",
    }
    self.roi_cols = roi_cols or {
        "both": "bvp_clean_both",
        "left": "bvp_clean_left",
        "right": "bvp_clean_right",
    }

    # 移動平均パラメータ
    self.ma_window_sec = ma_window_sec
    self.ma_enabled = ma_enabled
    self.ma_center = ma_center

  def clean(self, df_rgb: pd.DataFrame) -> pd.DataFrame:
    """Step 3: 欠損値・外れ値除去 (LM同期なし、fps=25)"""
    return clean_rppg_outliers(df_rgb, df_lm=None, fps=self.input_fps)

  def resample(self, df_cleaned: pd.DataFrame) -> pd.DataFrame:
    """Step 4: リサンプリング (50Hz PCHIP補間)"""
    return resample_rgb_dataframe(
        df_cleaned, target_fps=self.fs, time_col="time"
    )

  def extract_pos(self, df_resampled: pd.DataFrame) -> pd.DataFrame:
    """Step 5: POS Methodによる脈波(BVP)抽出"""
    return extract_pos_bvp_dataframe(
        df_resampled,
        region_prefix=self.roi_regions,
        fs=self.fs,
        window_sec=self.pos_window_sec,
    )

  def filter_bvp(self, df_pos: pd.DataFrame) -> pd.DataFrame:
    """Step 6: 脈波バンドパスフィルタリング (0.75 - 2.5Hz Butterworth)"""
    return process_filtering_dataframe(
        df_pos,
        target_col="bvp_raw",
        fs=self.fs,
        lowcut=self.lowcut,
        highcut=self.highcut,
        order=self.filter_order,
    )

  def extract_hr(self, df_filtered: pd.DataFrame) -> Tuple[pd.DataFrame, str, float]:
    """Step 7: 生体指標算出 (最適ROI選択, BPM, PPGピーク, SNR)"""
    return process_hr_extraction(
        df_filtered, roi_cols=self.roi_cols, fs=self.fs
    )

  def process(self, df_rgb: pd.DataFrame) -> Tuple[pd.DataFrame, str, float]:
    """Step 3〜7 を順次実行し、BPM出力後に移動平均を適用した DataFrame を返します。"""
    # Step 3
    df_clean = self.clean(df_rgb)
    # Step 4
    df_res = self.resample(df_clean)
    # Step 5
    df_pos = self.extract_pos(df_res)
    # Step 6
    df_filt = self.filter_bvp(df_pos)
    # Step 7 (本来のpreprocessのBPM出力)
    df_hr, selected_roi, snr = self.extract_hr(df_filt)

    # BPM出力後に移動平均を適用
    if self.ma_enabled and self.ma_window_sec > 0:
      df_hr = apply_bpm_moving_average(
          df_hr,
          target_col="bpm",
          fs=self.fs,
          window_sec=self.ma_window_sec,
          center=self.ma_center,
          backup_raw=True,
      )

    return df_hr, selected_roi, snr
