"""
hrv_preprocessor.py
preprocess パッケージの rPPG 前処理コードと全く同じアルゴリズムを適用し、
【HRV 特徴量専用ライン】を実行するモジュール
  Step 3: 欠損値・外れ値除去 (clean_rppg_outliers, fps=25)
  Step 4: 50Hz PCHIP リサンプリング (resample_rgb_dataframe, target_fps=50.0)
  Step 5: POS法による脈波抽出 (extract_pos_bvp_dataframe, window_sec=1.6)
  Step 6: バンドパスフィルタ (process_filtering_dataframe, 0.75-2.5Hz, Butterworth order=2)
  Step 7: 最適ROI選択 & 共通ピーク粗検出 (process_hr_extraction)
  HRV 専用ライン:
    ① サブサンプル補間 (放物線近似によるミリ秒精度補正)
    ② IBI 外れ値・異所性拍の検出と補正 (Kubios 法)
    ③ スライディング窓 (45〜60秒) による時系列分割
    ④ NeuroKit2 による 4大指標算出 (rmssd, sdnn, sd2/sd1, mean_pp)
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

# preprocess パッケージのインポートパス設定
_PREPROCESS_DIR = Path(__file__).resolve().parent.parent.parent / "preprocess"
if str(_PREPROCESS_DIR) not in sys.path:
  sys.path.insert(0, str(_PREPROCESS_DIR))

try:
  from cleaning.rgb_cleaner import clean_rppg_outliers
  from features.rppg.hr_extraction import process_hr_extraction
  from features.rppg.hrv_extraction import process_hrv_pipeline
  from features.rppg.pos_algorithm import extract_pos_bvp_dataframe
  from filtering.bandpass_filter import process_filtering_dataframe
  from resampling.rgb_resampler import resample_rgb_dataframe
except ImportError as e:
  raise ImportError(f"preprocess モジュールの読み込みに失敗しました: {e}")


class HRVPreprocessor:
  """preprocess のパイプラインと全く同一のコードで前処理を行い、
  HRV 特徴量専用ラインを実行するクラス
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
      hrv_window_sec: float = 45.0,
      hrv_stride_sec: float = 1.0,
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

    self.hrv_window_sec = hrv_window_sec
    self.hrv_stride_sec = hrv_stride_sec

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

  def process(
      self, df_rgb: pd.DataFrame
  ) -> Tuple[pd.DataFrame, str, float, Optional[pd.DataFrame]]:
    """Step 3〜7 を順次実行し、(df_hr, selected_roi, snr, df_hrv) を返します。"""
    # Step 3
    df_clean = self.clean(df_rgb)
    # Step 4
    df_res = self.resample(df_clean)
    # Step 5
    df_pos = self.extract_pos(df_res)
    # Step 6
    df_filt = self.filter_bvp(df_pos)
    # Step 7 & HRV専用ライン (preprocessと全く同じ process_hr_extraction を使用)
    res = process_hr_extraction(
        df_filt,
        roi_cols=self.roi_cols,
        fs=self.fs,
        compute_hrv=True,
        hrv_window_sec=self.hrv_window_sec,
        hrv_stride_sec=self.hrv_stride_sec,
    )
    df_hr, selected_roi, snr, df_hrv = res
    return df_hr, selected_roi, snr, df_hrv
