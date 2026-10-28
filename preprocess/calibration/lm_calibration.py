"""
lm_calibration.py

顔ランドマーク (LM) 特徴量に対する外部マスター基準 (calibration_master.csv) キャリブレーションモジュール。
被験者の安静時正面注視データ（基準値）に基づき、比率 (ear_ratio) および差分 (*_diff) を算出します。
"""

from pathlib import Path
import sys
from typing import Any
import numpy as np
import pandas as pd

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

try:
    from config_loader import log
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("lm_calibration")


def load_calibration_master(master_path: str | Path) -> pd.DataFrame:
    """キャリブレーションマスターCSVを読み込み、subject_id をインデックスにしたDataFrameを返す"""
    p = Path(master_path)
    if not p.exists():
        raise FileNotFoundError(f"キャリブレーションマスターが見つかりません: {p}")
    master_df = pd.read_csv(p)
    if "subject_id" in master_df.columns:
        master_df["subject_id"] = master_df["subject_id"].astype(str)
        master_df.set_index("subject_id", inplace=True)
    elif "username" in master_df.columns:
        master_df["username"] = master_df["username"].astype(str)
        master_df.set_index("username", inplace=True)
    return master_df


def calibrate_landmarks(df: pd.DataFrame, calib_row: pd.Series | dict, cfg: dict = None) -> pd.DataFrame:
    """
    キャリブレーションマスター値を用いたランドマーク正規化（比率 & 差分）
    - EAR: ear_ratio = ear_avg / ear_calib
    - Head Pose: *_diff = * - *_calib
    - Gaze: *_diff = * - *_calib
    """
    res_df = df.copy()
    c = cfg or {}
    lm_cfg = c.get("landmarks", {})
    calib_cfg = lm_cfg.get("calibration", {})

    ratio_targets = calib_cfg.get("ratio_targets", {"ear_avg": "ear_calib"})
    diff_targets = calib_cfg.get("diff_targets", {
        "head_pitch": "head_pitch_calib",
        "head_yaw": "head_yaw_calib",
        "head_roll": "head_roll_calib",
        "gaze_v_deg": "gaze_pitch_calib",
        "gaze_h_deg": "gaze_yaw_calib",
    })

    # 1. 比率 (Ratio)
    for col, calib_col in ratio_targets.items():
        ratio_col = f"{col.replace('_avg', '')}_ratio"
        if col in res_df.columns:
            val = calib_row.get(calib_col) if isinstance(calib_row, (dict, pd.Series)) else None
            if val is not None and not pd.isna(val) and val != 0:
                res_df[ratio_col] = res_df[col] / val
            else:
                res_df[ratio_col] = res_df[col]  # キャリブレーション値がない場合は等倍

    # 2. 視線角のフォールバック候補の適用
    gaze_fallback = {
        "gaze_v_deg": ["gaze_v_deg_calib", "gaze_pitch_calib"],
        "gaze_h_deg": ["gaze_h_deg_calib", "gaze_yaw_calib"],
    }

    # 3. 差分 (Diff)
    for col, calib_col in diff_targets.items():
        diff_col = f"{col}_diff"
        if col in res_df.columns:
            val = calib_row.get(calib_col) if isinstance(calib_row, (dict, pd.Series)) else None
            # 視線角の別名フォールバック
            if (val is None or pd.isna(val)) and col in gaze_fallback:
                for cand in gaze_fallback[col]:
                    cand_val = calib_row.get(cand) if isinstance(calib_row, (dict, pd.Series)) else None
                    if cand_val is not None and not pd.isna(cand_val):
                        val = cand_val
                        break
            if val is not None and not pd.isna(val):
                res_df[diff_col] = res_df[col] - val
            else:
                res_df[diff_col] = res_df[col] - 0.0  # 欠落時は0基準フォールバック

    return res_df


def apply_calibration(df: pd.DataFrame, calib_data: pd.Series | dict, cfg: dict = None) -> pd.DataFrame:
    """calibrate_landmarks へのエイリアス（後方互換性用）"""
    return calibrate_landmarks(df, calib_data, cfg)
