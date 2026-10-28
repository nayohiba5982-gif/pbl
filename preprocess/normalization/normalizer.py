import pandas as pd
import numpy as np


def calibrate_landmarks(df: pd.DataFrame, calib_row: pd.Series | dict, cfg: dict) -> pd.DataFrame:
    """
    Step 8: キャリブレーションマスター値を用いた正規化（比率 & 差分）
    - EAR: ear_ratio = ear_avg / ear_calib
    - Head Pose: *_diff = * - *_calib
    - Gaze: *_diff = * - *_calib
    """
    res_df = df.copy()
    lm_cfg = cfg.get("landmarks", {})
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


class LandmarkFeaturePipeline:
    """辞書設定(cfg)を保持し、キャリブレーション正規化を行うクラス"""
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def transform(self, df: pd.DataFrame, calib_row: pd.Series | dict) -> pd.DataFrame:
        """Step 8: キャリブレーション正規化を実行"""
        return calibrate_landmarks(df, calib_row, self.cfg)
