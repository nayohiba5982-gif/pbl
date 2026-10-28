import numpy as np
import pandas as pd
try:
    from calibration.feature_calculator import FeatureCalculator
except ImportError:
    from features.lm.feature_calculator import FeatureCalculator

def extract_lm_features_dataframe(
    df: pd.DataFrame, 
    r_eye: float = 11.7
) -> pd.DataFrame:
    """
    DataFrame の各行（フレーム）に対し、FeatureCalculator を用いて
    EAR, まばたき判定, 頭部姿勢(Yaw/Roll/Pitch), 眼球回転角(H/V) を計算します。
    """
    df_out = df.copy()
    calc = FeatureCalculator(r_eye=r_eye)

    ear_left_list, ear_right_list, ear_avg_list = [], [], []
    blink_list = []
    yaw_list, roll_list, pitch_list = [], [], []
    gaze_h_list, gaze_v_list = [], []

    # 各行（フレーム）に対して計算 (to_dict('records') で高速化)
    records = df_out.to_dict('records')
    for row_dict in records:
        
        try:
            # 1. EAR (瞬時値) & まばたき判定
            l_ear, r_ear, avg_ear = calc.calculate_instant_ear(row_dict)
            is_blink = calc.is_blinking(row_dict)
            
            # 2. 頭部姿勢 (Head Pose)
            yaw, roll, pitch = calc.calculate_head_pose(row_dict)
            
            # 3. 視線・眼球回転角 (Eye Rotation)
            gaze_h, gaze_v = calc.calculate_eye_rotation(row_dict)

        except Exception:
            # 顔検出不可などの欠損フレーム対策
            l_ear, r_ear, avg_ear = np.nan, np.nan, np.nan
            is_blink = False
            yaw, roll, pitch = np.nan, np.nan, np.nan
            gaze_h, gaze_v = np.nan, np.nan

        ear_left_list.append(l_ear)
        ear_right_list.append(r_ear)
        ear_avg_list.append(avg_ear)
        blink_list.append(is_blink)
        
        yaw_list.append(yaw)
        roll_list.append(roll)
        pitch_list.append(pitch)
        
        gaze_h_list.append(gaze_h)
        gaze_v_list.append(gaze_v)

    # DataFrame へのカラム追加
    df_out['ear_left'] = ear_left_list
    df_out['ear_right'] = ear_right_list
    df_out['ear_avg'] = ear_avg_list
    df_out['is_blink'] = blink_list

    df_out['head_yaw'] = yaw_list
    df_out['head_roll'] = roll_list
    df_out['head_pitch'] = pitch_list

    df_out['gaze_h_deg'] = gaze_h_list
    df_out['gaze_v_deg'] = gaze_v_list

    return df_out