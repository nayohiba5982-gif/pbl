import numpy as np
import pandas as pd

def clean_landmark_outliers(
    df: pd.DataFrame, 
    fps: int = 30, 
    scale_threshold: float = 0.20, 
    speed_threshold: float = 0.10, 
    geo_std_threshold: float = 3.0, 
    interp_limit: int = 6
) -> pd.DataFrame:
    """
    3つのアプローチ（スケール・幾何構造・移動速度）を用いて異常値を検知・NaN化し、線形補完を行います。
    """
    df_clean = df.copy()
    
    # クレンジング対象となる列（lm_, centroid_, calc_ 関連）
    lm_cols = [c for c in df.columns if c.startswith('lm_') or c.startswith('centroid_') or c.startswith('calc_')]
    
    # ─── 1. 「顔のスケール（目間距離 D_eye）」によるチェック ───
    lx, ly, lz = df['lm_gaze_473_x'], df['lm_gaze_473_y'], df['lm_gaze_473_z']
    rx, ry, rz = df['lm_gaze_468_x'], df['lm_gaze_468_y'], df['lm_gaze_468_z']
    
    d_eye = np.sqrt((lx - rx)**2 + (ly - ry)**2 + (lz - rz)**2)
    
    window_size = max(1, int(fps * 0.5))
    d_eye_rolling = d_eye.rolling(window=window_size, min_periods=1, center=True).mean()
    
    # ±20% 以上の急激なスケール変動を検知
    mask_scale = (d_eye - d_eye_rolling).abs() / d_eye_rolling > scale_threshold
    
    # ─── 2. 「三角形の形状（幾何構造 Ratio R）」による型崩れチェック ───
    nx, ny, nz = df['lm_headPose_4_x'], df['lm_headPose_4_y'], df['lm_headPose_4_z']
    mx, my, mz = (lx + rx) / 2, (ly + ry) / 2, (lz + rz) / 2
    d_nose = np.sqrt((nx - mx)**2 + (ny - my)**2 + (nz - mz)**2)
    
    ratio_r = d_nose / d_eye
    ratio_r_rolling = ratio_r.rolling(window=window_size, min_periods=1, center=True).mean()
    ratio_r_std = ratio_r.rolling(window=window_size, min_periods=1, center=True).std().fillna(0)
    
    diff_r = (ratio_r - ratio_r.shift(1)).abs()
    mask_geo = ((ratio_r - ratio_r_rolling).abs() > geo_std_threshold * ratio_r_std.replace(0, 0.01)) | (diff_r > 0.15)
    
    # ─── 3. 「移動速度（1次差分）」による物理限界チェック ───
    fx, fy, fz = df['lm_headPose_10_x'], df['lm_headPose_10_y'], df['lm_headPose_10_z']
    
    disp_nose = np.sqrt((nx.diff())**2 + (ny.diff())**2 + (nz.diff())**2)
    disp_forehead = np.sqrt((fx.diff())**2 + (fy.diff())**2 + (fz.diff())**2)
    
    mask_speed = (disp_nose > speed_threshold) | (disp_forehead > speed_threshold)
    
    # ─── 4. マスクの結合と NaN 化 ───
    combined_mask = mask_scale | mask_geo | mask_speed
    df_clean.loc[combined_mask, lm_cols] = np.nan
    
    # ─── 5. 補完処理 (短時間のみ線形補完) ───
    df_clean[lm_cols] = df_clean[lm_cols].interpolate(method='linear', limit=interp_limit)
    
    # print(f"  [検知ログ] スケール: {mask_scale.sum()} / 幾何: {mask_geo.sum()} / 速度: {mask_speed.sum()} ➔ 合計除外: {combined_mask.sum()} 行")
    
    return df_clean