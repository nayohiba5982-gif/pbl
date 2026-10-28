import numpy as np
import pandas as pd
from scipy.signal import detrend

def remove_moving_average_trend(
    series: pd.Series, 
    window_sec: float = 10.0, 
    fs: float = 50.0
) -> pd.Series:
    """
    ① 移動平均差分法 (High-pass / Moving Average Subtraction)
    適用対象: 頭部角度 (Pitch, Yaw, Roll), 顔の中心座標 (Centroid)
    効果: 徐々の姿勢崩れを除去し、瞬間的なうなずき・カクつき等の動きのみを0中心で抽出
    """
    window_size = max(3, int(window_sec * fs))
    # ローリング平均（移動平均）の算出
    rolling_mean = series.rolling(window=window_size, min_periods=1, center=True).mean()
    # 元の信号から移動平均を差し引く
    detrended = series - rolling_mean
    return detrended


def normalize_ear_baseline(
    series: pd.Series, 
    window_sec: float = 20.0, 
    fs: float = 50.0, 
    quantile: float = 0.95
) -> pd.Series:
    """
    ② ローリング・ベースライン補正 (Rolling Quantile Normalization)
    適用対象: まばたき指標 (EAR: Eye Aspect Ratio)
    効果: 直近の開眼時(95%パーセンタイル)を 1.0 に正規化し、個人差や頭の傾きによるバイアスを防止
    """
    window_size = max(3, int(window_sec * fs))
    # 直近N秒間の開眼状態のEAR(上位95%点)をベースラインとして抽出
    baseline = series.rolling(window=window_size, min_periods=1, center=True).quantile(quantile)
    # ベースラインで割り算 (通常開眼時 ≒ 1.0, 閉眼時 ≒ 0.5〜0.7 にドロップ)
    normalized = series / (baseline + 1e-6)
    return normalized


def remove_linear_trend(series: pd.Series) -> pd.Series:
    """
    ③ 線形・多項式デトレンド (Linear Detrending)
    適用対象: 視線座標 (Gaze X, Y)
    効果: 視線トラッキング特有の「時間経過に伴う全般的な座標のズレ（ドリフト）」を除去
    """
    valid_mask = series.notna()
    if valid_mask.sum() < 3:
        return series

    # scipy.signal.detrend は NaN に未対応のため一時的に線形補完
    s_filled = series.interpolate(method='linear').bfill().ffill().values
    detrended_vals = detrend(s_filled, type='linear')

    result = pd.Series(detrended_vals, index=series.index)
    result[~valid_mask] = np.nan
    return result


def detrend_landmark_dataframe(df: pd.DataFrame, fs: float = 50.0) -> pd.DataFrame:
    """
    DataFrame内の各カラムに対して最適なトレンド除去手法を自動適用します。
    """
    df_out = df.copy()
    skip_cols = {'time', 'clientEpochMs', 'clientPerfMs', 'label', 'username'}
    for col in df_out.columns:
        if col in skip_cols or not np.issubdtype(df_out[col].dtype, np.number):
            continue

        # ランドマーク生座標 (lm_*) は各軸独立にデトレンドすると顔の幾何構造（ユークリッド距離・角度）が
        # 歪むためスキップし、幾何特徴量算出後の角度・指標列に対して適用します
        if col.startswith('lm_'):
            continue

        # ─── 1. EAR (まばたき) ➔ ローリングベースライン補正 (通常時=1.0) ───
        if 'calc_ear' in col or 'ear' in col.lower() or 'blink' in col.lower():
            df_out[col] = normalize_ear_baseline(df_out[col], window_sec=20.0, fs=fs, quantile=0.95)

        # ─── 2. 頭部角度・中心座標 ➔ 移動平均差分法 (10秒窓でドリフト除去) ───
        elif any(k in col for k in ['headPose', 'centroid', 'pitch', 'yaw', 'roll']):
            df_out[col] = remove_moving_average_trend(df_out[col], window_sec=10.0, fs=fs)

        # ─── 3. 視線座標 ➔ 線形デトレンド (全般的な位置ズレ除去) ───
        elif 'gaze' in col or 'eye' in col:
            df_out[col] = remove_linear_trend(df_out[col])

    return df_out