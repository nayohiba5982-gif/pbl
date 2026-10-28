import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def apply_butterworth_lowpass(
    series: pd.Series, 
    cutoff: float = 3.0, 
    fs: float = 50.0, 
    order: int = 2
) -> np.ndarray:
    """
    1次元の時系列データ（Series）に対して Butterworth ゼロ位相ローパスフィルタを適用します。
    
    Parameters:
        series (pd.Series): 入力データ列
        cutoff (float): カットオフ周波数 [Hz] (この周波数以上のノイズを遮断)
        fs (float): サンプリング周波数 [Hz] (デフォルト: 50.0Hz)
        order (int): フィルタの次数 (デフォルト: 2次)
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    
    # カットオフ周波数がナイキスト周波数以上の場合は処理しない（安全策）
    if normal_cutoff >= 1.0:
        return series.values

    # フィルタ係数 (b, a) の算出
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # 欠損値 (NaN) マスクの保持
    valid_mask = series.notna()
    
    # 有効データ点が少なすぎる場合はそのまま返す
    if valid_mask.sum() < (order * 3):
        return series.values

    # filtfilt は内部に NaN があると全体が NaN になるため、一時的に線形補完して計算
    s_filled = series.interpolate(method='linear').bfill().ffill().values
    
    # ゼロ位相フィルタリング (前方向 + 逆方向の2回掛けで位相遅れをゼロ化)
    filtered = filtfilt(b, a, s_filled)
    
    # 元々の長時間 NaN 領域を復元
    filtered[~valid_mask] = np.nan
    return filtered


def apply_lowpass_to_dataframe(
    df: pd.DataFrame, 
    fs: float = 50.0,
    default_cutoff: float = 3.0
) -> pd.DataFrame:
    """
    DataFrame内の各データ列の特性に合わせたカットオフ周波数でローパスフィルタを一括適用します。
    """
    df_filtered = df.copy()
    
    # 時間軸・メタデータ関連カラムは除外
    skip_cols = {'time', 'clientEpochMs', 'clientPerfMs', 'label', 'username'}
    
    for col in df_filtered.columns:
        if col in skip_cols:
            continue
            
        # 数値列以外はスキップ
        if not np.issubdtype(df_filtered[col].dtype, np.number):
            continue

        # ─── データ種別ごとに最適なカットオフ周波数を設定 ───
        # 1. 頭部姿勢・顔中心 (動きが緩やかなため、低めのカットオフで平滑化: 2.0Hz)
        if any(k in col for k in ['headPose', 'centroid']):
            cutoff = 2.0
            
        # 2. 視線座標 (サッケード/視線移動の速さに追従させるため少し高め: 4.5Hz)
        elif 'gaze' in col or 'eye' in col:
            cutoff = 4.5
            
        # 3. EAR（まばたき）や目・口の開閉量 (まばたきの鋭さを保つため: 5.0Hz)
        elif 'calc_' in col or 'ear' in col.lower() or 'blink' in col.lower():
            cutoff = 5.0
            
        # 4. その他の一般ランドマーク座標 (標準: 3.0Hz)
        else:
            cutoff = default_cutoff

        # フィルタ実行
        df_filtered[col] = apply_butterworth_lowpass(df_filtered[col], cutoff=cutoff, fs=fs, order=2)

    return df_filtered