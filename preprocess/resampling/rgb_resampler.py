import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, PchipInterpolator


def resample_rgb_dataframe(
    df: pd.DataFrame, 
    target_fps: float = 50.0, 
    time_col: str = "time"
) -> pd.DataFrame:
    """
    RGB (rPPG) 時系列データを 50Hz の等間隔格子へリサンプリングします。
    脈波波形（ピーク）の崩れを防ぐため PCHIP 補間を適用します。
    """
    if time_col not in df.columns:
        raise ValueError(f"時間カラム '{time_col}' が存在しません。")

    # 重複除去 & 昇順ソート
    df_sorted = df.drop_duplicates(subset=[time_col]).sort_values(by=time_col).reset_index(drop=True)
    t_orig = df_sorted[time_col].values

    # 50Hz グリッド生成
    dt = 1.0 / target_fps
    t_new = np.arange(t_orig[0], t_orig[-1], dt)

    resampled_data = {time_col: t_new}

    for col in df_sorted.columns:
        if col == time_col:
            continue

        y_orig = df_sorted[col].values

        # 非数値列および離散メタデータ列（label, username等）は直前値保持 (Forward Fill)
        is_discrete = (col in ['label', 'username', 'session_id', 'clientEpochMs', 'clientPerfMs'])
        if not np.issubdtype(y_orig.dtype, np.number) or is_discrete:
            idx = np.searchsorted(t_orig, t_new, side='right') - 1
            idx = np.clip(idx, 0, len(y_orig) - 1)
            resampled_data[col] = y_orig[idx]
            continue

        # 欠損値対策
        nan_mask = np.isnan(y_orig)
        if np.all(nan_mask):
            resampled_data[col] = np.nan
            continue

        # PCHIP 補間による波形補完
        if np.any(nan_mask):
            y_temp = pd.Series(y_orig).interpolate(method='linear').bfill().ffill().values
        else:
            y_temp = y_orig

        f_interp = PchipInterpolator(t_orig, y_temp, extrapolate=True)
        resampled_data[col] = f_interp(t_new)

        # 全体的な欠損（カメラ不覆い等）のマスク復元
        if np.any(nan_mask):
            f_nan = interp1d(t_orig, nan_mask.astype(float), kind='nearest', bounds_error=False, fill_value=1.0)
            resampled_data[col][f_nan(t_new) > 0.5] = np.nan

    return pd.DataFrame(resampled_data)