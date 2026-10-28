import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, PchipInterpolator


def resample_lm_dataframe(
    df: pd.DataFrame, 
    target_fps: float = 50.0, 
    time_col: str = "time",
    method: str = "pchip"
) -> pd.DataFrame:
    """
    顔ランドマーク (x, y, z 座標等) データを 50Hz の等間隔格子へリサンプリングします。
    顔検出不可（NaN）区間の制御を厳密に行います。
    """
    if time_col not in df.columns:
        raise ValueError(f"時間カラム '{time_col}' が存在しません。")

    # 0. 重複除去 & 昇順ソート
    df_sorted = df.drop_duplicates(subset=[time_col]).sort_values(by=time_col).reset_index(drop=True)
    t_orig = df_sorted[time_col].values

    # 1. 50Hz グリッド生成 (dt = 0.02s)
    dt = 1.0 / target_fps
    t_new = np.arange(t_orig[0], t_orig[-1], dt)

    resampled_data = {time_col: t_new}

    for col in df_sorted.columns:
        if col == time_col:
            continue

        y_orig = df_sorted[col].values

        # 非数値列および離散メタデータ列（label, username等）は直近値保持 (Forward Fill)
        is_discrete = (col in ['label', 'username', 'session_id', 'clientEpochMs', 'clientPerfMs'])
        if not np.issubdtype(y_orig.dtype, np.number) or is_discrete:
            idx = np.searchsorted(t_orig, t_new, side='right') - 1
            idx = np.clip(idx, 0, len(y_orig) - 1)
            resampled_data[col] = y_orig[idx]
            continue

        # 顔未検出など全フレーム NaN の場合
        nan_mask = np.isnan(y_orig)
        if np.all(nan_mask):
            resampled_data[col] = np.nan
            continue

        # 2. 座標補完 (pchip または linear)
        if method == "linear":
            f_interp = interp1d(t_orig, y_orig, kind='linear', bounds_error=False, fill_value="extrapolate")
            resampled_data[col] = f_interp(t_new)
        else:
            if np.any(nan_mask):
                y_temp = pd.Series(y_orig).interpolate(method='linear').bfill().ffill().values
            else:
                y_temp = y_orig

            f_interp = PchipInterpolator(t_orig, y_temp, extrapolate=True)
            resampled_data[col] = f_interp(t_new)

            # 顔検出が切れていた区間（NaN 区間）はリサンプリング後も正確に NaN へ戻す
            if np.any(nan_mask):
                f_nan = interp1d(t_orig, nan_mask.astype(float), kind='nearest', bounds_error=False, fill_value=1.0)
                resampled_data[col][f_nan(t_new) > 0.5] = np.nan

    return pd.DataFrame(resampled_data)