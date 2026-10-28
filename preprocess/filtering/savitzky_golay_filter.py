import pandas as pd
from scipy.signal import savgol_filter

def apply_savgol_filter(df: pd.DataFrame, target_cols: list, cfg: dict) -> pd.DataFrame:
    """Savitzky-Golayフィルタを適用して高周波ノイズを除去"""
    res_df = df.copy()

    savgol_cfg = cfg.get("landmarks", {}).get("savgol", {})
    window = savgol_cfg.get("window_length", 11)
    poly = savgol_cfg.get("polyorder", 2)

    # データ長がウィンドウサイズより短い場合の安全策
    if len(res_df) < window:
        window = len(res_df) if len(res_df) % 2 != 0 else len(res_df) - 1

    if window <= poly:
        return res_df  # ウィンドウ長が不足している場合はフィルタ処理をスキップ

    for col in target_cols:
        if col in res_df.columns:
            valid_mask = res_df[col].notna()
            if valid_mask.sum() <= poly:
                res_df[f"{col}_smooth"] = res_df[col]
                continue

            series = res_df[col].interpolate(method='linear').bfill().ffill().values
            smoothed = savgol_filter(
                series,
                window_length=window,
                polyorder=poly
            )
            smoothed[~valid_mask.values] = np.nan
            res_df[f"{col}_smooth"] = smoothed

    return res_df