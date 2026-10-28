import numpy as np
import pandas as pd

try:
    from features.rppg.sqi import calculate_snr
except ImportError:
    from .sqi import calculate_snr


def select_best_bvp(
    df: pd.DataFrame, 
    roi_cols: dict, 
    fs: float = 50.0
) -> tuple[np.ndarray, str, float]:
    """
    3領域のSNRを比較し、最も適切なBVP信号、選択領域名、SNR値を返します。
    """
    # 存在しないカラムは除外
    valid_rois = {r: c for r, c in roi_cols.items() if c in df.columns}
    
    # 指定ROIカラムが1つもない場合は bvp_clean または bvp_raw を探す
    if not valid_rois:
        for fallback_col in ["bvp_clean", "bvp_raw"]:
            if fallback_col in df.columns:
                valid_rois = {"default": fallback_col}
                break

    if not valid_rois:
        raise ValueError("DataFrameに有効なBVPカラムが見つかりません。")

    snrs = {}
    for region, col in valid_rois.items():
        snrs[region] = calculate_snr(df[col].values, fs=fs)

    snr_both = snrs.get("both", -99.0)
    snr_left = snrs.get("left", -99.0)
    snr_right = snrs.get("right", -99.0)

    # ロジック: 片側が both より 2.0dB 以上勝っていれば片側を採用、それ以外は空間平均効果のある both を優先
    if "left" in snrs and "right" in snrs:
        max_single_region = "left" if snr_left > snr_right else "right"
        max_single_snr = max(snr_left, snr_right)
        if max_single_snr > snr_both + 2.0:
            selected_region = max_single_region
        else:
            selected_region = "both" if "both" in snrs else max(snrs, key=snrs.get)
    else:
        selected_region = max(snrs, key=snrs.get)

    selected_col = valid_rois[selected_region]
    best_bvp = df[selected_col].values
    best_snr = snrs[selected_region]

    return best_bvp, selected_region, best_snr