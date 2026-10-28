import numpy as np
import pandas as pd

def pos_algorithm_paper(
    r: np.ndarray, 
    g: np.ndarray, 
    b: np.ndarray, 
    fs: float = 50.0, 
    window_sec: float = 1.6
) -> np.ndarray:
    """
    原著論文 (Wang et al., 2016 Algorithm 1) に完全準拠した POS アルゴリズム。
    RGB 3チャンネルの時系列輝度データから 1次元の脈波 (bvp_raw) を抽出します。
    """
    N = len(r)
    H = np.zeros(N)
    l = int(window_sec * fs)

    for n in range(N):
        m = n - l + 1
        if m >= 0:
            r_win = r[m : n + 1]
            g_win = g[m : n + 1]
            b_win = b[m : n + 1]

            # 時間軸方向の平均正規化 (Temporal normalization)
            r_n = r_win / (np.mean(r_win) + 1e-8)
            g_n = g_win / (np.mean(g_win) + 1e-8)
            b_n = b_win / (np.mean(b_win) + 1e-8)

            # 色空間への投影 (Projection)
            s1 = g_n - b_n
            s2 = -2 * r_n + g_n + b_n

            # チューニング (Tuning)
            std_s1 = np.std(s1)
            std_s2 = np.std(s2)
            alpha = std_s1 / (std_s2 + 1e-8)
            h = s1 + alpha * s2

            # 重複加算 (Overlap-adding)
            H[m : n + 1] = H[m : n + 1] + (h - np.mean(h))

    return H


def extract_pos_bvp_dataframe(
    df: pd.DataFrame, 
    region_prefix: str | dict = "rppg_bothCheeks", 
    fs: float = 50.0,
    window_sec: float = 1.6
) -> pd.DataFrame:
    """
    DataFrame 内の RGB 列から POS アルゴリズムで脈波を抽出します。
    複数ROI（辞書指定: {"both": "rppg_bothCheeks", "left": "rppg_leftCheek", "right": "rppg_rightCheek"}）
    または単一プレフィックス（文字列）に対応。
    """
    df_out = df.copy()

    if isinstance(region_prefix, dict):
        # 複数領域の処理
        for key, prefix in region_prefix.items():
            r_col, g_col, b_col = f"{prefix}_r", f"{prefix}_g", f"{prefix}_b"
            if r_col in df_out.columns and g_col in df_out.columns and b_col in df_out.columns:
                r = df_out[r_col].interpolate(method='linear').bfill().ffill().values
                g = df_out[g_col].interpolate(method='linear').bfill().ffill().values
                b = df_out[b_col].interpolate(method='linear').bfill().ffill().values
                bvp_raw_roi = pos_algorithm_paper(r, g, b, fs=fs, window_sec=window_sec)
                df_out[f"bvp_raw_{key}"] = bvp_raw_roi
                if key == "both" or "bvp_raw" not in df_out.columns:
                    df_out["bvp_raw"] = bvp_raw_roi
    else:
        # 単一領域の処理
        r_col, g_col, b_col = f"{region_prefix}_r", f"{region_prefix}_g", f"{region_prefix}_b"
        if r_col in df_out.columns and g_col in df_out.columns and b_col in df_out.columns:
            r = df_out[r_col].interpolate(method='linear').bfill().ffill().values
            g = df_out[g_col].interpolate(method='linear').bfill().ffill().values
            b = df_out[b_col].interpolate(method='linear').bfill().ffill().values
            bvp_raw = pos_algorithm_paper(r, g, b, fs=fs, window_sec=window_sec)
            df_out['bvp_raw'] = bvp_raw
        else:
            df_out['bvp_raw'] = np.nan

    return df_out