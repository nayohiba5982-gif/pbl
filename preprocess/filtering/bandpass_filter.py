import numpy as np
import pandas as pd
import neurokit2 as nk

def apply_bvp_bandpass_filter(
    bvp_signal: np.ndarray, 
    fs: float = 50.0, 
    lowcut: float = 0.75, 
    highcut: float = 2.5,
    order: int = 2
) -> np.ndarray:
    """
    1次元脈波 (bvp_raw) に対し、心拍帯域 (0.75Hz - 2.5Hz) の
    ゼロ位相 Butterworth バンドパスフィルタ (NeuroKit2) を適用します。
    """
    s_series = pd.Series(bvp_signal)
    valid_mask = s_series.notna()

    if valid_mask.sum() < order * 3:
        return bvp_signal

    # フィルタ入力用の欠損補間
    s_filled = s_series.interpolate(method='linear').bfill().ffill().values

    # バンドパスフィルタの適用 (ゼロ位相)
    bvp_clean = nk.signal_filter(
        s_filled, 
        sampling_rate=int(fs), 
        lowcut=lowcut, 
        highcut=highcut, 
        method='butterworth', 
        order=order
    )

    # 元の NaN 位置を再復元
    bvp_clean[~valid_mask] = np.nan
    return bvp_clean


def process_filtering_dataframe(
    df: pd.DataFrame, 
    target_col: str | list[str] = "bvp_raw", 
    fs: float = 50.0,
    lowcut: float = 0.75,
    highcut: float = 2.5,
    order: int = 2
) -> pd.DataFrame:
    """
    DataFrame 内の target_col（または存在するすべての bvp_raw* 列）にバンドパスフィルタを適用し、
    対応する 'bvp_clean*' カラムを追加します。
    """
    df_out = df.copy()

    # 対象カラムの特定
    if isinstance(target_col, str):
        # target_col が存在すればそれを使い、さらに bvp_raw_both 等があればそれらも対象にする
        candidates = [target_col] if target_col in df_out.columns else []
        for c in df_out.columns:
            if c.startswith("bvp_raw") and c not in candidates:
                candidates.append(c)
    else:
        candidates = [c for c in target_col if c in df_out.columns]

    if not candidates:
        raise ValueError(f"フィルタ適用対象の脈波カラムが見つかりません。")

    for col in candidates:
        clean_col = col.replace("bvp_raw", "bvp_clean")
        df_out[clean_col] = apply_bvp_bandpass_filter(
            df_out[col].values, 
            fs=fs, 
            lowcut=lowcut, 
            highcut=highcut, 
            order=order
        )

    return df_out