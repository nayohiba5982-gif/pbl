"""
固定長スライディングウィンドウ生成コアモジュール
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# preprocess 直下のモジュールをインポート可能にする
_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

try:
    from config_loader import log
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("windowing")

try:
    from .trim_margins import normalize_trim_margins
except ImportError:
    from trim_margins import normalize_trim_margins

try:
    from features.rppg.hrv_extraction import HRV_METRIC_COLS, ensure_hrv_in_dataframe
except ImportError:
    HRV_METRIC_COLS = ["rmssd", "sdnn", "sd2_sd1", "mean_pp"]
    ensure_hrv_in_dataframe = None


def create_sliding_windows_for_segment(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_sec: float,
    stride_sec: float = 1.0,
    fs: float = 50.0,
    max_nan_ratio: float = 0.1,
    time_col: str = "time",
    user_id: str = "unknown",
    label_val: int = 0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    同一被験者・同一ラベルの連続時系列セグメントから固定長スライディングウィンドウを作成します。

    Returns:
        X (np.ndarray): 形状 [N_windows, Window_steps, Channels] の3次元配列
        meta_df (pd.DataFrame): 各ウィンドウの被験者名、ラベル、開始・終了インデックス、時刻
    """
    win_steps = int(window_sec * fs)
    stride_steps = int(stride_sec * fs)
    total_steps = len(df)

    if total_steps < win_steps:
        return np.empty((0, win_steps, len(feature_cols))), pd.DataFrame()

    windows = []
    meta_records = []

    for start_idx in range(0, total_steps - win_steps + 1, stride_steps):
        end_idx = start_idx + win_steps
        sub_df = df.iloc[start_idx:end_idx]

        # NaN チェック（欠損率チェック）
        feature_data = sub_df[feature_cols].values
        nan_ratio = np.isnan(feature_data).mean()

        if nan_ratio > max_nan_ratio:
            continue

        # 軽微な NaN は線形補間/ゼロ埋め
        if np.isnan(feature_data).any():
            feature_data = (
                pd.DataFrame(feature_data)
                .interpolate(method="linear")
                .bfill()
                .ffill()
                .fillna(0.0)
                .values
            )

        windows.append(feature_data)

        t_start = sub_df[time_col].iloc[0] if time_col in sub_df.columns else start_idx
        t_end = sub_df[time_col].iloc[-1] if time_col in sub_df.columns else end_idx

        rec = {
            "username": user_id,
            "label": int(label_val),
            "window_sec": window_sec,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_time": t_start,
            "end_time": t_end,
        }
        if "bpm_base_10s" in sub_df.columns:
            rec["bpm_base_10s"] = float(np.nanmean(sub_df["bpm_base_10s"]))
        if "bpm" in sub_df.columns:
            rec["bpm_mean"] = float(np.nanmean(sub_df["bpm"]))
        for hrv_col in HRV_METRIC_COLS:
            if hrv_col in sub_df.columns:
                val = np.nanmean(sub_df[hrv_col])
                rec[hrv_col] = float(0.0 if np.isnan(val) else val)

        meta_records.append(rec)

    if not windows:
        return np.empty((0, win_steps, len(feature_cols))), pd.DataFrame()

    X = np.array(windows)
    meta_df = pd.DataFrame(meta_records)
    return X, meta_df


def create_sliding_windows_by_label_from_df(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_sec: float,
    stride_sec: float = 1.0,
    fs: float = 50.0,
    max_nan_ratio: float = 0.1,
    time_col: str = "time",
    user_col: str = "username",
    label_col: str = "label",
    trim_margins: dict | None = None,
) -> dict[int, tuple[np.ndarray, pd.DataFrame]]:
    """
    1被験者のDataFrameから、label（0/1）ごとに分けてスライディングウィンドウを作成します。
    動画の切り替え境界をまたがないよう、各ラベルの連続区間ごとにウィンドウを抽出します。
    また、各ラベルの区間開始・終了部分のマージン（不要な区間）を除外（トリミング）します。
    デフォルト除外設定:
      - label 1: 最初の 15s 間、最後の 4s 間
      - label 0: 最初の 15s 間

    Returns:
        dict[int, tuple[np.ndarray, pd.DataFrame]]:
            {0: (X_0, meta_0), 1: (X_1, meta_1)} の辞書
    """
    norm_trim_margins = normalize_trim_margins(trim_margins)

    # 特徴量列の存在チェックと次元数の一致保証 (HRV列が未算出の場合は45秒窓で自動算出・同期)
    df_work = df.copy()
    if ensure_hrv_in_dataframe is not None and any(c in HRV_METRIC_COLS for c in feature_cols):
        df_work = ensure_hrv_in_dataframe(df_work, fs=fs, window_sec=45.0, stride_sec=1.0, time_col=time_col)
    missing_cols = [c for c in feature_cols if c not in df_work.columns]
    if len(missing_cols) == len(feature_cols):
        raise KeyError(f"データフレームに必要な特徴量列が1つも存在しません: {feature_cols}")
    if missing_cols:
        log.warning(f"一部の特徴量列 {missing_cols} が存在しないため、NaN列として補填しチャンネル数を維持します。")
        for mc in missing_cols:
            df_work[mc] = np.nan
    actual_feature_cols = list(feature_cols)

    user_id = str(df_work[user_col].iloc[0]) if user_col in df_work.columns else "unknown"
    win_steps = int(window_sec * fs)

    # label 列がない場合は全体を label=0 として扱う
    if label_col not in df_work.columns:
        margin = norm_trim_margins.get(0, {"start_sec": 0.0, "end_sec": 0.0})
        start_trim_steps = int(round(margin.get("start_sec", 0.0) * fs))
        end_trim_steps = int(round(margin.get("end_sec", 0.0) * fs))
        total_steps = len(df_work)
        if total_steps <= start_trim_steps + end_trim_steps:
            return {0: (np.empty((0, win_steps, len(actual_feature_cols))), pd.DataFrame())}
        end_slice = total_steps - end_trim_steps if end_trim_steps > 0 else total_steps
        trimmed_df = df_work.iloc[start_trim_steps:end_slice].reset_index(drop=True)

        X, meta = create_sliding_windows_for_segment(
            df=trimmed_df,
            feature_cols=actual_feature_cols,
            window_sec=window_sec,
            stride_sec=stride_sec,
            fs=fs,
            max_nan_ratio=max_nan_ratio,
            time_col=time_col,
            user_id=user_id,
            label_val=0,
        )
        return {0: (X, meta)}

    # PCHIP等のリサンプリング補間による小数を二値化（0.5を閾値として 0 または 1）
    labels_binary = (df_work[label_col].fillna(0) >= 0.5).astype(int).values
    df_work["_clean_label"] = labels_binary

    # 連続する同一ラベル区間（ブロック）を検出
    df_work["_block_id"] = (df_work["_clean_label"] != df_work["_clean_label"].shift(1)).cumsum()

    result_by_label = {}
    unique_labels = sorted(df_work["_clean_label"].unique())

    for l_val in unique_labels:
        l_blocks = df_work[df_work["_clean_label"] == l_val]
        x_list = []
        meta_list = []

        margin = norm_trim_margins.get(int(l_val), {"start_sec": 0.0, "end_sec": 0.0})
        start_trim_sec = float(margin.get("start_sec", 0.0))
        end_trim_sec = float(margin.get("end_sec", 0.0))
        start_trim_steps = int(round(start_trim_sec * fs))
        end_trim_steps = int(round(end_trim_sec * fs))

        for block_id, block_df in l_blocks.groupby("_block_id"):
            block_len = len(block_df)
            total_trim = start_trim_steps + end_trim_steps

            if block_len <= total_trim:
                log.info(
                    f"  ✂️ [除外] 被験者={user_id}, Label={l_val}, Block={block_id}: "
                    f"ステップ数 ({block_len}) <= トリム合計ステップ ({total_trim} steps, {start_trim_sec + end_trim_sec:.1f}s) のためスキップ"
                )
                continue

            end_slice = block_len - end_trim_steps if end_trim_steps > 0 else block_len
            trimmed_df = block_df.iloc[start_trim_steps:end_slice].reset_index(drop=True)

            if len(trimmed_df) < win_steps:
                log.info(
                    f"  ⚠️ [短小スキップ] 被験者={user_id}, Label={l_val}, Block={block_id}: "
                    f"トリム後ステップ数 ({len(trimmed_df)}) < 窓幅 ({win_steps} steps, {window_sec}s)"
                )
                continue

            X_b, meta_b = create_sliding_windows_for_segment(
                df=trimmed_df,
                feature_cols=actual_feature_cols,
                window_sec=window_sec,
                stride_sec=stride_sec,
                fs=fs,
                max_nan_ratio=max_nan_ratio,
                time_col=time_col,
                user_id=user_id,
                label_val=int(l_val),
            )

            if len(X_b) > 0:
                x_list.append(X_b)
                meta_list.append(meta_b)

        if x_list:
            X_l = np.concatenate(x_list, axis=0)
            meta_l = pd.concat(meta_list, ignore_index=True)
        else:
            X_l = np.empty((0, win_steps, len(actual_feature_cols)))
            meta_l = pd.DataFrame()

        result_by_label[int(l_val)] = (X_l, meta_l)

    return result_by_label


def create_sliding_windows_from_df(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_sec: float,
    stride_sec: float = 1.0,
    fs: float = 50.0,
    max_nan_ratio: float = 0.1,
    time_col: str = "time",
    user_col: str = "username",
    label_col: str = "label",
    trim_margins: dict | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    【後方互換性用】
    単一のDataFrameからラベル境界をまたがないようにウィンドウを作成し、
    被験者内の全ラベルを結合した (X, meta_df) を返します。
    """
    by_label = create_sliding_windows_by_label_from_df(
        df=df,
        feature_cols=feature_cols,
        window_sec=window_sec,
        stride_sec=stride_sec,
        fs=fs,
        max_nan_ratio=max_nan_ratio,
        time_col=time_col,
        user_col=user_col,
        label_col=label_col,
        trim_margins=trim_margins,
    )

    x_list = [v[0] for v in by_label.values() if len(v[0]) > 0]
    meta_list = [v[1] for v in by_label.values() if len(v[1]) > 0]

    win_steps = int(window_sec * fs)
    if not x_list:
        return np.empty((0, win_steps, len(feature_cols))), pd.DataFrame()

    X_all = np.concatenate(x_list, axis=0)
    meta_all = pd.concat(meta_list, ignore_index=True)
    return X_all, meta_all


def create_sliding_windows_for_user_from_df(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_sec: float,
    stride_sec: float = 1.0,
    fs: float = 50.0,
    max_nan_ratio: float = 0.1,
    time_col: str = "time",
    user_col: str = "username",
    label_col: str = "label",
    hrv_window_sec: float = 45.0,
    hrv_stride_sec: float = 1.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    1被験者の全時系列DataFrameから、ラベル境界で時系列を分割せず、
    被験者一人につき1つの連続したスライディングウィンドウ配列 (X, meta_df) を作成します。

    Returns:
        tuple[np.ndarray, pd.DataFrame]:
            X: 形状 [N_windows, Window_steps, Channels]
            meta_df: 各ウィンドウのメタデータ (username, label, label_mean, start_time, end_time等)
    """
    user_id = str(df[user_col].iloc[0]) if user_col in df.columns else "unknown"
    win_steps = int(window_sec * fs)
    stride_steps = int(stride_sec * fs)
    total_steps = len(df)

    if total_steps < win_steps:
        return np.empty((0, win_steps, len(feature_cols))), pd.DataFrame()

    # 欠損列の補填 (HRV列が要求され、かつ未算出の場合は指定窓幅で自動算出・タイムスタンプ同期)
    df_work = df.copy()
    if ensure_hrv_in_dataframe is not None and any(c in HRV_METRIC_COLS for c in feature_cols):
        df_work = ensure_hrv_in_dataframe(
            df_work,
            fs=fs,
            window_sec=hrv_window_sec,
            stride_sec=hrv_stride_sec,
            time_col=time_col,
        )
    missing_cols = [c for c in feature_cols if c not in df_work.columns]
    for mc in missing_cols:
        df_work[mc] = np.nan

    windows = []
    meta_records = []

    for start_idx in range(0, total_steps - win_steps + 1, stride_steps):
        end_idx = start_idx + win_steps
        sub_df = df_work.iloc[start_idx:end_idx]

        feature_data = sub_df[feature_cols].values
        nan_ratio = np.isnan(feature_data).sum() / feature_data.size
        if nan_ratio > max_nan_ratio:
            continue

        # 欠損補間
        if np.isnan(feature_data).any():
            feature_data = (
                pd.DataFrame(feature_data)
                .interpolate(method="linear")
                .bfill()
                .ffill()
                .fillna(0.0)
                .values
            )

        windows.append(feature_data)

        t_start = sub_df[time_col].iloc[0] if time_col in sub_df.columns else start_idx / fs
        t_end = sub_df[time_col].iloc[-1] if time_col in sub_df.columns else end_idx / fs

        # ウィンドウ内のラベル判定 (最頻値 & 平均比率)
        if label_col in sub_df.columns:
            l_series = sub_df[label_col].dropna()
            if len(l_series) > 0:
                l_mean = float(l_series.mean())
                l_val = int(round(l_mean >= 0.5))
            else:
                l_val = 0
                l_mean = 0.0
        else:
            l_val = 0
            l_mean = 0.0

        rec = {
            "username": user_id,
            "label": l_val,
            "label_mean": round(l_mean, 4),
            "window_sec": window_sec,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_time": round(float(t_start), 3),
            "end_time": round(float(t_end), 3),
        }
        if "bpm_base_10s" in sub_df.columns:
            rec["bpm_base_10s"] = float(np.nanmean(sub_df["bpm_base_10s"]))
        if "bpm" in sub_df.columns:
            rec["bpm_mean"] = float(np.nanmean(sub_df["bpm"]))
        for hrv_col in HRV_METRIC_COLS:
            if hrv_col in sub_df.columns:
                val = np.nanmean(sub_df[hrv_col])
                rec[hrv_col] = float(0.0 if np.isnan(val) else val)

        meta_records.append(rec)

    if not windows:
        return np.empty((0, win_steps, len(feature_cols))), pd.DataFrame()

    X = np.array(windows)
    meta_df = pd.DataFrame(meta_records)
    return X, meta_df
