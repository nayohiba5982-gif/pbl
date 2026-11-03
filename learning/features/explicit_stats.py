"""
各ウィンドウ時系列から人間が解釈可能な明示的物理統計量を算出するモジュール (Explicit Statistics Aggregation)
(learning/features/explicit_stats.py)
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

try:
    from config_loader import log
except ImportError:
    import logging
    log = logging.getLogger("explicit_statistics")

try:
    from .constants import (
        DEFAULT_FEATURE_COLS,
        DEFAULT_HRV_COLS,
        DEFAULT_HEAD_POSE_COLS,
        DEFAULT_GAZE_COLS,
    )
except ImportError:
    from constants import (
        DEFAULT_FEATURE_COLS,
        DEFAULT_HRV_COLS,
        DEFAULT_HEAD_POSE_COLS,
        DEFAULT_GAZE_COLS,
    )


def _compute_hrv_from_subject_csvs(
    meta_df: pd.DataFrame,
    hrv_cols: list[str],
    n_samples: int,
) -> dict[str, np.ndarray] | None:
    """
    X_windows.npy が旧8チャネル形式で、かつ metadata.csv に HRV 列が未付与の場合に、
    preprocess/data/features/{username}.csv から 45秒窓 HRV を算出して各ウィンドウへ同期します。
    """
    if meta_df is None or len(meta_df) != n_samples or "username" not in meta_df.columns:
        return None

    repo_root = Path(__file__).resolve().parent.parent.parent
    features_dir = repo_root / "preprocess" / "data" / "features"
    rppg_dir = repo_root / "preprocess" / "features" / "rppg"
    if not features_dir.exists() or not rppg_dir.exists():
        return None

    if str(rppg_dir) not in sys.path:
        sys.path.insert(0, str(rppg_dir))
    try:
        from hrv_extraction import ensure_hrv_in_dataframe
    except Exception:
        return None

    result = {c: np.zeros(n_samples, dtype=np.float32) for c in hrv_cols}
    usernames = meta_df["username"].astype(str).values
    has_idx = "start_idx" in meta_df.columns and "end_idx" in meta_df.columns
    has_time = "start_time" in meta_df.columns and "end_time" in meta_df.columns
    if not has_idx and not has_time:
        return None

    matched_any = False
    for u in np.unique(usernames):
        csv_path = features_dir / f"{u}.csv"
        if not csv_path.exists():
            continue
        try:
            u_df = pd.read_csv(csv_path)
            u_df = ensure_hrv_in_dataframe(u_df, fs=50.0, window_sec=45.0, stride_sec=1.0)
        except Exception:
            continue

        u_indices = np.where(usernames == u)[0]
        matched_any = True
        for row_idx in u_indices:
            row = meta_df.iloc[row_idx]
            if has_idx:
                s_i = int(row["start_idx"])
                e_i = int(row["end_idx"])
                sub = u_df.iloc[s_i:e_i]
            else:
                s_t = float(row["start_time"])
                e_t = float(row["end_time"])
                sub = u_df[(u_df["time"] >= s_t) & (u_df["time"] <= e_t)]

            for c in hrv_cols:
                if c in sub.columns and len(sub) > 0:
                    val = np.nanmean(sub[c].values)
                    result[c][row_idx] = 0.0 if np.isnan(val) else float(val)

    return result if matched_any else None


def compute_explicit_statistics(
    X: np.ndarray,
    feature_cols: list[str] | None = None,
    head_pose_cols: list[str] | None = None,
    gaze_cols: list[str] | None = None,
    hrv_cols: list[str] | None = None,
    mode: str = "standard_24d",
    meta_df: pd.DataFrame | None = None,
    bpm_base: np.ndarray | None = None,
    baseline_mode: str = "absolute",
) -> tuple[np.ndarray, list[str]]:
    """
    時系列ウィンドウ配列 X [N, T, C] から、人間が解釈可能な明示的物理統計量を集約・算出します。

    Parameters:
        X (np.ndarray): 形状 [N_samples, Time_steps, Channels] の3次元配列 (12ch または 旧8ch)
        feature_cols (list[str] | None): チャネルごとの名前リスト (デフォルト: 12チャネル)
        head_pose_cols (list[str] | None): 頭部姿勢のカラム名
        gaze_cols (list[str] | None): 視線のカラム名
        hrv_cols (list[str] | None): HRV主要4指標のカラム名 (rmssd, sdnn, sd2_sd1, mean_pp)
        mode (str):
            - "standard_24d" / "standard_20d": 平均(8) + 標準偏差(8) + 移動量分散(4) + HRV(4) = 24次元
            - "all_variances": 平均(8) + 標準偏差(8) + 移動量分散全種(7) + HRV(4) = 27次元
        meta_df (pd.DataFrame | None): 各ウィンドウのメタデータ (bpm_base_10s, rmssd, sdnn, sd2_sd1, mean_pp 等を含む)
        bpm_base (np.ndarray | None): 明示的に渡された局所心拍ベースライン水準配列 [N]
        baseline_mode (str): "absolute" (そのままのBPM水準) または "relative_to_first" (先頭基準窓との差分)

    Returns:
        stat_matrix (np.ndarray): 形状 [N_samples, D_stat] (例: [N, 24]) の2次元配列
        stat_feature_names (list[str]): 特徴量名のリスト
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS
    if head_pose_cols is None:
        head_pose_cols = DEFAULT_HEAD_POSE_COLS
    if gaze_cols is None:
        gaze_cols = DEFAULT_GAZE_COLS
    if hrv_cols is None:
        hrv_cols = DEFAULT_HRV_COLS

    n_samples, time_steps, n_channels = X.shape
    col_to_idx = {col: i for i, col in enumerate(feature_cols)}

    # 基本8チャネル（HRV 4指標を除いたチャネル群）
    hrv_set = set(hrv_cols)
    base_cols = [col for col in feature_cols if col not in hrv_set and col_to_idx[col] < n_channels]

    features_dict: dict[str, np.ndarray] = {}

    # 心拍生値・ベースライン水準（直流成分）の取得ロジック（フォールバック用）
    bpm_fallback_mean = None
    if bpm_base is not None:
        bpm_fallback_mean = np.asarray(bpm_base, dtype=np.float32)
        if bpm_fallback_mean.ndim == 2:
            bpm_fallback_mean = np.nanmean(bpm_fallback_mean, axis=1)
    elif meta_df is not None:
        for candidate_col in ["bpm_mean", "bpm_raw_mean", "bpm_base_10s", "bpm_base"]:
            if candidate_col in meta_df.columns:
                bpm_fallback_mean = meta_df[candidate_col].values.astype(np.float32)
                break

    # 1. 各基本特徴量チャネルの平均値 (Mean) [8次元]
    for col in base_cols:
        idx = col_to_idx[col]
        if col in ("bpm", "bpm_diff"):
            # 心拍チャネル: ウィンドウ内の生心拍数平均 (bpm_mean) を算出
            bpm_vals = X[:, :, idx]
            # X が bpm_diff (平均がほぼ0) の場合は meta_df 側の生平均でフォールバック
            if col == "bpm_diff" or (np.nanmean(np.abs(bpm_vals)) < 15.0 and bpm_fallback_mean is not None):
                features_dict["bpm_mean"] = bpm_fallback_mean if bpm_fallback_mean is not None else np.nanmean(bpm_vals, axis=1)
            else:
                features_dict["bpm_mean"] = np.nanmean(bpm_vals, axis=1)
        else:
            features_dict[f"{col}_mean"] = np.nanmean(X[:, :, idx], axis=1)

    # 2. 各基本特徴量チャネルの標準偏差 (Std) [8次元]
    for col in base_cols:
        idx = col_to_idx[col]
        if col in ("bpm", "bpm_diff"):
            features_dict["bpm_std"] = np.nanstd(X[:, :, idx], axis=1)
        else:
            features_dict[f"{col}_std"] = np.nanstd(X[:, :, idx], axis=1)

    # 3. 移動量の分散 (Movement Variance) [4次元]
    # (A) 頭部姿勢の動きの分散
    head_indices = [col_to_idx[c] for c in head_pose_cols if c in col_to_idx and col_to_idx[c] < n_channels]
    head_vars = [np.nanvar(X[:, :, idx], axis=1) for idx in head_indices]
    if head_vars:
        head_movement_var = np.sum(head_vars, axis=0)
    else:
        head_movement_var = np.zeros(n_samples)

    # (B) 視線の動きの分散
    gaze_indices = [col_to_idx[c] for c in gaze_cols if c in col_to_idx and col_to_idx[c] < n_channels]
    gaze_vars = [np.nanvar(X[:, :, idx], axis=1) for idx in gaze_indices]
    if gaze_vars:
        gaze_movement_var = np.sum(gaze_vars, axis=0)
    else:
        gaze_movement_var = np.zeros(n_samples)

    # モードに応じた移動量分散特徴量の登録
    if mode in ("standard_20d", "standard_24d"):
        # 8(mean) + 8(std) + 4(movement var) + 4(HRV) = 24次元
        features_dict["head_movement_var"] = head_movement_var
        features_dict["gaze_movement_var"] = gaze_movement_var
        if "head_pitch_diff" in col_to_idx and col_to_idx["head_pitch_diff"] < n_channels:
            features_dict["head_pitch_diff_var"] = np.nanvar(
                X[:, :, col_to_idx["head_pitch_diff"]], axis=1
            )
        if "head_yaw_diff" in col_to_idx and col_to_idx["head_yaw_diff"] < n_channels:
            features_dict["head_yaw_diff_var"] = np.nanvar(
                X[:, :, col_to_idx["head_yaw_diff"]], axis=1
            )
    else:
        # "all_variances" モード (総合 + 各軸すべて)
        features_dict["head_movement_var"] = head_movement_var
        features_dict["gaze_movement_var"] = gaze_movement_var
        for c, idx in zip(head_pose_cols, head_indices):
            features_dict[f"{c}_var"] = np.nanvar(X[:, :, idx], axis=1)
        for c, idx in zip(gaze_cols, gaze_indices):
            features_dict[f"{c}_var"] = np.nanvar(X[:, :, idx], axis=1)

    # 4. HRV 主要4特徴量 (rmssd, sdnn, sd2_sd1, mean_pp) [4次元]
    fallback_hrv_dict = None
    for h_col in hrv_cols:
        if h_col in col_to_idx and col_to_idx[h_col] < n_channels:
            features_dict[h_col] = np.nanmean(X[:, :, col_to_idx[h_col]], axis=1)
        elif meta_df is not None and h_col in meta_df.columns and len(meta_df) == n_samples:
            features_dict[h_col] = meta_df[h_col].values.astype(np.float32)
        else:
            if fallback_hrv_dict is None:
                fallback_hrv_dict = _compute_hrv_from_subject_csvs(meta_df, hrv_cols, n_samples) or {}
            features_dict[h_col] = fallback_hrv_dict.get(
                h_col, np.zeros(n_samples, dtype=np.float32)
            )

    # 2次元配列と特徴量名リストに整形
    stat_feature_names = list(features_dict.keys())
    stat_cols_list = [features_dict[name].reshape(-1, 1) for name in stat_feature_names]
    stat_matrix = np.hstack(stat_cols_list).astype(np.float32)

    # 欠損値（NaN）のチェックと補完
    if np.isnan(stat_matrix).any():
        log.warning("⚠️ 物理統計量に NaN が検出されたため、0で補完します。")
        stat_matrix = np.nan_to_num(stat_matrix, nan=0.0)

    return stat_matrix, stat_feature_names
