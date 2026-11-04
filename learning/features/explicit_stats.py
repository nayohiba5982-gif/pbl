"""
各ウィンドウ時系列から人間が解釈可能な明示的物理統計量を算出するモジュール (Explicit Statistics Aggregation)
(learning/features/explicit_stats.py)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from config_loader import log
except ImportError:
    import logging
    log = logging.getLogger("explicit_statistics")

try:
    from .constants import DEFAULT_FEATURE_COLS, DEFAULT_HEAD_POSE_COLS, DEFAULT_GAZE_COLS
except ImportError:
    from constants import DEFAULT_FEATURE_COLS, DEFAULT_HEAD_POSE_COLS, DEFAULT_GAZE_COLS


def compute_explicit_statistics(
    X: np.ndarray,
    feature_cols: list[str] | None = None,
    head_pose_cols: list[str] | None = None,
    gaze_cols: list[str] | None = None,
    mode: str = "standard_20d",
    meta_df: pd.DataFrame | None = None,
    bpm_base: np.ndarray | None = None,
    baseline_mode: str = "absolute",
) -> tuple[np.ndarray, list[str]]:
    """
    時系列ウィンドウ配列 X [N, T, C] から、人間が解釈可能な明示的物理統計量を集約・算出します。

    Parameters:
        X (np.ndarray): 形状 [N_samples, Time_steps, Channels] の3次元配列
        feature_cols (list[str] | None): チャネルごとの名前リスト (例: 8チャネル)
        head_pose_cols (list[str] | None): 頭部姿勢のカラム名
        gaze_cols (list[str] | None): 視線のカラム名
        mode (str):
            - "standard_20d": 平均(8) + 標準偏差(8) + 移動量分散(4) = 20次元
            - "all_variances": 平均(8) + 標準偏差(8) + 移動量分散全種(7) = 23次元
        meta_df (pd.DataFrame | None): 各ウィンドウのメタデータ (bpm_base_10s 等を含む)
        bpm_base (np.ndarray | None): 明示的に渡された局所心拍ベースライン水準配列 [N]
        baseline_mode (str): "absolute" (そのままのBPM水準) または "relative_to_first" (先頭基準窓との差分)

    Returns:
        stat_matrix (np.ndarray): 形状 [N_samples, D_stat] (例: [N, 20]) の2次元配列
        stat_feature_names (list[str]): 特徴量名のリスト
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS
    if head_pose_cols is None:
        head_pose_cols = DEFAULT_HEAD_POSE_COLS
    if gaze_cols is None:
        gaze_cols = DEFAULT_GAZE_COLS

    n_samples, time_steps, n_channels = X.shape
    if n_channels == 7 and "bvp_selected" in feature_cols:
        # vital が HR のみ (bvp_selected 除外) の 7 チャネル時系列に対応
        feature_cols = [c for c in feature_cols if c != "bvp_selected"]

    col_to_idx = {col: i for i, col in enumerate(feature_cols)}

    features_dict: dict[str, np.ndarray] = {}

    # 心拍ベースライン水準（直流成分）の取得ロジック（20次元統計量の情報消失バグ回避）
    bpm_baseline_arr = None
    if bpm_base is not None:
        bpm_baseline_arr = np.asarray(bpm_base, dtype=np.float32)
        if bpm_baseline_arr.ndim == 2:
            bpm_baseline_arr = np.nanmean(bpm_baseline_arr, axis=1)
    elif meta_df is not None:
        for candidate_col in ["bpm_base_10s", "bpm_base", "bpm_raw_mean", "bpm_mean"]:
            if candidate_col in meta_df.columns:
                bpm_baseline_arr = meta_df[candidate_col].values.astype(np.float32)
                break
    elif "bpm_base_10s" in col_to_idx:
        bpm_baseline_arr = np.nanmean(X[:, :, col_to_idx["bpm_base_10s"]], axis=1)
    elif "bpm" in col_to_idx:
        bpm_baseline_arr = np.nanmean(X[:, :, col_to_idx["bpm"]], axis=1)

    if bpm_baseline_arr is None:
        # メタデータ等にベースラインがない場合のフォールバック
        bpm_baseline_arr = np.zeros(n_samples, dtype=np.float32)
    elif baseline_mode == "relative_to_first" and len(bpm_baseline_arr) > 0:
        # 先頭基準窓との差分（タスク中の心拍ドリフト・疲労応答）
        first_val = bpm_baseline_arr[0]
        bpm_baseline_arr = bpm_baseline_arr - first_val

    # 1. 各特徴量チャネルの平均値 (Mean) [8次元]
    for col in feature_cols:
        if col in ("bpm_diff", "bpm"):
            # 心拍チャネル: ゼロ平均系列から mean を取る情報消失バグを回避し、
            # 局所ベースライン水準（直流成分）を割り当てる
            features_dict["bpm_base_mean"] = bpm_baseline_arr
        else:
            idx = col_to_idx[col]
            features_dict[f"{col}_mean"] = np.nanmean(X[:, :, idx], axis=1)

    # 2. 各特徴量チャネルの標準偏差 (Std) [8次元]
    # bpm_diff からは「分散・標準偏差」などの変動成分（AC成分）のみを算出
    for col in feature_cols:
        idx = col_to_idx[col]
        features_dict[f"{col}_std"] = np.nanstd(X[:, :, idx], axis=1)

    # 3. 移動量の分散 (Movement Variance)
    # (A) 頭部姿勢の動きの分散
    head_indices = [col_to_idx[c] for c in head_pose_cols if c in col_to_idx]
    head_vars = [np.nanvar(X[:, :, idx], axis=1) for idx in head_indices]
    if head_vars:
        head_movement_var = np.sum(head_vars, axis=0)
    else:
        head_movement_var = np.zeros(n_samples)

    # (B) 視線の動きの分散
    gaze_indices = [col_to_idx[c] for c in gaze_cols if c in col_to_idx]
    gaze_vars = [np.nanvar(X[:, :, idx], axis=1) for idx in gaze_indices]
    if gaze_vars:
        gaze_movement_var = np.sum(gaze_vars, axis=0)
    else:
        gaze_movement_var = np.zeros(n_samples)

    # モードに応じた移動量分散特徴量の登録
    if mode == "standard_20d":
        # 8(mean) + 8(std) + 4(movement var) = 20次元
        features_dict["head_movement_var"] = head_movement_var
        features_dict["gaze_movement_var"] = gaze_movement_var
        if "head_pitch_diff" in col_to_idx:
            features_dict["head_pitch_diff_var"] = np.nanvar(
                X[:, :, col_to_idx["head_pitch_diff"]], axis=1
            )
        if "head_yaw_diff" in col_to_idx:
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

    # 2次元配列と特徴量名リストに整形
    stat_feature_names = list(features_dict.keys())
    stat_cols_list = [features_dict[name].reshape(-1, 1) for name in stat_feature_names]
    stat_matrix = np.hstack(stat_cols_list).astype(np.float32)

    # 欠損値（NaN）のチェックと補完
    if np.isnan(stat_matrix).any():
        log.warning("⚠️ 物理統計量に NaN が検出されたため、0で補完します。")
        stat_matrix = np.nan_to_num(stat_matrix, nan=0.0)

    return stat_matrix, stat_feature_names
