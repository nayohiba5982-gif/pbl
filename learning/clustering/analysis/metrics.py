from pathlib import Path
import numpy as np
import pandas as pd

from config_loader import log


# デフォルトの特徴量カラム順（preprocess パイプライン準拠）
DEFAULT_FEATURE_COLS = [
    "ear_ratio",
    "head_pitch_diff",
    "head_yaw_diff",
    "head_roll_diff",
    "gaze_v_deg_diff",
    "gaze_h_deg_diff",
    "bvp_selected",
    "bpm",
]


def compute_physical_metrics_from_windows(
    X: np.ndarray,
    feature_cols: list[str],
    bpm_col: str = "bpm",
    ear_col: str = "ear_ratio",
    head_pose_cols: list[str] | None = None,
    gaze_cols: list[str] | None = None,
    meta_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    各ウィンドウ時系列配列 X [N, T, C] から、
    心拍数 (BPM)、瞬目比率 (EAR)、頭部姿勢の動き (分散)、視線の動き (分散) を算出します。
    """
    if head_pose_cols is None:
        head_pose_cols = ["head_pitch_diff", "head_yaw_diff", "head_roll_diff"]
    if gaze_cols is None:
        gaze_cols = ["gaze_v_deg_diff", "gaze_h_deg_diff"]

    col_to_idx = {col: i for i, col in enumerate(feature_cols)}
    n_samples = X.shape[0]

    metrics: dict[str, np.ndarray] = {}

    # 1. 心拍数 (BPM): ウィンドウ内平均（ベースライン水準） & 標準偏差（変動成分）
    actual_bpm_col = bpm_col if bpm_col in col_to_idx else ("bpm_diff" if "bpm_diff" in col_to_idx else None)
    if actual_bpm_col is not None:
        idx = col_to_idx[actual_bpm_col]
        metrics["bpm_std"] = np.nanstd(X[:, :, idx], axis=1)

        # ベースライン水準の抽出
        if meta_df is not None and "bpm_base_10s" in meta_df.columns and len(meta_df) == n_samples:
            metrics["bpm_mean"] = meta_df["bpm_base_10s"].values.astype(np.float32)
        elif "bpm_base_10s" in col_to_idx:
            metrics["bpm_mean"] = np.nanmean(X[:, :, col_to_idx["bpm_base_10s"]], axis=1)
        elif actual_bpm_col != "bpm_diff":
            metrics["bpm_mean"] = np.nanmean(X[:, :, idx], axis=1)
        else:
            metrics["bpm_mean"] = np.nanmean(X[:, :, idx], axis=1)
    else:
        log.warning(f"'{bpm_col}' (または 'bpm_diff') が特徴量一覧に見つかりません。")

    # 2. 瞬目比率 (EAR): ウィンドウ内平均 & 標準偏差
    if ear_col in col_to_idx:
        idx = col_to_idx[ear_col]
        metrics["ear_mean"] = np.nanmean(X[:, :, idx], axis=1)
        metrics["ear_std"] = np.nanstd(X[:, :, idx], axis=1)
    else:
        log.warning(f"'{ear_col}' が特徴量一覧に見つかりません。")

    # 3. 頭部の動き（姿勢角度の時間分散）
    head_indices = [col_to_idx[c] for c in head_pose_cols if c in col_to_idx]
    if head_indices:
        head_vars = [np.nanvar(X[:, :, idx], axis=1) for idx in head_indices]
        metrics["head_movement_var"] = np.sum(head_vars, axis=0)
        for c, h_idx in zip(head_pose_cols, head_indices):
            metrics[f"{c}_var"] = np.nanvar(X[:, :, h_idx], axis=1)
    else:
        log.warning("頭部姿勢カラムが見つかりません。")

    # 4. 視線の動き（視線角度の時間分散）
    gaze_indices = [col_to_idx[c] for c in gaze_cols if c in col_to_idx]
    if gaze_indices:
        gaze_vars = [np.nanvar(X[:, :, idx], axis=1) for idx in gaze_indices]
        metrics["gaze_movement_var"] = np.sum(gaze_vars, axis=0)
        for c, g_idx in zip(gaze_cols, gaze_indices):
            metrics[f"{c}_var"] = np.nanvar(X[:, :, g_idx], axis=1)
    else:
        log.warning("視線カラムが見つかりません。")

    df_phys = pd.DataFrame(metrics, index=np.arange(n_samples))
    return df_phys


def attach_physical_metrics_to_df(
    split_dfs: dict[str, pd.DataFrame],
    X_all: np.ndarray | None,
    split_indices: dict | None,
    feature_cols: list[str],
    bpm_col: str = "bpm",
    ear_col: str = "ear_ratio",
    head_pose_cols: list[str] | None = None,
    gaze_cols: list[str] | None = None,
    meta_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    メタデータ DataFrame に物理指標（BPM, EAR, 頭部・視線分散）を紐付け、
    解析用 DataFrame を作成します。
    """
    if X_all is not None:
        df_phys_all = compute_physical_metrics_from_windows(
            X_all,
            feature_cols=feature_cols,
            bpm_col=bpm_col,
            ear_col=ear_col,
            head_pose_cols=head_pose_cols,
            gaze_cols=gaze_cols,
            meta_df=meta_raw,
        )

        for split_name, df_meta in split_dfs.items():
            phys_subset = pd.DataFrame()
            if (
                split_indices is not None
                and f"{split_name}_idx" in split_indices
                and len(split_indices[f"{split_name}_idx"]) == len(df_meta)
            ):
                idx_arr = split_indices[f"{split_name}_idx"]
                phys_subset = df_phys_all.iloc[idx_arr].reset_index(drop=True)
            elif len(df_meta) == len(df_phys_all):
                phys_subset = df_phys_all.copy().reset_index(drop=True)
            elif (
                "username" in df_meta.columns
                and meta_raw is not None
                and "username" in meta_raw.columns
                and len(meta_raw) == len(df_phys_all)
            ):
                # ユーザーID順およびサンプル数に合わせて被験者単位で正確にマッチング
                df_phys_with_user = df_phys_all.copy()
                df_phys_with_user["_u_str"] = meta_raw["username"].astype(str).values
                matched_parts = []
                for u in df_meta["username"].astype(str).unique():
                    u_cnt = int((df_meta["username"].astype(str) == u).sum())
                    u_rows = df_phys_with_user[df_phys_with_user["_u_str"] == u].iloc[:u_cnt]
                    matched_parts.append(u_rows.drop(columns=["_u_str"]))
                if matched_parts:
                    cand_df = pd.concat(matched_parts, ignore_index=True)
                    if len(cand_df) == len(df_meta):
                        phys_subset = cand_df

            if len(phys_subset) == len(df_meta):
                for col in phys_subset.columns:
                    if col not in df_meta.columns:
                        df_meta[col] = phys_subset[col].values

            # 既に df_meta に bpm_base_mean / ear_ratio_mean が存在する場合はエイリアス列を補完
            if "bpm_mean" not in df_meta.columns and "bpm_base_mean" in df_meta.columns:
                df_meta["bpm_mean"] = df_meta["bpm_base_mean"].values
            if "ear_mean" not in df_meta.columns and "ear_ratio_mean" in df_meta.columns:
                df_meta["ear_mean"] = df_meta["ear_ratio_mean"].values

    all_dfs = []
    for s_name, df in split_dfs.items():
        df_copy = df.copy()
        df_copy["split"] = s_name
        all_dfs.append(df_copy)

    df_combined = pd.concat(all_dfs, ignore_index=True)
    return df_combined
