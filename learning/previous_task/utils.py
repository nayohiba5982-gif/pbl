"""
PreviousTask 共通ユーティリティ (ウィンドウ正規化・被験者別標準化)
(learning/previous_task/utils.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ALL_SUPPORTED_WINDOWS = ["5s", "10s", "15s"]


def normalize_window_list(raw_windows: str | list[str] | None, fallback: str = "15s") -> list[str]:
    """
    config.yaml または CLI 引数で指定されたウィンドウサイズ (単一文字列またはリスト) を正規化します。
    """
    if raw_windows is None:
        raw_list = [fallback]
    elif isinstance(raw_windows, str):
        raw_list = [w.strip() for w in raw_windows.replace(",", " ").split() if w.strip()]
    elif isinstance(raw_windows, (list, tuple, set)):
        raw_list = []
        for item in raw_windows:
            for w in str(item).replace(",", " ").split():
                if w.strip():
                    raw_list.append(w.strip())
    else:
        raw_list = [str(raw_windows).strip()]

    if not raw_list:
        raw_list = [fallback]

    if "all" in [w.lower() for w in raw_list]:
        return list(ALL_SUPPORTED_WINDOWS)

    target_windows: list[str] = []
    for w in raw_list:
        w_lower = w.lower()
        if w_lower in ["5", "5s"]:
            target_windows.append("5s")
        elif w_lower in ["10", "10s"]:
            target_windows.append("10s")
        elif w_lower in ["15", "15s"]:
            target_windows.append("15s")
        elif w_lower == "all":
            target_windows.extend(ALL_SUPPORTED_WINDOWS)
        else:
            target_windows.append(w)

    return list(dict.fromkeys(target_windows))


def standardize_subject_wise(
    features: np.ndarray,
    df_meta: pd.DataFrame | None = None,
    group_col: str = "username",
    eps: float = 1e-6,
) -> tuple[np.ndarray, pd.DataFrame | None, list[dict]]:
    """
    PLS 変換後の特徴量を GMM に入力する直前に、
    被験者 (username / subject_id) ごとに平均0・標準偏差1へ Zスコア標準化します。
    """
    arr = np.asarray(features, dtype=np.float32)
    stats_list: list[dict] = []

    if df_meta is None or len(df_meta) != len(arr):
        return arr, df_meta, stats_list

    col = group_col if group_col in df_meta.columns else ("subject_id" if "subject_id" in df_meta.columns else None)
    if col is None:
        return arr, df_meta, stats_list

    normed = np.zeros_like(arr, dtype=np.float32)
    users = df_meta[col].astype(str).values
    unique_u = np.unique(users)

    for u in unique_u:
        u_mask = users == u
        u_feat = arr[u_mask]
        u_mean = np.mean(u_feat, axis=0, keepdims=True)
        u_std = np.std(u_feat, axis=0, keepdims=True)
        u_std_safe = np.where(u_std < eps, 1.0, u_std)

        normed[u_mask] = (u_feat - u_mean) / u_std_safe

        stats_list.append({
            "username": u,
            "n_samples": int(np.sum(u_mask)),
            "mean_before": float(np.round(u_mean.mean(), 3)),
            "std_before": float(np.round(u_std.mean(), 3)),
            "mean_after": float(np.round(normed[u_mask].mean(), 3)),
            "std_after": float(np.round(normed[u_mask].std(), 3)),
        })

    meta_out = df_meta.copy()
    pls_cols = [c for c in meta_out.columns if c.startswith(("vis_pls_", "bio_pls_"))]
    if len(pls_cols) == normed.shape[1]:
        for idx, c_name in enumerate(pls_cols):
            meta_out[c_name] = normed[:, idx]

    return normed, meta_out, stats_list
