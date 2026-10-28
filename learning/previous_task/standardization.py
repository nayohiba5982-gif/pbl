"""
GMM 入力前 被験者別 (Subject-wise) Zスコア標準化モジュール
(learning/previous_task/standardization.py)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .utils import standardize_subject_wise
except ImportError:
    from utils import standardize_subject_wise


def run_subject_standardization(
    d_dir: Path,
    resolve_split_files_fn: Callable[[str, str], Tuple[Path | None, Path | None, Path | None]],
    apply_subject_norm: bool,
    group_col: str,
    n_components_visual: int,
    ws: str,
    save_plots: bool,
    log,
) -> bool:
    """
    PLS 特徴量に残存する被験者ベースライン（個体差）を相殺するため、
    GMM 入力直前に被験者 (username) ごとに平均0・標準偏差1へ Zスコア標準化して保存します。
    """
    log.info("=" * 70)
    log.info(f"▶ GMM 入力前 被験者別 (Subject-wise) Zスコア標準化開始 [Window: {ws}]")
    log.info("=" * 70)

    splits = ["train", "val", "test"]
    std_stats: list[dict] = []
    any_processed = False
    n_vis = n_components_visual

    for s_name in splits:
        fused_path = d_dir / f"pls_{s_name}_fused.npy"
        _, _, meta_path = resolve_split_files_fn(ws, s_name)
        if meta_path is None:
            cand_m = d_dir / f"meta_{s_name}.csv"
            if cand_m.exists():
                meta_path = cand_m

        if not fused_path.exists() or meta_path is None or not meta_path.exists():
            log.warning(f"⚠️ [{ws} | {s_name}] 標準化対象ファイルが見つかりません: {fused_path.name}")
            continue

        X_fused = np.load(fused_path)
        meta_df = pd.read_csv(meta_path)

        if apply_subject_norm:
            X_std, _, split_stats = standardize_subject_wise(
                X_fused, meta_df, group_col=group_col
            )
            for row in split_stats:
                std_stats.append({"split": s_name, **row})
        else:
            X_std = X_fused.astype(np.float32)

        save_fused_path = d_dir / f"pls_{s_name}_fused_standardized.npy"
        np.save(save_fused_path, X_std)

        vis_dim = n_vis if n_vis < X_std.shape[1] else X_std.shape[1] // 2
        np.save(d_dir / f"pls_{s_name}_visual_standardized.npy", X_std[:, :vis_dim])
        np.save(d_dir / f"pls_{s_name}_vital_standardized.npy", X_std[:, vis_dim:])

        user_col = group_col if group_col in meta_df.columns else "subject_id"
        n_users = meta_df[user_col].nunique() if user_col in meta_df.columns else 0
        log.info(
            f"✓ [{s_name}] 標準化完了: {n_users} 名 (サンプル数: {len(X_std)}) ➔ 保存: {save_fused_path.name}"
        )
        any_processed = True

    if not any_processed:
        return False

    if std_stats:
        df_stats = pd.DataFrame(std_stats)
        stats_csv_path = d_dir / "pls_subject_normalization_stats.csv"
        df_stats.to_csv(stats_csv_path, index=False)
        log.info(f"✓ 被験者別 Zスコア標準化サマリーを保存しました: {stats_csv_path}")

    # 可視化: 標準化前後の特徴量分布比較 (Train)
    tr_fused_path = d_dir / "pls_train_fused.npy"
    tr_std_path = d_dir / "pls_train_fused_standardized.npy"
    tr_meta_path = d_dir / "meta_train.csv"
    if save_plots and tr_fused_path.exists() and tr_std_path.exists() and tr_meta_path.exists():
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), dpi=120)
        X_raw = np.load(tr_fused_path)
        X_norm = np.load(tr_std_path)
        meta_tr = pd.read_csv(tr_meta_path)
        u_col = group_col if group_col in meta_tr.columns else "subject_id"
        u_tr = meta_tr[u_col].values

        for u in np.unique(u_tr)[:5]:  # 代表5名の分布
            mask = u_tr == u
            axes[0].hist(X_raw[mask, 0], bins=20, alpha=0.5, label=f"User {str(u)[:6]}")
            axes[1].hist(X_norm[mask, 0], bins=20, alpha=0.5, label=f"User {str(u)[:6]}")

        axes[0].set_title(
            f"Before Subject-wise Normalization (PLS Dim 1) [{ws}]\n(Showing Individual Baseline Shifts)",
            fontsize=11,
            fontweight="bold",
        )
        axes[0].set_xlabel("Feature Value")
        axes[0].set_ylabel("Count")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, linestyle="--", alpha=0.5)

        axes[1].set_title(
            f"After Subject-wise Normalization (PLS Dim 1) [{ws}]\n(Centered at Mean=0, Std=1 per User)",
            fontsize=11,
            fontweight="bold",
        )
        axes[1].set_xlabel("Normalized Feature Value")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        norm_plot_path = d_dir / "pls_subject_normalization_check.png"
        plt.savefig(norm_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"✓ 標準化前後の検証プロットを保存しました: {norm_plot_path}")

    return True
