"""
PLS 累積説明分散比 (R²) エルボー分析モジュール
(learning/previous_task/elbow_analysis.py)
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression


def run_pls_elbow_analysis(
    d_dir: Path,
    vis_emb_path: Path | None,
    bio_emb_path: Path | None,
    meta_path: Path | None,
    candidate_components: list[int],
    ws: str,
    save_plots: bool,
    log,
) -> pd.DataFrame | None:
    """
    Train 被験者の埋め込み (emb_train_visual.npy, emb_train_vital.npy) のみを用いて
    成分数候補 (例: 1〜6) の寄与率 (累積 R²) を評価・プロットします。
    """
    if not (vis_emb_path and bio_emb_path and meta_path):
        log.warning(
            f"⚠️ [{ws}] Train 埋め込みファイルまたは meta_train.csv が見つかりません: {d_dir}\n"
            "  先に Google Colab (previous_task.ipynb) の Cell 1〜5 を実行して emb_*.npy を配置してください。"
        )
        return None

    emb_vis = np.load(vis_emb_path)
    emb_bio = np.load(bio_emb_path)
    meta_df = pd.read_csv(meta_path)
    label_col = "label" if "label" in meta_df.columns else "label_mean"
    y = meta_df[label_col].values.astype(float)

    records = []
    prev_r2_vis = 0.0
    prev_r2_bio = 0.0

    max_k_vis = min(emb_vis.shape[0] - 1, emb_vis.shape[1])
    max_k_bio = min(emb_bio.shape[0] - 1, emb_bio.shape[1])

    for k in candidate_components:
        if k > max_k_vis or k > max_k_bio:
            continue
        pls_v = PLSRegression(n_components=k, scale=True).fit(emb_vis, y)
        pls_b = PLSRegression(n_components=k, scale=True).fit(emb_bio, y)
        r2_v = float(pls_v.score(emb_vis, y))
        r2_b = float(pls_b.score(emb_bio, y))

        delta_v = r2_v - prev_r2_vis
        delta_b = r2_b - prev_r2_bio
        prev_r2_vis = r2_v
        prev_r2_bio = r2_b

        records.append({
            "n_components": k,
            "r2_visual(%)": round(r2_v * 100, 2),
            "delta_vis(%)": round(delta_v * 100, 2),
            "r2_vital(%)": round(r2_b * 100, 2),
            "delta_bio(%)": round(delta_b * 100, 2),
        })

    res_df = pd.DataFrame(records)
    log.info("=" * 70)
    log.info(f"📊 【Train データ PLS 累積説明分散比 (R²) エルボー分析サマリー】 [Window: {ws}]")
    log.info("=" * 70)
    for _, row in res_df.iterrows():
        log.info(
            f"  k={int(row['n_components'])}: "
            f"Visual R²={row['r2_visual(%)']:6.2f}% (+{row['delta_vis(%)']:5.2f}%) | "
            f"Vital R²={row['r2_vital(%)']:6.2f}% (+{row['delta_bio(%)']:5.2f}%)"
        )

    elbow_csv_path = d_dir / "pls_variance_explained_elbow_train.csv"
    res_df.to_csv(elbow_csv_path, index=False)

    if save_plots and not res_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
        ks = res_df["n_components"].values
        y_vis = res_df["r2_visual(%)"].values
        y_bio = res_df["r2_vital(%)"].values

        ax.plot(
            ks, y_vis, marker="o", markersize=7, linewidth=2,
            color="#2563eb", label="Visual Stream (X_vis ➔ PLS)"
        )
        ax.plot(
            ks, y_bio, marker="s", markersize=7, linewidth=2,
            color="#ea580c", label="Vital Stream (X_bio ➔ PLS)"
        )

        for k_val, v_val, b_val in zip(ks, y_vis, y_bio):
            ax.annotate(
                f"{v_val:.1f}%", (k_val, v_val),
                textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=8.5, fontweight="bold", color="#1d4ed8"
            )
            ax.annotate(
                f"{b_val:.1f}%", (k_val, b_val),
                textcoords="offset points", xytext=(0, -13),
                ha="center", fontsize=8.5, fontweight="bold", color="#c2410c"
            )

        ax.set_title(f"PLS Cumulative R² Elbow Curve (Train Only) [{ws}]", fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of PLS Components (n_components)", fontsize=10)
        ax.set_ylabel("Variance Explained of Label y (%)", fontsize=10)
        ax.set_xticks(ks)
        y_min = max(0.0, min(float(y_vis.min()), float(y_bio.min())) - 10.0)
        y_max = min(105.0, max(float(y_vis.max()), float(y_bio.max())) + 12.0)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower right")
        plt.tight_layout()

        plot_path = d_dir / "pls_variance_explained_elbow_train.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"✓ エルボープロット図を保存しました: {plot_path}")

    return res_df
