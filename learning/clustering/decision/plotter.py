"""
モデル選択メトリクス可視化モジュール (Decision Metrics Plotter)

【責務】
- BIC, AIC, Silhouette スコアの4共分散タイプ比較マルチパネルグラフの描画・保存
"""

from pathlib import Path
import pandas as pd

import sys
_LEARNING_DIR = Path(__file__).resolve().parent.parent.parent
if str(_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(_LEARNING_DIR))

try:
    from config_loader import log
except ImportError:
    import logging
    log = logging.getLogger("clustering.decision.plotter")


def plot_model_selection_metrics(
    df_metrics: pd.DataFrame,
    optimal_k: int,
    optimal_cov: str,
    save_path: Path,
    window_size: str = "10s",
    k_min: int = 2,
    k_max: int = 10,
    covariance_types: list[str] | None = None,
):
    """
    4つの共分散タイプを同時に比較できるマルチパネルグラフ（BIC, AIC, Silhouette）を描画・保存します。
    """
    if covariance_types is None:
        covariance_types = ["full", "tied", "diag", "spherical"]

    try:
        import matplotlib.pyplot as plt

        style_dict = {
            "full": {"color": "#1f77b4", "marker": "o", "label": "full"},
            "tied": {"color": "#ff7f0e", "marker": "s", "label": "tied"},
            "diag": {"color": "#2ca02c", "marker": "^", "label": "diag"},
            "spherical": {"color": "#9467bd", "marker": "D", "label": "spherical"},
        }

        fig, (ax_bic, ax_aic, ax_sil) = plt.subplots(
            nrows=3, ncols=1, figsize=(11, 14), sharex=True
        )

        # 1. BIC
        for cov in covariance_types:
            sub = df_metrics[df_metrics["covariance_type"] == cov]
            st = style_dict.get(cov, {"color": "black", "marker": "x", "label": cov})
            ax_bic.plot(
                sub["k"], sub["bic"], marker=st["marker"], color=st["color"],
                label=st["label"], linewidth=2, markersize=6,
            )

        opt_row = df_metrics[
            (df_metrics["covariance_type"] == optimal_cov) & (df_metrics["k"] == optimal_k)
        ]
        if not opt_row.empty:
            opt_bic = opt_row["bic"].values[0]
            ax_bic.scatter(
                [optimal_k], [opt_bic], color="crimson", s=180, zorder=5,
                edgecolors="black", linewidth=1.5, marker="*",
                label=f"Optimal (k={optimal_k}, {optimal_cov})",
            )
            ax_bic.annotate(
                f"Best BIC: k={optimal_k} ({optimal_cov})\n{opt_bic:.1f}",
                xy=(optimal_k, opt_bic), xytext=(optimal_k + 0.3, opt_bic),
                arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6),
            )

        ax_bic.set_ylabel("BIC (Lower is better)", fontsize=11, fontweight="bold")
        ax_bic.set_title("Bayesian Information Criterion (BIC) by Covariance Type", fontsize=12)
        ax_bic.grid(True, linestyle=":", alpha=0.6)
        ax_bic.legend(loc="best", frameon=True, fontsize=10)

        # 2. AIC
        for cov in covariance_types:
            sub = df_metrics[df_metrics["covariance_type"] == cov]
            st = style_dict.get(cov, {"color": "black", "marker": "x", "label": cov})
            ax_aic.plot(
                sub["k"], sub["aic"], marker=st["marker"], color=st["color"],
                label=st["label"], linewidth=2, markersize=6,
            )

        ax_aic.set_ylabel("AIC (Lower is better)", fontsize=11, fontweight="bold")
        ax_aic.set_title("Akaike Information Criterion (AIC) by Covariance Type", fontsize=12)
        ax_aic.grid(True, linestyle=":", alpha=0.6)
        ax_aic.legend(loc="best", frameon=True, fontsize=10)

        # 3. Silhouette
        for cov in covariance_types:
            sub = df_metrics[df_metrics["covariance_type"] == cov]
            st = style_dict.get(cov, {"color": "black", "marker": "x", "label": cov})
            ax_sil.plot(
                sub["k"], sub["silhouette"], marker=st["marker"], color=st["color"],
                label=st["label"], linewidth=2, markersize=6,
            )

        ax_sil.set_xlabel("Number of Clusters (k)", fontsize=12, fontweight="bold")
        ax_sil.set_ylabel("Silhouette Score (Higher is better)", fontsize=11, fontweight="bold")
        ax_sil.set_title("Silhouette Score by Covariance Type", fontsize=12)
        ax_sil.set_xticks(range(k_min, k_max + 1))
        ax_sil.grid(True, linestyle=":", alpha=0.6)
        ax_sil.legend(loc="best", frameon=True, fontsize=10)

        fig.suptitle(
            f"GMM Model Selection Across 4 Covariance Types (Window: {window_size})",
            fontsize=14, fontweight="bold", y=0.995,
        )
        fig.tight_layout()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        log.info(f"📊 4共分散タイプの比較可視化グラフを保存しました: {save_path}")

    except Exception as e:
        log.warning(f"グラフ描画・保存中にエラーが発生しました: {e}")
