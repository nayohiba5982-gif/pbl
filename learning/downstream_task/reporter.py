"""
LOSO-CV 評価レポート & 可視化グラフ出力モジュール
(learning/downstream_task/reporter.py)
"""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def plot_model_comparison(df_summary: pd.DataFrame, output_dir: Path, log) -> Path:
    """モデル間の主要指標比較プロット (6 パネル & Fold 間標準偏差エラーバー付き)"""
    fig_path = output_dir / "loso_models_comparison.png"
    models = df_summary["model"].tolist()

    metrics_info = [
        ("mae", "mae_std", "MAE (Lower is better)", True),
        ("rmse", "rmse_std", "RMSE (Lower is better)", True),
        ("pearson_r", "pearson_r_std", "Pearson r (Higher is better)", False),
        ("roc_auc", "roc_auc_std", "ROC-AUC (Higher is better)", False),
        ("pr_auc", "pr_auc_std", "PR-AUC (Higher is better)", False),
        ("balanced_acc", "balanced_acc_std", "Balanced Acc (Higher is better)", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), dpi=130)
    axes = axes.flatten()
    colors = ["#2563eb", "#d97706", "#059669", "#7c3aed", "#db2777", "#4b5563", "#0891b2", "#65a30d"]

    for idx, (m_key, std_key, title, lower_better) in enumerate(metrics_info):
        ax = axes[idx]
        if m_key in df_summary.columns:
            vals = df_summary[m_key].values
            yerrs = df_summary[std_key].values if std_key in df_summary.columns else None

            bars = ax.bar(
                models,
                vals,
                yerr=yerrs,
                capsize=4,
                color=colors[: len(models)],
                alpha=0.85,
                edgecolor="black",
            )
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_ylabel(m_key, fontsize=10.5)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9.5)

            for i, bar in enumerate(bars):
                yval = bar.get_height()
                if not np.isnan(yval):
                    std_str = f"±{yerrs[i]:.2f}" if yerrs is not None and not np.isnan(yerrs[i]) else ""
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        yval + 0.015 if yval >= 0 else yval - 0.03,
                        f"{yval:.3f}\n{std_str}".strip(),
                        ha="center",
                        va="bottom" if yval >= 0 else "top",
                        fontsize=8.5,
                        fontweight="bold",
                    )

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)
    log.info(f"📊 モデル比較プロットを保存しました: {fig_path}")
    return fig_path


def generate_visualizations(
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold_df: pd.DataFrame,
    model_name: str,
    mae: float,
    rmse: float,
    r2: float,
    r_val: float,
    acc: float,
    balanced_acc: float,
    auc: float,
    pr_auc: float,
    log,
) -> Path:
    """可視化グラフ (散布図, 被験者別性能棒グラフ, 混同行列) の保存"""
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), dpi=130)

    # 1. 散布図 (True vs Pred)
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.25, color="#2563eb", edgecolors="none", s=20)
    ax1.plot([0, 1], [0, 1], "--", color="#ef4444", linewidth=1.5, label="Ideal (y = x)")
    ax1.set_xlabel("True Label (y)", fontsize=11)
    ax1.set_ylabel(f"OOF Predicted ({model_name})", fontsize=11)
    ax1.set_title(
        f"True vs OOF Prediction\nMAE={mae:.3f}, RMSE={rmse:.3f}, r={r_val:.3f}, R²={r2:.3f}",
        fontsize=11.5,
        fontweight="bold",
    )
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    # 2. 被験者別 MAE & 相関
    ax2 = axes[1]
    users = [str(u)[:8] for u in fold_df["test_user"].values]
    x_indices = np.arange(len(users))
    width = 0.35
    ax2.bar(x_indices - width / 2, fold_df["mae"], width=width, label="MAE", color="#f97316")
    ax2.bar(x_indices + width / 2, fold_df["pearson_r"].fillna(0.0), width=width, label="Pearson r", color="#3b82f6")
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(users, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Metric Value", fontsize=11)
    ax2.set_title("Performance by Test Subject", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax2.legend()

    # 3. 混同行列 (二値化 0.5)
    ax3 = axes[2]
    y_true_b = (y_true >= 0.5).astype(int)
    y_pred_b = (y_pred >= 0.5).astype(int)
    cm = confusion_matrix(y_true_b, y_pred_b)
    im = ax3.imshow(cm, cmap="Blues", interpolation="nearest")
    for i in range(2):
        for j in range(2):
            ax3.text(
                j, i, f"{cm[i, j]:,d}",
                ha="center", va="center",
                color="black" if cm[i, j] < cm.max() / 2 else "white",
                fontsize=12, fontweight="bold"
            )
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Relax (0)", "Focus (1)"])
    ax3.set_yticklabels(["Relax (0)", "Focus (1)"])
    ax3.set_xlabel("Predicted", fontsize=11)
    ax3.set_ylabel("True", fontsize=11)
    
    auc_str = f"{auc:.3f}" if not np.isnan(auc) else "N/A"
    pr_str = f"{pr_auc:.3f}" if not np.isnan(pr_auc) else "N/A"
    bacc_str = f"{balanced_acc:.1%}" if not np.isnan(balanced_acc) else "N/A"
    ax3.set_title(
        f"Confusion Matrix (Th=0.5)\nAcc={acc:.1%}, B-Acc={bacc_str} | AUC={auc_str}, PR-AUC={pr_str}",
        fontsize=11.5,
        fontweight="bold",
    )

    plt.tight_layout()
    plot_path = output_dir / f"loso_{model_name}_evaluation_plots.png"
    plt.savefig(plot_path)
    plt.close(fig)
    log.info(f"📊 評価プロットを保存しました: {plot_path}")
    return plot_path


def save_markdown_report(
    output_dir: Path,
    window_size: str,
    model_name: str,
    n_users: int,
    mae: float,
    rmse: float,
    r2: float,
    r_val: float,
    rho_val: float,
    acc: float,
    balanced_acc: float,
    auc: float,
    pr_auc: float,
    f1_macro: float,
    stats_dict: dict[str, tuple[float, float]],
    fold_df: pd.DataFrame,
    log,
) -> Path:
    """Markdown 評価レポートの出力 (総合 OOF スコア & Fold 間 Mean ± Std 併記版)"""
    report_path = output_dir / f"loso_{model_name}_report.md"

    def fmt_ms(key: str, is_pct: bool = False) -> str:
        mean_v, std_v = stats_dict.get(key, (np.nan, np.nan))
        if np.isnan(mean_v):
            return "N/A"
        if is_pct:
            return f"{mean_v:.4f} ± {std_v:.4f} ({mean_v:.1%})"
        return f"{mean_v:.4f} ± {std_v:.4f}"

    lines = [
        f"# 🛡️ LOSO-CV 回帰モデル評価レポート [{model_name.upper()} | Window: {window_size}]",
        "",
        "## 1. 概要 & 評価プロトコル",
        "- **評価手法**: Leave-One-Subject-Out Cross-Validation (LOSO-CV) - 完全データリークフリー",
        f"- **対象被験者数**: {n_users} 名 (計 {len(fold_df)} Folds)",
        f"- **回帰モデル**: `{model_name}` (Regressor, 目的変数: GMMソフト確率ラベル ∈ [0, 1])",
        "- **Fold内処理**: Train被験者のみで PLS fit ➔ 被験者別 Zスコア標準化 ➔ Train被験者のみで GMM (k=2) fit ➔ Train被験者のみで Late Fusion 学習 ➔ Test被験者推論",
        "",
        "## 2. 全体 Out-Of-Fold (OOF) 総合性能 & Fold間ばらつき (Mean ± Std)",
        "| 指標区分 | 評価指標 | 総合 OOF スコア | Fold 間平均 ± 標準偏差 (Mean ± Std) | 理想値・備考 |",
        "|---|---|---|---|---|",
        f"| **回帰性能** | **MAE (平均絶対誤差)** | **{mae:.4f}** | {fmt_ms('mae')} | 0.0 (小さいほど優秀) |",
        f"| | **RMSE (二乗平均平方根誤差)** | **{rmse:.4f}** | {fmt_ms('rmse')} | 0.0 (小さいほど優秀) |",
        f"| | **R² Score (決定係数)** | **{r2:.4f}** | {fmt_ms('r2')} | 1.0 (正値ほど説明力高) |",
        f"| | **Pearson 相関係数 (r)** | **{r_val:.4f}** | {fmt_ms('pearson_r')} | 1.0 (線形相関) |",
        f"| | **Spearman 順位相関係数 (ρ)** | **{rho_val:.4f}** | {fmt_ms('spearman_rho')} | 1.0 (単調順位相関) |",
        f"| **二値判定性能 (閾値 0.5)** | **ROC-AUC** | **{auc:.4f}** | {fmt_ms('roc_auc')} | 1.0 (総合判別能) |",
        f"| | **PR-AUC (Average Precision)** | **{pr_auc:.4f}** | {fmt_ms('pr_auc')} | 1.0 (不均衡データ少数派検出能) |",
        f"| | **Accuracy (正解率)** | **{acc:.4f} ({acc:.1%})** | {fmt_ms('accuracy', is_pct=True)} | 100.0% |",
        f"| | **Balanced Accuracy (均衡正解率)** | **{balanced_acc:.4f} ({balanced_acc:.1%})** | {fmt_ms('balanced_acc', is_pct=True)} | 100.0% (不均衡補正版) |",
        f"| | **F1-Macro** | **{f1_macro:.4f}** | {fmt_ms('f1_macro')} | 1.0 |",
        "",
        "## 3. 被験者別 (Fold別) 性能詳細一覧",
        "| Fold | Test 被験者 ID | N | MAE | RMSE | R² | Pearson r | Spearman ρ | Acc | Balanced Acc | ROC-AUC | PR-AUC | F1-Macro |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for _, r in fold_df.iterrows():
        r2_s = f"{r['r2']:.4f}" if pd.notna(r.get('r2')) else "N/A"
        r_s = f"{r['pearson_r']:.3f}" if pd.notna(r.get('pearson_r')) else "N/A"
        rho_s = f"{r['spearman_rho']:.3f}" if pd.notna(r.get('spearman_rho')) else "N/A"
        acc_s = f"{r['acc_binary']:.1%}" if pd.notna(r.get('acc_binary')) else "N/A"
        bacc_s = f"{r['balanced_acc']:.1%}" if pd.notna(r.get('balanced_acc')) else "N/A"
        auc_s = f"{r['roc_auc']:.3f}" if pd.notna(r.get('roc_auc')) else "N/A"
        pr_s = f"{r['pr_auc']:.3f}" if pd.notna(r.get('pr_auc')) else "N/A"
        f1_s = f"{r['f1_macro']:.3f}" if pd.notna(r.get('f1_macro')) else "N/A"

        lines.append(
            f"| {int(r['fold']):2d} | `{r['test_user']}` | {int(r['n_test']):,d} | "
            f"{r['mae']:.4f} | {r['rmse']:.4f} | {r2_s} | {r_s} | {rho_s} | "
            f"{acc_s} | {bacc_s} | {auc_s} | {pr_s} | {f1_s} |"
        )

    # 平均・標準偏差のサマリー行
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines.append(
        f"| **Mean** | - | - | "
        f"**{stats_dict['mae'][0]:.4f}** | **{stats_dict['rmse'][0]:.4f}** | **{stats_dict['r2'][0]:.4f}** | "
        f"**{stats_dict['pearson_r'][0]:.3f}** | **{stats_dict['spearman_rho'][0]:.3f}** | "
        f"**{stats_dict['accuracy'][0]:.1%}** | **{stats_dict['balanced_acc'][0]:.1%}** | "
        f"**{stats_dict['roc_auc'][0]:.3f}** | **{stats_dict['pr_auc'][0]:.3f}** | **{stats_dict['f1_macro'][0]:.3f}** |"
    )
    lines.append(
        f"| **Std** | - | - | "
        f"±{stats_dict['mae'][1]:.4f} | ±{stats_dict['rmse'][1]:.4f} | ±{stats_dict['r2'][1]:.4f} | "
        f"±{stats_dict['pearson_r'][1]:.3f} | ±{stats_dict['spearman_rho'][1]:.3f} | "
        f"±{stats_dict['accuracy'][1]:.3f} | ±{stats_dict['balanced_acc'][1]:.3f} | "
        f"±{stats_dict['roc_auc'][1]:.3f} | ±{stats_dict['pr_auc'][1]:.3f} | ±{stats_dict['f1_macro'][1]:.3f} |"
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"💾 Markdown レポートを保存しました: {report_path}")
    return report_path
