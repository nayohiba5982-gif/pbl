"""
閾値決定（Threshold Tuning）結果の可視化・Markdownレポート出力モジュール
(learning/downstream_task/threshold_decision/reporter.py)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_threshold_visualizations(
    output_dir: Path,
    model_tag: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tuning_res: Dict,
    log=None,
) -> Optional[Path]:
    """
    閾値探索曲線・Fold別最適閾値・固定閾値(0.5)との性能比較の4パネル図を生成・保存します。
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        crit = tuning_res["criterion"]
        plot_path = output_dir / f"loso_{model_tag}_threshold_{crit}_plots.png"

        curve_df: pd.DataFrame = tuning_res.get("global_val_curve_df", pd.DataFrame())
        fold_df: pd.DataFrame = tuning_res.get("fold_df", pd.DataFrame())
        all_ths: Dict = tuning_res.get("global_all_criteria_thresholds", {})
        mean_th = float(tuning_res.get("mean_threshold", 0.5))

        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        fig.suptitle(
            f"Leak-Free LOSO Threshold Tuning: {model_tag} (Criterion: {crit})",
            fontsize=14,
            fontweight="bold",
        )

        # Panel 1: 閾値探索曲線 (3大基準)
        ax1 = axes[0, 0]
        if not curve_df.empty:
            ths = curve_df["threshold"].values
            ax1.plot(ths, curve_df["balanced_acc"].values, label="Balanced Acc ((J+1)/2)", color="#1f77b4", lw=2)
            ax1.plot(ths, curve_df["f1_macro"].values, label="F1-Macro", color="#ff7f0e", lw=2)
            ax1.plot(ths, curve_df["pr_product"].values, label="PR Product (P*R)", color="#2ca02c", lw=1.8, ls="--")
            ax1.plot(ths, 1.0 - curve_df["pr_diff"].values, label="1 - |P - R|", color="#9467bd", lw=1.5, ls=":")

        ax1.axvline(0.5, color="gray", linestyle="--", lw=1.2, label="Fixed th=0.50")
        ax1.axvline(mean_th, color="#d62728", linestyle="-", lw=2.2, label=f"Mean Tuned th*={mean_th:.3f}")
        if "youden_j" in all_ths:
            ax1.axvline(all_ths["youden_j"], color="#1f77b4", linestyle=":", alpha=0.7)
        if "f1_macro" in all_ths:
            ax1.axvline(all_ths["f1_macro"], color="#ff7f0e", linestyle=":", alpha=0.7)
        ax1.set_title("1. Validation Threshold Curves (3 Criteria)")
        ax1.set_xlabel("Threshold (th)")
        ax1.set_ylabel("Metric Score")
        ax1.set_xlim(0.0, 1.0)
        ax1.set_ylim(0.0, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="lower left", fontsize=8.5)

        # Panel 2: LOSO Fold 別の最適閾値 th*_k の推移
        ax2 = axes[0, 1]
        if not fold_df.empty and "optimal_threshold" in fold_df.columns:
            folds = fold_df["fold"].values
            opt_ths = fold_df["optimal_threshold"].values
            ax2.plot(folds, opt_ths, marker="o", color="#d62728", lw=1.8, label=f"Selected ({crit})")
            if "th_youden_j" in fold_df.columns:
                ax2.plot(folds, fold_df["th_youden_j"].values, marker="x", ls=":", alpha=0.6, label="Youden's J th*")
            if "th_f1_macro" in fold_df.columns:
                ax2.plot(folds, fold_df["th_f1_macro"].values, marker="^", ls=":", alpha=0.6, label="F1-Macro th*")
            ax2.axhline(mean_th, color="#d62728", ls="--", alpha=0.7, label=f"Mean th*={mean_th:.3f}")
            ax2.axhline(0.5, color="gray", ls="--", alpha=0.6, label="Default 0.50")
            ax2.set_xticks(folds)
        ax2.set_title("2. Optimal Threshold (th*_k) per LOSO Fold (Val-derived)")
        ax2.set_xlabel("LOSO Fold (Held-out Test Subject)")
        ax2.set_ylabel("Optimal Threshold th*_k")
        ax2.set_ylim(0.0, 1.0)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best", fontsize=8.5)

        # Panel 3: Fixed (0.5) vs Tuned (th*) OOF Test 性能比較
        ax3 = axes[1, 0]
        base_m = tuning_res["baseline_oof_detail"]
        tuned_m = tuning_res["tuned_oof_detail"]
        metric_names = ["Balanced Acc", "F1-Macro", "F1 (Rest:0)", "Specificity", "Accuracy"]
        base_vals = [
            base_m.get("balanced_acc", 0.0),
            base_m.get("f1_macro", 0.0),
            base_m.get("f1_neg", 0.0),
            base_m.get("specificity", 0.0),
            base_m.get("accuracy", 0.0),
        ]
        tuned_vals = [
            tuned_m.get("balanced_acc", 0.0),
            tuned_m.get("f1_macro", 0.0),
            tuned_m.get("f1_neg", 0.0),
            tuned_m.get("specificity", 0.0),
            tuned_m.get("accuracy", 0.0),
        ]
        x_pos = np.arange(len(metric_names))
        width = 0.36
        rects1 = ax3.bar(x_pos - width / 2, base_vals, width, label="Fixed (th=0.50)", color="#8c8c8c", alpha=0.85)
        rects2 = ax3.bar(x_pos + width / 2, tuned_vals, width, label=f"Tuned ({crit})", color="#2b7bba", alpha=0.9)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(metric_names, fontsize=9)
        ax3.set_ylim(0.0, 1.12)
        ax3.set_ylabel("OOF Test Score")
        ax3.set_title("3. Test Performance: Fixed (0.5) vs Val-Tuned Threshold")
        ax3.grid(True, axis="y", alpha=0.3)
        ax3.legend(loc="upper right", fontsize=9)

        for r in rects1:
            h = r.get_height()
            if np.isfinite(h):
                ax3.text(r.get_x() + r.get_width() / 2.0, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        for r in rects2:
            h = r.get_height()
            if np.isfinite(h):
                ax3.text(
                    r.get_x() + r.get_width() / 2.0,
                    h + 0.015,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        # Panel 4: 予測確率ヒストグラム (Class 0 vs Class 1) とカットオフ位置
        ax4 = axes[1, 1]
        y_bin = (np.asarray(y_true, dtype=float) >= 0.5).astype(int)
        preds = np.asarray(y_pred, dtype=float)
        p_neg = preds[y_bin == 0]
        p_pos = preds[y_bin == 1]
        bins = np.linspace(0.0, 1.0, 35)
        if len(p_neg) > 0:
            ax4.hist(p_neg, bins=bins, density=True, alpha=0.55, color="#ff7f0e", label=f"Rest (Class 0, n={len(p_neg)})")
        if len(p_pos) > 0:
            ax4.hist(p_pos, bins=bins, density=True, alpha=0.55, color="#1f77b4", label=f"Focus (Class 1, n={len(p_pos)})")
        ax4.axvline(0.5, color="gray", ls="--", lw=1.5, label="Fixed th=0.50")
        ax4.axvline(mean_th, color="#d62728", ls="-", lw=2.2, label=f"Mean Tuned th*={mean_th:.3f}")
        ax4.set_title("4. OOF Prediction Distribution & Cutoff Threshold")
        ax4.set_xlabel("Predicted Score (y_pred)")
        ax4.set_ylabel("Density")
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc="upper left", fontsize=8.5)

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        if log:
            log.info(f"📊 閾値決定プロットを保存しました: {plot_path}")
        return plot_path
    except Exception as e:
        if log:
            log.warning(f"⚠️ 閾値決定プロット生成中にエラーが発生しました: {e}")
        return None


def save_single_threshold_report(
    output_dir: Path,
    window_size: str,
    model_tag: str,
    tuning_res: Dict,
    log=None,
) -> Path:
    """単一モデル・単一基準の詳細な閾値決定 Markdown レポートを保存します。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    crit = tuning_res["criterion"]
    crit_label = tuning_res.get("criterion_label", crit)
    report_path = output_dir / f"loso_{model_tag}_threshold_{crit}_report.md"

    base_oof = tuning_res["baseline_oof_detail"]
    tuned_oof = tuning_res["tuned_oof_detail"]
    base_st = tuning_res["baseline_fold_stats"]
    tuned_st = tuning_res["tuned_fold_stats"]
    imp = tuning_res["improvement"]
    fold_df: pd.DataFrame = tuning_res["fold_df"]
    all_ths = tuning_res.get("global_all_criteria_thresholds", {})

    lines = [
        f"# 🎯 閾値決定 (Threshold Decision) 評価レポート: `{model_tag}` [Window: {window_size}]",
        "",
        "## 1. 閾値最適化の設定（完全データリークフリー LOSO-CV）",
        f"- **対象モデル**: `{model_tag}`",
        f"- **採用基準 (Criterion)**: `{crit}` ({crit_label})",
        f"- **検証データ抽出モード**: `{tuning_res.get('val_split_mode', 'oof_non_test')}` (Test被験者を完全除外)",
        f"- **Fold間 最適閾値 $th^*$ (Mean ± Std)**: **`{tuning_res['mean_threshold']:.4f} ± {tuning_res['std_threshold']:.4f}`** (Median: `{tuning_res['median_threshold']:.4f}`)",
        "",
        "### 参考：全体検証データにおける3大基準それぞれの最適閾値",
        f"- **① Youden's J Index 最大化 (`youden_j`)**: `th* = {all_ths.get('youden_j', 0.5):.4f}`",
        f"- **② F1-Macro 最大化 (`f1_macro`)**: `th* = {all_ths.get('f1_macro', 0.5):.4f}`",
        f"- **③ Precision-Recall バランス (`pr_balance`)**: `th* = {all_ths.get('pr_balance', 0.5):.4f}`",
        "",
        "## 2. 固定閾値 ($th = 0.5$) vs 検証データ最適化閾値 ($th = th^*_k$) の比較",
        "| 評価指標 | 固定閾値 ($th=0.50$) [OOF] | 最適化閾値 ($th=th^*_k$) [OOF] | 改善幅 ($\\Delta$ OOF) | 固定 ($th=0.50$) [Fold Mean±Std] | 最適化 ($th=th^*_k$) [Fold Mean±Std] |",
        "|---|---|---|---|---|---|",
        f"| **Balanced Accuracy** | {base_oof['balanced_acc']:.4f} | **{tuned_oof['balanced_acc']:.4f}** | **{imp['oof_balanced_acc_diff']:+.4f}** | {base_st['balanced_acc'][0]:.4f} ± {base_st['balanced_acc'][1]:.4f} | **{tuned_st['balanced_acc'][0]:.4f} ± {tuned_st['balanced_acc'][1]:.4f}** |",
        f"| **F1-Macro** | {base_oof['f1_macro']:.4f} | **{tuned_oof['f1_macro']:.4f}** | **{imp['oof_f1_macro_diff']:+.4f}** | {base_st['f1_macro'][0]:.4f} ± {base_st['f1_macro'][1]:.4f} | **{tuned_st['f1_macro'][0]:.4f} ± {tuned_st['f1_macro'][1]:.4f}** |",
        f"| **F1 (休憩: Class 0)** | {base_oof['f1_neg']:.4f} | **{tuned_oof['f1_neg']:.4f}** | **{imp['oof_f1_neg_diff']:+.4f}** | {base_st['f1_neg'][0]:.4f} ± {base_st['f1_neg'][1]:.4f} | **{tuned_st['f1_neg'][0]:.4f} ± {tuned_st['f1_neg'][1]:.4f}** |",
        f"| **F1 (集中: Class 1)** | {base_oof['f1_pos']:.4f} | **{tuned_oof['f1_pos']:.4f}** | {tuned_oof['f1_pos'] - base_oof['f1_pos']:+.4f} | - | {tuned_st['f1_pos'][0]:.4f} ± {tuned_st['f1_pos'][1]:.4f} |",
        f"| **Specificity (休憩検出率)** | {base_oof['specificity']:.4f} | **{tuned_oof['specificity']:.4f}** | **{imp['oof_specificity_diff']:+.4f}** | {base_st['specificity'][0]:.4f} ± {base_st['specificity'][1]:.4f} | **{tuned_st['specificity'][0]:.4f} ± {tuned_st['specificity'][1]:.4f}** |",
        f"| **Recall (集中検出率)** | {base_oof['recall']:.4f} | **{tuned_oof['recall']:.4f}** | {tuned_oof['recall'] - base_oof['recall']:+.4f} | - | {tuned_st['recall'][0]:.4f} ± {tuned_st['recall'][1]:.4f} |",
        f"| **Precision (集中適合率)** | {base_oof['precision']:.4f} | **{tuned_oof['precision']:.4f}** | {tuned_oof['precision'] - base_oof['precision']:+.4f} | - | {tuned_st['precision'][0]:.4f} ± {tuned_st['precision'][1]:.4f} |",
        f"| **Accuracy (全体正解率)** | {base_oof['accuracy']:.4f} | **{tuned_oof['accuracy']:.4f}** | {imp['oof_accuracy_diff']:+.4f} | {base_st['accuracy'][0]:.4f} ± {base_st['accuracy'][1]:.4f} | {tuned_st['accuracy'][0]:.4f} ± {tuned_st['accuracy'][1]:.4f} |",
        f"| **Youden's J Index** | {base_oof['youden_j']:.4f} | **{tuned_oof['youden_j']:.4f}** | {tuned_oof['youden_j'] - base_oof['youden_j']:+.4f} | - | {tuned_st['youden_j'][0]:.4f} ± {tuned_st['youden_j'][1]:.4f} |",
        "",
        "## 3. 混同行列 (OOF 全体)",
        f"- **固定閾値 ($th=0.50$)**: TP={base_oof['tp']}, FP={base_oof['fp']}, TN={base_oof['tn']}, FN={base_oof['fn']}",
        f"- **最適化閾値 ($th=th^*_k$)**: TP={tuned_oof['tp']}, FP={tuned_oof['fp']}, TN={tuned_oof['tn']}, FN={tuned_oof['fn']}",
        "",
        "## 4. LOSO Fold 別の決定閾値とテスト被験者スコア",
        "| Fold | Test User | $th^*_k$ (Val決定) | Balanced Acc (0.5) | Balanced Acc ($th^*_k$) | $\\Delta$ BalAcc | F1-Macro (0.5) | F1-Macro ($th^*_k$) | $\\Delta$ F1-Macro |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for _, r in fold_df.iterrows():
        b_ba = f"{r['balanced_acc_base_05']:.4f}" if pd.notna(r["balanced_acc_base_05"]) else "N/A"
        t_ba = f"{r['balanced_acc']:.4f}" if pd.notna(r["balanced_acc"]) else "N/A"
        d_ba = f"{r['delta_balanced_acc']:+.4f}" if pd.notna(r["delta_balanced_acc"]) else "N/A"
        lines.append(
            f"| {int(r['fold'])} | `{r['test_user']}` | **{r['optimal_threshold']:.4f}** | "
            f"{b_ba} | **{t_ba}** | {d_ba} | "
            f"{r['f1_macro_base_05']:.4f} | **{r['f1_macro']:.4f}** | {r['delta_f1_macro']:+.4f} |"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if log:
        log.info(f"📝 閾値決定レポートを保存しました: {report_path}")
    return report_path


def save_threshold_decision_summary_report(
    output_dir: Path,
    window_size: str,
    default_criterion: str,
    comparison_df: pd.DataFrame,
    log=None,
) -> Path:
    """全モデル・3大基準の閾値決定 総合比較レポート (report.md) を出力します。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"

    lines = [
        f"# 🎯 Step 7-3: 閾値決定 (Threshold Decision) 総合比較レポート [Window: {window_size}]",
        "",
        "## 1. 概要とリークフリー設計",
        "- **鉄則**: テスト被験者（Test）のデータは一切参照せず、各 LOSO Fold 内の学習・検証データ（Train/Val）のみで最適カットオフ閾値 $th^*_k \\in [0, 1]$ を決定し、テスト被験者の二値化に適用しています。",
        f"- **デフォルト採用基準 (`config.yaml`)**: `{default_criterion}`",
        "- **3大基準**:",
        "  1. `youden_j` (**Youden's J index**): $J(th) = \\text{Recall}(th) + \\text{Specificity}(th) - 1 = 2 \\times \\text{Balanced Accuracy}(th) - 1$",
        "  2. `f1_macro` (**F1-Macro 最大化**): 集中（多数派）と休憩（少数派）の両クラスの調和平均を最大化",
        "  3. `pr_balance` (**Precision-Recall バランス**): 適合率と再現率の差が最小（または積が最大）となる均衡点",
        "",
        "## 2. モデル × 基準別の性能比較一覧 (Fixed 0.50 vs Tuned $th^*_k$)",
        "| スコープ | モデル | 最適化基準 | Mean $th^* \\pm$ Std | BalAcc (0.5) | **BalAcc ($th^*$)** | $\\Delta$ BalAcc | F1-Macro (0.5) | **F1-Macro ($th^*$)** | $\\Delta$ F1-Macro | F1-Rest (0.5) | **F1-Rest ($th^*$)** | Accuracy ($th^*$) | ROC-AUC |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for _, row in comparison_df.iterrows():
        lines.append(
            f"| `{row['scope']}` | `{row['model']}` | `{row['criterion']}` | "
            f"`{row['mean_th']:.4f} ± {row['std_th']:.4f}` | "
            f"{row['balanced_acc_05']:.4f} | **{row['balanced_acc_tuned']:.4f}** | **{row['delta_balanced_acc']:+.4f}** | "
            f"{row['f1_macro_05']:.4f} | **{row['f1_macro_tuned']:.4f}** | **{row['delta_f1_macro']:+.4f}** | "
            f"{row['f1_neg_05']:.4f} | **{row['f1_neg_tuned']:.4f}** | "
            f"{row['accuracy_tuned']:.4f} | {row['roc_auc']:.4f} |"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if log:
        log.info(f"🏆 閾値決定 総合比較レポートを保存しました: {report_path}")
    return report_path
