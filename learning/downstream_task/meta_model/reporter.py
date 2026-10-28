"""
メタモデル (Visual + Vital 統合) 評価レポート & 可視化モジュール
(learning/downstream_task/meta_model/reporter.py)

選択された Visual ベースモデルと Vital ベースモデルの予測値を統合する
4種のメタモデル (LogisticRegression, Ridge, Lasso, ElasticNet) の評価 report.md を出力します。
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict
import pandas as pd

try:
    from ..reporter import generate_visualizations, plot_model_comparison, save_markdown_report
except (ImportError, ValueError):
    from downstream_task.reporter import generate_visualizations, plot_model_comparison, save_markdown_report


def save_meta_model_comparison_report(
    meta_output_dir: Path,
    window_size: str,
    selected_visual_model: str,
    selected_vital_model: str,
    df_meta: pd.DataFrame,
    base_metrics_summary: Dict[str, Dict],
    log,
) -> Path:
    """
    メタモデル (LogisticRegression, Ridge, Lasso, ElasticNet) の総合評価 report.md を出力します。
    """
    report_path = meta_output_dir / "report.md"

    best_auc_model = "-"
    best_r_model = "-"
    best_mae_model = "-"
    if not df_meta.empty:
        if "roc_auc" in df_meta.columns and df_meta["roc_auc"].notna().any():
            best_auc_model = str(df_meta.loc[df_meta["roc_auc"].idxmax(), "model"])
        if "pearson_r" in df_meta.columns and df_meta["pearson_r"].notna().any():
            best_r_model = str(df_meta.loc[df_meta["pearson_r"].idxmax(), "model"])
        if "mae" in df_meta.columns and df_meta["mae"].notna().any():
            best_mae_model = str(df_meta.loc[df_meta["mae"].idxmin(), "model"])

    lines = [
        f"# 🏆 Meta Model 総合評価レポート (Visual + Vital Late Fusion) [Window: {window_size}]",
        "",
        "## 1. 採用されたベースモデル設定 (`config.yaml` 指定)",
        f"- **Visual ベースモデル**: `{selected_visual_model}`",
        f"- **Vital ベースモデル**: `{selected_vital_model}`",
        "- **評価手法**: Leave-One-Subject-Out Cross-Validation (LOSO-CV) - 完全データリークフリー",
        "- **メタモデル一覧**: `logistic_regression`, `ridge`, `lasso`, `elastic_net`",
        "",
    ]

    if base_metrics_summary:
        lines.extend([
            "### 【参考】入力された単一モダリティ Base Model の単体性能 (OOF)",
            "| モダリティ | 採用ベースモデル | ROC-AUC | PR-AUC | Pearson $r$ | MAE | RMSE | Balanced Acc |",
            "|---|---|---|---|---|---|---|---|",
        ])
        for mod_key, b_m in base_metrics_summary.items():
            lines.append(
                f"| `{mod_key}` | `{b_m.get('model_name', '-')}` | "
                f"{b_m.get('roc_auc', float('nan')):.4f} | "
                f"{b_m.get('pr_auc', float('nan')):.4f} | "
                f"{b_m.get('pearson_r', float('nan')):.4f} | "
                f"{b_m.get('mae', float('nan')):.4f} | "
                f"{b_m.get('rmse', float('nan')):.4f} | "
                f"{b_m.get('balanced_acc', float('nan')):.1%} |"
            )
        lines.append("")

    lines.extend([
        "## 2. メタモデル比較サマリー (OOF 総合スコア & Fold間 Mean ± Std)",
        "| メタモデル | ROC-AUC (OOF / Mean±Std) | PR-AUC (OOF / Mean±Std) | Pearson $r$ (OOF / Mean±Std) | MAE (OOF / Mean±Std) | RMSE (OOF / Mean±Std) | Balanced Acc | F1-Macro | Visual重み | Vital重み | 保存モデル |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ])

    for _, r in df_meta.iterrows():
        m = r["model"]
        auc_s = f"**{r['roc_auc']:.4f}** ({r['roc_auc_mean']:.3f}±{r['roc_auc_std']:.3f})" if pd.notna(r.get("roc_auc")) else "N/A"
        pr_s = f"**{r['pr_auc']:.4f}** ({r['pr_auc_mean']:.3f}±{r['pr_auc_std']:.3f})" if pd.notna(r.get("pr_auc")) else "N/A"
        r_s = f"**{r['pearson_r']:.4f}** ({r['pearson_r_mean']:.3f}±{r['pearson_r_std']:.3f})" if pd.notna(r.get("pearson_r")) else "N/A"
        mae_s = f"{r['mae']:.4f} ({r['mae_mean']:.3f}±{r['mae_std']:.3f})"
        rmse_s = f"{r['rmse']:.4f} ({r['rmse_mean']:.3f}±{r['rmse_std']:.3f})"
        bacc_s = f"{r['balanced_acc']:.1%} ({r['balanced_acc_mean']:.3f}±{r['balanced_acc_std']:.3f})"
        f1_s = f"{r['f1_macro']:.4f}"
        w_vis = f"{r.get('visual_weight', 0.0):.4f}" if pd.notna(r.get("visual_weight")) else "-"
        w_vit = f"{r.get('vital_weight', 0.0):.4f}" if pd.notna(r.get("vital_weight")) else "-"
        model_file = f"`{m}_model.joblib`"

        lines.append(
            f"| `{m}` | {auc_s} | {pr_s} | {r_s} | {mae_s} | {rmse_s} | {bacc_s} | {f1_s} | {w_vis} | {w_vit} | {model_file} |"
        )

    lines.extend([
        "",
        "## 3. 最良メタモデル指標別ハイライト",
        f"- 🏆 **最高 ROC-AUC メタモデル**: `{best_auc_model}`",
        f"- 📈 **最高 Pearson $r$ メタモデル**: `{best_r_model}`",
        f"- 🎯 **最小 MAE メタモデル**: `{best_mae_model}`",
        "",
        "## 4. 最終推論結果 & 個別レポート一覧",
        "- 📄 **全メタモデル統合最終推論 CSV**: [`final_inference_predictions.csv`](final_inference_predictions.csv)",
        "- 📊 **メタモデル比較プロット**: [`meta_models_comparison.png`](meta_models_comparison.png)",
    ])

    for _, r in df_meta.iterrows():
        m = r["model"]
        lines.append(
            f"- **`{m}`**: [詳細レポート](loso_meta_{m}_report.md) | "
            f"[Fold別CSV](loso_meta_{m}_folds.csv) | "
            f"[OOF推論CSV](loso_meta_{m}_oof_predictions.csv) | "
            f"[保存モデル]({m}_model.joblib)"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"💾 Meta Model 総合評価レポート (report.md) を保存しました: {report_path}")
    return report_path


__all__ = [
    "generate_visualizations",
    "plot_model_comparison",
    "save_markdown_report",
    "save_meta_model_comparison_report",
]
