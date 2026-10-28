"""
ベースモデル (Visual / Vital) 評価レポート & 可視化モジュール
(learning/downstream_task/base_model/reporter.py)

Visual / Vital 各モダリティの 5 モデル (RandomForest, XGBoost, LightGBM, CatBoost, AdaBoost) について
ROC-AUC, PR-AUC, Pearson r, MAE, RMSE 等の評価レポート (report.md) および比較図を生成します。
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from ..reporter import generate_visualizations, plot_model_comparison, save_markdown_report
except (ImportError, ValueError):
    from downstream_task.reporter import generate_visualizations, plot_model_comparison, save_markdown_report


def save_modality_comparison_report(
    modality: str,
    window_size: str,
    df_mod: pd.DataFrame,
    modality_dir: Path,
    feature_names: List[str],
    early_stopping_rounds: int,
    log,
) -> Path:
    """
    単一モダリティ (visual または vital) 内の全ベースモデル比較 report.md を保存します。
    """
    report_path = modality_dir / "report.md"

    # ROC-AUC 最大 (同率なら Pearson r 最大) のモデルを推奨としてハイライト
    best_model_auc = "-"
    best_model_r = "-"
    best_model_mae = "-"
    if not df_mod.empty:
        if "roc_auc" in df_mod.columns and df_mod["roc_auc"].notna().any():
            best_model_auc = str(df_mod.loc[df_mod["roc_auc"].idxmax(), "model"])
        if "pearson_r" in df_mod.columns and df_mod["pearson_r"].notna().any():
            best_model_r = str(df_mod.loc[df_mod["pearson_r"].idxmax(), "model"])
        if "mae" in df_mod.columns and df_mod["mae"].notna().any():
            best_model_mae = str(df_mod.loc[df_mod["mae"].idxmin(), "model"])

    lines = [
        f"# 📊 Base Model 評価レポート [{modality.upper()} モダリティ | Window: {window_size}]",
        "",
        "## 1. 実験・モデル設定概要",
        f"- **対象モダリティ**: `{modality}` (入力特徴量数: {len(feature_names)} 次元)",
        f"- **使用特徴量**: `{', '.join(feature_names)}`",
        "- **評価プロトコル**: Leave-One-Subject-Out Cross-Validation (LOSO-CV) - 完全データリークフリー",
        "- **学習ターゲット**: Fold内 GMM 期待値ソフトラベル $y_{\\text{soft}} \\in [0, 1]$ (サンプル重み付け回帰)",
        "- **評価正解ラベル**: 実験タスク正解ラベル `label` (集中=1 / リラックス=0)",
        f"- **Early Stopping**: 有効 (`stopping_rounds = {early_stopping_rounds}`, Fold内検証データ重み付きRMSE監視)",
        "",
        "## 2. モデル比較サマリー (OOF 総合スコア & Fold間 Mean ± Std)",
        "| モデル | ROC-AUC (OOF / Mean±Std) | PR-AUC (OOF / Mean±Std) | Pearson $r$ (OOF / Mean±Std) | MAE (OOF / Mean±Std) | RMSE (OOF / Mean±Std) | Balanced Acc | F1-Macro | 平均Best Iter | 保存モデルファイル |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for _, r in df_mod.iterrows():
        m = r["model"]
        auc_s = f"**{r['roc_auc']:.4f}** ({r['roc_auc_mean']:.3f}±{r['roc_auc_std']:.3f})" if pd.notna(r.get("roc_auc")) else "N/A"
        pr_s = f"**{r['pr_auc']:.4f}** ({r['pr_auc_mean']:.3f}±{r['pr_auc_std']:.3f})" if pd.notna(r.get("pr_auc")) else "N/A"
        r_s = f"**{r['pearson_r']:.4f}** ({r['pearson_r_mean']:.3f}±{r['pearson_r_std']:.3f})" if pd.notna(r.get("pearson_r")) else "N/A"
        mae_s = f"{r['mae']:.4f} ({r['mae_mean']:.3f}±{r['mae_std']:.3f})"
        rmse_s = f"{r['rmse']:.4f} ({r['rmse_mean']:.3f}±{r['rmse_std']:.3f})"
        bacc_s = f"{r['balanced_acc']:.1%} ({r['balanced_acc_mean']:.3f}±{r['balanced_acc_std']:.3f})"
        f1_s = f"{r['f1_macro']:.4f}"
        best_it = f"{r.get('avg_best_iteration', 0):.1f}" if pd.notna(r.get("avg_best_iteration")) else "-"
        model_file = f"`{m}_model.joblib`"

        lines.append(
            f"| `{m}` | {auc_s} | {pr_s} | {r_s} | {mae_s} | {rmse_s} | {bacc_s} | {f1_s} | {best_it} | {model_file} |"
        )

    lines.extend([
        "",
        "## 3. 最良モデル指標別ハイライト",
        f"- 🏆 **最高 ROC-AUC モデル**: `{best_model_auc}`",
        f"- 📈 **最高 Pearson $r$ モデル**: `{best_model_r}`",
        f"- 🎯 **最小 MAE モデル**: `{best_model_mae}`",
        "",
        "## 4. 各モデル詳細レポート・成果物リンク",
    ])

    for _, r in df_mod.iterrows():
        m = r["model"]
        lines.append(
            f"- **`{m}`**: [詳細レポート](loso_{modality}_{m}_report.md) | "
            f"[Fold別CSV](loso_{modality}_{m}_folds.csv) | "
            f"[OOF予測CSV](loso_{modality}_{m}_oof_predictions.csv) | "
            f"[保存モデル]({m}_model.joblib)"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"💾 [{modality.upper()}] モダリティ比較レポートを保存しました: {report_path}")
    return report_path


def save_overall_base_model_report(
    base_output_dir: Path,
    window_size: str,
    df_all: pd.DataFrame,
    modality_features: Dict[str, List[str]],
    early_stopping_rounds: int,
    log,
) -> Path:
    """
    Visual / Vital 両モダリティの全ベースモデル評価を統合した総合 report.md を保存します。
    """
    report_path = base_output_dir / "report.md"

    lines = [
        f"# 🛡️ Base Model 総合評価レポート (Visual & Vital) [Window: {window_size}]",
        "",
        "## 1. 概要",
        "- **対象モデル (全5種・回帰)**: `random_forest`, `xgboost`, `lightgbm`, `catboost`, `adaboost`",
        "- **モダリティ分離**: `visual` (視覚特徴量) と `vital` (生体特徴量) でそれぞれ独立してモデルを構築・評価・保存",
        f"- **Early Stopping**: 全モデルに適用 (`stopping_rounds = {early_stopping_rounds}`)",
        "- **評価指標**: ROC-AUC, PR-AUC, Pearson $r$, Spearman $\\rho$, MAE, RMSE, $R^2$, Accuracy, Balanced Accuracy, F1-Macro",
        "",
    ]

    best_per_mod: Dict[str, str] = {}

    for modality in ["visual", "vital"]:
        df_sub = df_all[df_all["modality"] == modality] if "modality" in df_all.columns else pd.DataFrame()
        if df_sub.empty:
            continue

        feats = modality_features.get(modality, [])
        best_auc_model = (
            str(df_sub.loc[df_sub["roc_auc"].idxmax(), "model"])
            if "roc_auc" in df_sub.columns and df_sub["roc_auc"].notna().any()
            else str(df_sub.iloc[0]["model"])
        )
        best_r_model = (
            str(df_sub.loc[df_sub["pearson_r"].idxmax(), "model"])
            if "pearson_r" in df_sub.columns and df_sub["pearson_r"].notna().any()
            else str(df_sub.iloc[0]["model"])
        )
        best_per_mod[modality] = best_auc_model

        lines.extend([
            f"## 2-{1 if modality == 'visual' else 2}. `{modality.upper()}` モダリティ 評価結果 (特徴量: {len(feats)}次元)",
            "| モデル | ROC-AUC (OOF) | ROC-AUC (Mean±Std) | PR-AUC (OOF) | Pearson $r$ (OOF) | MAE (OOF) | RMSE (OOF) | Balanced Acc | F1-Macro | 平均Best Iter | 保存パス |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ])

        for _, r in df_sub.iterrows():
            m = r["model"]
            auc_oof = f"**{r['roc_auc']:.4f}**" if pd.notna(r.get("roc_auc")) else "N/A"
            auc_ms = f"{r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f}" if pd.notna(r.get("roc_auc_mean")) else "N/A"
            pr_oof = f"**{r['pr_auc']:.4f}**" if pd.notna(r.get("pr_auc")) else "N/A"
            r_oof = f"**{r['pearson_r']:.4f}**" if pd.notna(r.get("pearson_r")) else "N/A"
            mae_oof = f"{r['mae']:.4f}"
            rmse_oof = f"{r['rmse']:.4f}"
            bacc_oof = f"{r['balanced_acc']:.1%}"
            f1_oof = f"{r['f1_macro']:.4f}"
            best_it = f"{r.get('avg_best_iteration', 0):.1f}" if pd.notna(r.get("avg_best_iteration")) else "-"
            save_rel = f"`{modality}/{m}_model.joblib`"

            lines.append(
                f"| `{m}` | {auc_oof} | {auc_ms} | {pr_oof} | {r_oof} | {mae_oof} | {rmse_oof} | {bacc_oof} | {f1_oof} | {best_it} | {save_rel} |"
            )

        lines.extend([
            "",
            f"- **最高 ROC-AUC モデル ({modality})**: `{best_auc_model}` | **最高 Pearson $r$ モデル ({modality})**: `{best_r_model}`",
            f"- 📁 モダリティ別詳細レポート: [`{modality}/report.md`]({modality}/report.md)",
            "",
        ])

    vis_rec = best_per_mod.get("visual", "lightgbm")
    vit_rec = best_per_mod.get("vital", "lightgbm")

    lines.extend([
        "## 3. 次ステップ（メタモデル学習）に向けた `config.yaml` 設定ガイド",
        "上記の結果を確認し、`learning/config.yaml` の `downstream_task.meta_model.selected_base_models` に採用するモデル名を指定してください：",
        "",
        "```yaml",
        "downstream_task:",
        "  meta_model:",
        "    selected_base_models:",
        f'      visual: "{vis_rec}"   # random_forest, xgboost, lightgbm, catboost, adaboost から選択',
        f'      vital: "{vit_rec}"    # random_forest, xgboost, lightgbm, catboost, adaboost から選択',
        "```",
        "",
        "設定後、以下のコマンドでメタモデル（LogisticRegression, Ridge, Lasso, ElasticNet）の学習・評価を実行できます：",
        "```bash",
        "uv run python learning/downstream_task/meta_model/main.py",
        "# または",
        "uv run python learning/downstream_task/main.py --stage meta_model",
        "```",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"💾 Base Model 総合評価レポート (report.md) を保存しました: {report_path}")
    return report_path


def plot_dual_modality_comparison(
    df_all: pd.DataFrame,
    output_dir: Path,
    log,
) -> Optional[Path]:
    """Visual と Vital の全5モデルの ROC-AUC, PR-AUC, Pearson r, MAE を並列比較するプロット"""
    if df_all.empty or "modality" not in df_all.columns:
        return None

    fig_path = output_dir / "base_models_comparison.png"
    metrics = [
        ("roc_auc", "ROC-AUC (Higher is better)"),
        ("pr_auc", "PR-AUC (Higher is better)"),
        ("pearson_r", "Pearson r (Higher is better)"),
        ("mae", "MAE (Lower is better)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=130)
    axes = axes.flatten()

    vis_df = df_all[df_all["modality"] == "visual"].set_index("model")
    vit_df = df_all[df_all["modality"] == "vital"].set_index("model")
    models = list(dict.fromkeys(df_all["model"].tolist()))
    x = np.arange(len(models))
    width = 0.36

    for idx, (m_key, title) in enumerate(metrics):
        ax = axes[idx]
        vis_vals = [vis_df.loc[m, m_key] if m in vis_df.index else np.nan for m in models]
        vit_vals = [vit_df.loc[m, m_key] if m in vit_df.index else np.nan for m in models]

        b1 = ax.bar(x - width / 2, vis_vals, width, label="Visual", color="#2563eb", alpha=0.88, edgecolor="black")
        b2 = ax.bar(x + width / 2, vit_vals, width, label="Vital", color="#059669", alpha=0.88, edgecolor="black")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend()

        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        h + (0.01 if h >= 0 else -0.025),
                        f"{h:.3f}",
                        ha="center",
                        va="bottom" if h >= 0 else "top",
                        fontsize=8.5,
                        fontweight="bold",
                    )

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)
    log.info(f"📊 Visual / Vital ベースモデル比較プロットを保存しました: {fig_path}")
    return fig_path


__all__ = [
    "generate_visualizations",
    "plot_model_comparison",
    "save_markdown_report",
    "save_modality_comparison_report",
    "save_overall_base_model_report",
    "plot_dual_modality_comparison",
]
