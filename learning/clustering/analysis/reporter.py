from pathlib import Path
import pandas as pd

from config_loader import log


def df_to_markdown_safe(df: pd.DataFrame) -> str:
    """tabulate パッケージの有無に依存しない安全な Markdown 表変換関数"""
    try:
        return df.to_markdown()
    except Exception:
        # 自前フォーマット
        headers = [str(c) for c in df.columns]
        if df.index.name or not isinstance(df.index, pd.RangeIndex):
            idx_name = str(df.index.name) if df.index.name else "Index"
            headers = [idx_name] + headers
            lines = [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
            ]
            for idx, row in df.iterrows():
                vals = [str(idx)] + [
                    f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.values
                ]
                lines.append("| " + " | ".join(vals) + " |")
        else:
            lines = [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
            ]
            for _, row in df.iterrows():
                vals = [
                    f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.values
                ]
                lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)


def generate_cluster_report(
    crosstab: pd.DataFrame,
    profiles: pd.DataFrame,
    metrics_summary: dict,
    save_path: Path,
    window_size: str = "10s",
) -> None:
    """
    各クラスタの特徴量（タスク比率、生体指標プロファイル）をまとめた Markdown レポートを出力します。
    """
    clusters = (
        sorted(crosstab.index.tolist())
        if not crosstab.empty
        else (sorted(profiles.index.tolist()) if not profiles.empty else [])
    )

    report_lines = [
        f"# GMM クラスタ特徴量分析レポート (Window: {window_size})",
        "",
        "## 1. 各クラスタの特徴量サマリー",
        "",
        "| クラスタ番号 | サンプル数 | 集中タスク割合 (%) | 休憩タスク割合 (%) | 平均心拍数 (BPM) | 平均瞬目 (EAR) | 頭部変動分散 | 視線変動分散 |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for c in clusters:
        total_samples = "-"
        conc_ratio = "-"
        relax_ratio = "-"
        if not crosstab.empty and c in crosstab.index:
            total_samples = str(crosstab.loc[c, "Total_Samples"]) if "Total_Samples" in crosstab.columns else "-"
            conc_ratio = (
                f"{crosstab.loc[c, 'Concentration_Ratio (%)']:.1f}%"
                if "Concentration_Ratio (%)" in crosstab.columns
                else "-"
            )
            relax_ratio = (
                f"{crosstab.loc[c, 'Relax_Ratio (%)']:.1f}%"
                if "Relax_Ratio (%)" in crosstab.columns
                else "-"
            )

        bpm_str = "-"
        ear_str = "-"
        head_str = "-"
        gaze_str = "-"
        if not profiles.empty and c in profiles.index:
            if "bpm_mean_mean" in profiles.columns:
                bpm_str = f"{profiles.loc[c, 'bpm_mean_mean']:.1f}"
            if "ear_mean_mean" in profiles.columns:
                ear_str = f"{profiles.loc[c, 'ear_mean_mean']:.3f}"
            if "head_movement_var_mean" in profiles.columns:
                head_str = f"{profiles.loc[c, 'head_movement_var_mean']:.4f}"
            if "gaze_movement_var_mean" in profiles.columns:
                gaze_str = f"{profiles.loc[c, 'gaze_movement_var_mean']:.4f}"

        report_lines.append(
            f"| **Cluster {c}** | {total_samples} | {conc_ratio} | {relax_ratio} | {bpm_str} | {ear_str} | {head_str} | {gaze_str} |"
        )

    report_lines.extend([
        "",
        "---",
        "",
        "## 2. 実験タスクとのクロス集計結果 (Contingency Table)",
        "",
        "- **タスク定義**: `label = 1`（集中タスク区間: 講義動画視聴） / `label = 0`（休憩区間: 焚き火動画視聴）",
        f"- **クラスタ純度 (Purity)**: `{metrics_summary.get('purity', 'N/A')}`",
        f"- **Adjusted Rand Index (ARI)**: `{metrics_summary.get('ari', 'N/A')}`",
        f"- **Normalized Mutual Information (NMI)**: `{metrics_summary.get('nmi', 'N/A')}`",
        "",
    ])

    if not crosstab.empty:
        report_lines.append(df_to_markdown_safe(crosstab))
        report_lines.append("")

    report_lines.extend([
        "---",
        "",
        "## 3. 各クラスタにおける生体シグナル物理指標プロファイル詳細",
        "",
        "- **心拍数 (BPM)**: 集中・認知負荷時に交感神経優位となり上昇傾向、リラックス時に副交感神経優位となり下降傾向。",
        "- **瞬目比率 (EAR)**: 集中時は瞬目が減少し開眼状態が持続、リラックス・傾眠時は瞬目増加や開眼度低下。",
        "- **頭部・視線の動き (分散)**: 集中時は画面注視のため頭部・視線の変動分散が極小化、リラックス時は姿勢変動や視線移動の分散が増大。",
        "",
    ])

    if not profiles.empty:
        report_lines.append(df_to_markdown_safe(profiles.round(4)))
        report_lines.append("")

    report_lines.extend([
        "---",
        "",
        "## 4. ラベル付与方針について",
        "",
        "各クラスタの集中・リラックス等のラベル割り当ては、上記のタスク比率や生体指標プロファイルを参照して手動で決定してください。",
        "",
    ])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    log.info(f"📄 分析レポート (Markdown) を保存しました: {save_path}")
