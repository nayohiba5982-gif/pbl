import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from config_loader import log


def compute_cross_tabulation(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    実験時に提示したタスク区間（label: 1=集中タスク, 0=休憩区間）と
    GMM クラスタ（cluster: 0, 1, ...）のクロス集計を実行します。
    """
    clusters = sorted(df["cluster"].unique())

    has_label = "label" in df.columns
    if not has_label:
        log.warning("⚠️ メタデータに実験タスク列 'label' が存在しません。クロス集計はスキップされます。")
        empty_df = pd.DataFrame(index=clusters)
        empty_df["count"] = df["cluster"].value_counts().sort_index()
        return empty_df, {"purity": None, "ari": None, "nmi": None}

    # label をバイナリ整数化 (0: リラックス/休憩, 1: 集中タスク)
    df_clean = df.dropna(subset=["cluster", "label"]).copy()
    df_clean["label_bin"] = (df_clean["label"] >= 0.5).astype(int)

    # 分割表の作成 (行: cluster, 列: label)
    crosstab = pd.crosstab(
        df_clean["cluster"],
        df_clean["label_bin"],
        margins=True,
        margins_name="Total",
    )

    col_mapping = {}
    if 0 in crosstab.columns:
        col_mapping[0] = "Task_Relax (label=0)"
    if 1 in crosstab.columns:
        col_mapping[1] = "Task_Concentration (label=1)"
    crosstab = crosstab.rename(columns=col_mapping)

    detailed_table = pd.DataFrame(index=clusters)
    total_per_cluster = df_clean["cluster"].value_counts().sort_index()

    detailed_table["Total_Samples"] = total_per_cluster

    for l_val, col_name in [(1, "Concentration_Count"), (0, "Relax_Count")]:
        cnt = (
            df_clean[df_clean["label_bin"] == l_val]["cluster"]
            .value_counts()
            .reindex(clusters, fill_value=0)
        )
        detailed_table[col_name] = cnt

    detailed_table["Concentration_Ratio (%)"] = (
        (detailed_table["Concentration_Count"] / detailed_table["Total_Samples"]) * 100
    ).round(2)
    detailed_table["Relax_Ratio (%)"] = (
        (detailed_table["Relax_Count"] / detailed_table["Total_Samples"]) * 100
    ).round(2)

    # 教師なしクラスタリング評価指標
    max_counts = detailed_table[["Concentration_Count", "Relax_Count"]].max(axis=1).sum()
    purity = float(max_counts / detailed_table["Total_Samples"].sum())

    y_true = df_clean["label_bin"].values
    y_pred = df_clean["cluster"].values
    ari = float(adjusted_rand_score(y_true, y_pred))
    nmi = float(normalized_mutual_info_score(y_true, y_pred))

    metrics_summary = {
        "purity": round(purity, 4),
        "ari": round(ari, 4),
        "nmi": round(nmi, 4),
        "total_evaluated_samples": len(df_clean),
    }

    return detailed_table, metrics_summary


def aggregate_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    各クラスタにおける心拍数 (BPM)、瞬目比率 (EAR)、頭部・視線の動き (分散) の
    平均値および標準偏差を集計します。
    """
    # 物理指標および PLS 潜在成分
    pls_cols = [c for c in df.columns if c.startswith(("vis_pls_", "bio_pls_"))]
    target_features = [
        "bpm_mean",
        "ear_mean",
        "head_movement_var",
        "gaze_movement_var",
        "head_pitch_diff_var",
        "head_yaw_diff_var",
        "head_roll_diff_var",
        "gaze_v_deg_diff_var",
        "gaze_h_deg_diff_var",
    ] + pls_cols

    available_features = [f for f in target_features if f in df.columns]

    if not available_features:
        log.warning("集計対象の物理生体特徴量列が見つかりません。")
        return pd.DataFrame()

    summary_rows = []
    clusters = sorted(df["cluster"].unique())

    for c in clusters:
        sub = df[df["cluster"] == c]
        row = {"cluster": c, "sample_count": len(sub)}

        for feat in available_features:
            vals = sub[feat].dropna()
            if len(vals) > 0:
                row[f"{feat}_mean"] = vals.mean()
                row[f"{feat}_std"] = vals.std()
            else:
                row[f"{feat}_mean"] = np.nan
                row[f"{feat}_std"] = np.nan

        summary_rows.append(row)

    profile_df = pd.DataFrame(summary_rows).set_index("cluster")
    return profile_df
