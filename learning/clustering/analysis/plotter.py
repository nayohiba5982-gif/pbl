from pathlib import Path
import numpy as np
import pandas as pd

from config_loader import log


def plot_cluster_features(
    df_combined: pd.DataFrame,
    crosstab: pd.DataFrame,
    profiles: pd.DataFrame,
    save_path: Path,
    window_size: str = "10s",
) -> None:
    """
    各クラスタの特徴量（タスク比率、心拍数BPM、瞬目比率EAR、頭部・視線の動き分散）を
    比較・確認できる4パネルグラフを描画・保存します。
    """
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        # フォント設定（文字化け回避）
        plt.rcParams["font.sans-serif"] = [
            "DejaVu Sans",
            "Noto Sans CJK JP",
            "TakaoPGothic",
            "IPAGothic",
            "sans-serif",
        ]
        plt.rcParams["axes.unicode_minus"] = False

        clusters = sorted(df_combined["cluster"].unique())
        n_clusters = len(clusters)
        cluster_tags = [f"Cluster {c}" for c in clusters]

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        cmap = mpl.colormaps.get_cmap("tab10" if n_clusters <= 10 else "tab20")

        # --- パネル 1: 実験タスクとのクロス集計 (積み上げバープロット) ---
        ax1 = axes[0, 0]
        if not crosstab.empty and "Concentration_Count" in crosstab.columns:
            c_counts = [crosstab.loc[c, "Concentration_Count"] for c in clusters]
            r_counts = [crosstab.loc[c, "Relax_Count"] for c in clusters]
            totals = [crosstab.loc[c, "Total_Samples"] for c in clusters]

            c_pct = [c / t * 100 if t > 0 else 0 for c, t in zip(c_counts, totals)]
            r_pct = [r / t * 100 if t > 0 else 0 for r, t in zip(r_counts, totals)]

            ax1.bar(
                cluster_tags,
                c_pct,
                label="Task: Concentration (Lecture)",
                color="#e74c3c",
                alpha=0.85,
            )
            ax1.bar(
                cluster_tags,
                r_pct,
                bottom=c_pct,
                label="Task: Relax (Campfire)",
                color="#3498db",
                alpha=0.85,
            )

            for i, (cp, rp) in enumerate(zip(c_pct, r_pct)):
                if cp > 5:
                    ax1.text(
                        i,
                        cp / 2,
                        f"{cp:.1f}%",
                        ha="center",
                        va="center",
                        color="white",
                        fontweight="bold",
                        fontsize=10,
                    )
                if rp > 5:
                    ax1.text(
                        i,
                        cp + rp / 2,
                        f"{rp:.1f}%",
                        ha="center",
                        va="center",
                        color="white",
                        fontweight="bold",
                        fontsize=10,
                    )

            ax1.set_ylabel("Composition Ratio (%)", fontsize=11)
            ax1.set_title("1. Task Composition per Cluster", fontsize=12, fontweight="bold")
            ax1.set_ylim(0, 100)
            ax1.legend(loc="upper right", framealpha=0.9)
            ax1.grid(axis="y", linestyle=":", alpha=0.6)
        else:
            ax1.text(0.5, 0.5, "Task Label Not Available", ha="center", va="center")
            ax1.set_title("1. Task Composition (N/A)")

        # --- パネル 2: 心拍数 (BPM) のクラスタ別比較 ---
        ax2 = axes[0, 1]
        if not profiles.empty and "bpm_mean_mean" in profiles.columns:
            bpm_means = [profiles.loc[c, "bpm_mean_mean"] for c in clusters]
            bpm_stds = [profiles.loc[c, "bpm_mean_std"] for c in clusters]

            ax2.bar(
                cluster_tags,
                bpm_means,
                yerr=bpm_stds,
                capsize=5,
                color=[cmap(i) for i in range(n_clusters)],
                alpha=0.8,
                edgecolor="black",
            )
            ax2.set_ylabel("Heart Rate (BPM)", fontsize=11)
            ax2.set_title("2. Heart Rate Mean ± Std (BPM)", fontsize=12, fontweight="bold")
            ax2.grid(axis="y", linestyle=":", alpha=0.6)

            for i, m in enumerate(bpm_means):
                if not np.isnan(m):
                    ax2.text(
                        i,
                        m * 0.5,
                        f"{m:.1f} BPM",
                        ha="center",
                        va="center",
                        color="white",
                        fontweight="bold",
                        fontsize=11,
                    )
        else:
            ax2.text(0.5, 0.5, "BPM Data Not Available", ha="center", va="center")
            ax2.set_title("2. Heart Rate (N/A)")

        # --- パネル 3: 瞬目比率 (EAR) のクラスタ別比較 ---
        ax3 = axes[1, 0]
        if not profiles.empty and "ear_mean_mean" in profiles.columns:
            ear_means = [profiles.loc[c, "ear_mean_mean"] for c in clusters]
            ear_stds = [profiles.loc[c, "ear_mean_std"] for c in clusters]

            ax3.bar(
                cluster_tags,
                ear_means,
                yerr=ear_stds,
                capsize=5,
                color=[cmap(i) for i in range(n_clusters)],
                alpha=0.8,
                edgecolor="black",
            )
            ax3.set_ylabel("EAR Ratio", fontsize=11)
            ax3.set_title("3. Eye Aspect Ratio Mean ± Std (EAR)", fontsize=12, fontweight="bold")
            ax3.grid(axis="y", linestyle=":", alpha=0.6)

            for i, m in enumerate(ear_means):
                if not np.isnan(m):
                    ax3.text(
                        i,
                        m * 0.5,
                        f"{m:.3f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontweight="bold",
                        fontsize=11,
                    )
        else:
            ax3.text(0.5, 0.5, "EAR Data Not Available", ha="center", va="center")
            ax3.set_title("3. Eye Aspect Ratio (N/A)")

        # --- パネル 4: 頭部・視線の動き（時間分散）の比較 ---
        ax4 = axes[1, 1]
        has_head = not profiles.empty and "head_movement_var_mean" in profiles.columns
        has_gaze = not profiles.empty and "gaze_movement_var_mean" in profiles.columns

        if has_head or has_gaze:
            x_pos = np.arange(n_clusters)
            width = 0.35

            if has_head:
                h_means = [profiles.loc[c, "head_movement_var_mean"] for c in clusters]
                ax4.bar(
                    x_pos - width / 2,
                    h_means,
                    width,
                    label="Head Pose Var",
                    color="#f39c12",
                    alpha=0.85,
                )
            if has_gaze:
                g_means = [profiles.loc[c, "gaze_movement_var_mean"] for c in clusters]
                ax4.bar(
                    x_pos + width / 2,
                    g_means,
                    width,
                    label="Gaze Var",
                    color="#9b59b6",
                    alpha=0.85,
                )

            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(cluster_tags)
            ax4.set_ylabel("Variance", fontsize=11)
            ax4.set_title("4. Head & Gaze Movement Variance", fontsize=12, fontweight="bold")
            ax4.legend(loc="upper right", framealpha=0.9)
            ax4.grid(axis="y", linestyle=":", alpha=0.6)
        else:
            ax4.text(0.5, 0.5, "Movement Data Not Available", ha="center", va="center")
            ax4.set_title("4. Movement Variance (N/A)")

        plt.suptitle(
            f"Cluster Feature Profiles (Window: {window_size})",
            fontsize=14,
            fontweight="bold",
            y=0.99,
        )
        plt.tight_layout()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        log.info(f"📊 クラスタ特徴量可視化プロットを保存しました: {save_path}")

    except Exception as e:
        log.warning(f"可視化プロット作成中にエラーが発生しました: {e}")
