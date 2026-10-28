"""
distribution_checker.py
顔ランドマーク前処理における Zスコア標準化後の特徴量分布を包括的に分析・可視化するモジュール。

【主な機能】
1. 記述統計量 & 正規性評価: 平均、分散、歪度 (Skewness)、尖度 (Kurtosis)、外れ値率 (|z|>3, |z|>5)、正規性検定
2. 分布ヒストグラム & KDE + 標準正規分布 N(0, 1) ガウス曲線オーバーレイ
3. Q-Q プロット (理論正規分布との分位点比較)
4. 被験者間分布比較 (Boxplot / 箱ひげ図による個人内スケーリングの整合性確認)
5. Zスコア時系列波形プレビュー (代表データの推移確認)
"""

import math
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# validation ルートを sys.path に追加
_LM_DIR = Path(__file__).resolve().parent
_VAL_DIR = _LM_DIR.parent
if str(_VAL_DIR) not in sys.path:
    sys.path.insert(0, str(_VAL_DIR))

from config_loader import log


class LMDistributionChecker:
    """Zスコア化後の特徴量分布の分析・可視化を行うクラス"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dist_cfg = cfg.get("distribution", {})
        self.target_cols = self.dist_cfg.get("target_columns")
        self.outlier_thresholds = self.dist_cfg.get("outlier_thresholds", [3.0, 5.0])
        self.hist_bins = int(self.dist_cfg.get("hist_bins", 50))

        vis_cfg = cfg.get("visualization", {})
        self.save_plots = vis_cfg.get("save_plots", True)
        self.save_csv = vis_cfg.get("save_csv", True)
        self.dpi = int(vis_cfg.get("dpi", 300))
        self.plot_dist_fn = vis_cfg.get("plot_distribution_filename", "lm_zscore_distribution.png")
        self.plot_qq_fn = vis_cfg.get("plot_qq_filename", "lm_zscore_qq_plots.png")
        self.plot_box_fn = vis_cfg.get("plot_boxplots_filename", "lm_subject_boxplots.png")
        self.plot_time_fn = vis_cfg.get("plot_timeline_filename", "lm_timeline_preview.png")

        paths_cfg = cfg.get("paths", {})
        self.output_dir = Path(paths_cfg.get("output_dir", "result"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_z_columns(self, df: pd.DataFrame) -> List[str]:
        """データフレームから Zスコア標準化済みカラム一覧を取得 (lm_*_zやcentroid_z等の3D座標を除外)"""
        if self.target_cols:
            return [c for c in self.target_cols if c in df.columns]
        
        # *_smooth_z を優先、または *_z かつ lm_ や centroid_ で始まらないカラム
        smooth_z = [c for c in df.columns if c.endswith("_smooth_z")]
        if smooth_z:
            return smooth_z
        z_cols = [
            c for c in df.columns
            if (c.endswith("_z") or "_z_" in c) and not c.startswith("lm_") and not c.startswith("centroid_")
        ]
        return z_cols

    def compute_distribution_metrics(
        self,
        dfs: Dict[str, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """各被験者および全被験者統合での各特徴量の統計量・分布指標を算出"""
        subject_metrics = []
        combined_dfs = []

        for sub_id, df in dfs.items():
            z_cols = self.extract_z_columns(df)
            for col in z_cols:
                series = df[col].dropna()
                if len(series) == 0:
                    continue

                m_val = float(series.mean())
                s_val = float(series.std())
                skew_val = float(stats.skew(series))
                kurt_val = float(stats.kurtosis(series))

                # 外れ値比率
                out_3 = float((series.abs() > self.outlier_thresholds[0]).mean())
                out_5 = float((series.abs() > self.outlier_thresholds[1]).mean()) if len(self.outlier_thresholds) > 1 else 0.0

                subject_metrics.append({
                    "subject_id": sub_id,
                    "column": col,
                    "count": len(series),
                    "mean": m_val,
                    "std": s_val,
                    "skewness": skew_val,
                    "kurtosis": kurt_val,
                    "min": float(series.min()),
                    "q25": float(series.quantile(0.25)),
                    "median": float(series.median()),
                    "q75": float(series.quantile(0.75)),
                    "max": float(series.max()),
                    f"outlier_ratio_gt_{self.outlier_thresholds[0]}": out_3,
                    f"outlier_ratio_gt_{self.outlier_thresholds[1]}": out_5,
                })

            df_sub = df[z_cols].copy()
            df_sub["subject_id"] = sub_id
            combined_dfs.append(df_sub)

        sub_df = pd.DataFrame(subject_metrics)

        # 全体統合統計
        overall_metrics = []
        if combined_dfs:
            combined_all = pd.concat(combined_dfs, ignore_index=True)
            all_z_cols = [c for c in combined_all.columns if c != "subject_id"]
            for col in all_z_cols:
                series = combined_all[col].dropna()
                if len(series) == 0:
                    continue

                m_val = float(series.mean())
                s_val = float(series.std())
                skew_val = float(stats.skew(series))
                kurt_val = float(stats.kurtosis(series))
                out_3 = float((series.abs() > self.outlier_thresholds[0]).mean())
                out_5 = float((series.abs() > self.outlier_thresholds[1]).mean()) if len(self.outlier_thresholds) > 1 else 0.0

                # 正規性検定 (サンプル数が多い場合は subsample して検定)
                sample_for_test = series.sample(min(len(series), 5000), random_state=42)
                try:
                    k2_stat, p_val = stats.normaltest(sample_for_test)
                except Exception:
                    k2_stat, p_val = np.nan, np.nan

                overall_metrics.append({
                    "column": col,
                    "total_samples": len(series),
                    "mean": m_val,
                    "std": s_val,
                    "skewness": skew_val,
                    "kurtosis": kurt_val,
                    "min": float(series.min()),
                    "q25": float(series.quantile(0.25)),
                    "median": float(series.median()),
                    "q75": float(series.quantile(0.75)),
                    "max": float(series.max()),
                    f"outlier_ratio_gt_{self.outlier_thresholds[0]}": out_3,
                    f"outlier_ratio_gt_{self.outlier_thresholds[1]}": out_5,
                    "normaltest_stat": float(k2_stat),
                    "normaltest_pvalue": float(p_val),
                })
        overall_df = pd.DataFrame(overall_metrics)

        if self.save_csv:
            sub_csv_path = self.output_dir / "lm_zscore_metrics_by_subject.csv"
            over_csv_path = self.output_dir / "lm_zscore_metrics_overall.csv"
            sub_df.to_csv(sub_csv_path, index=False)
            overall_df.to_csv(over_csv_path, index=False)
            log.info(f"✓ 被験者別統計サマリー保存: {sub_csv_path}")
            log.info(f"✓ 全体統合統計サマリー保存: {over_csv_path}")

        return sub_df, overall_df

    def plot_distributions(self, dfs: Dict[str, pd.DataFrame]) -> Path:
        """各特徴量の分布ヒストグラム + KDE + 標準正規分布 N(0, 1) を描画"""
        combined_dfs = []
        for _, df in dfs.items():
            z_cols = self.extract_z_columns(df)
            combined_dfs.append(df[z_cols])
        combined = pd.concat(combined_dfs, ignore_index=True)
        z_cols = [c for c in combined.columns]

        if not z_cols:
            log.warning("可視化対象の Zスコア カラムが見つかりません。")
            return self.output_dir / self.plot_dist_fn

        n_cols_plot = 3
        n_rows_plot = math.ceil(len(z_cols) / n_cols_plot)
        fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(15, 3.5 * n_rows_plot), tight_layout=True)
        if len(z_cols) == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        x_norm = np.linspace(-4, 4, 200)
        y_norm = stats.norm.pdf(x_norm, 0, 1)

        for i, col in enumerate(z_cols):
            ax = axes[i]
            data = combined[col].dropna().values
            # 外れ値クリップして描画領域を [-5, 5] に集中
            data_vis = data[(data >= -5) & (data <= 5)]

            # ヒストグラム
            ax.hist(data_vis, bins=self.hist_bins, density=True, alpha=0.55, color="#1f77b4", edgecolor="black", linewidth=0.5, label="Empirical")

            # KDE (Kernel Density Estimation)
            if len(data) > 10:
                try:
                    kde = stats.gaussian_kde(data_vis)
                    ax.plot(x_norm, kde(x_norm), color="#1f77b4", lw=2, label="KDE")
                except Exception:
                    pass

            # 標準正規分布 N(0, 1)
            ax.plot(x_norm, y_norm, "r--", lw=1.8, label=r"Standard $\mathcal{N}(0, 1)$")

            m = float(np.mean(data))
            s = float(np.std(data))
            sk = float(stats.skew(data))
            kt = float(stats.kurtosis(data))

            text_str = rf"$\mu$={m:.2f}, $\sigma$={s:.2f}" + "\n" + f"Skew={sk:.2f}, Kurt={kt:.2f}"
            ax.text(0.05, 0.95, text_str, transform=ax.transAxes, verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#ccc"),
                    fontsize=8.5)

            clean_title = col.replace("_smooth_z", "").replace("_z", "")
            ax.set_title(f"{clean_title} (z-score)", fontsize=10, fontweight="bold")
            ax.set_xlim(-5, 5)
            ax.set_xlabel("Z-score")
            ax.set_ylabel("Density")
            ax.grid(True, linestyle=":", alpha=0.6)
            if i == 0:
                ax.legend(loc="upper right", fontsize=8)

        # 残りのサブプロットを非表示
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Landmark Features Z-Score Empirical Distribution vs Standard Normal Distribution", fontsize=14, fontweight="bold", y=1.01)
        out_path = self.output_dir / self.plot_dist_fn
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        log.info(f"✓ Zスコア分布ヒストグラムプロット保存: {out_path}")
        return out_path

    def plot_qq_plots(self, dfs: Dict[str, pd.DataFrame]) -> Path:
        """各特徴量の Q-Q プロット (理論正規分布との比較) を描画"""
        combined_dfs = []
        for _, df in dfs.items():
            z_cols = self.extract_z_columns(df)
            combined_dfs.append(df[z_cols])
        combined = pd.concat(combined_dfs, ignore_index=True)
        z_cols = [c for c in combined.columns]

        if not z_cols:
            return self.output_dir / self.plot_qq_fn

        n_cols_plot = 3
        n_rows_plot = math.ceil(len(z_cols) / n_cols_plot)
        fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(15, 3.5 * n_rows_plot), tight_layout=True)
        if len(z_cols) == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, col in enumerate(z_cols):
            ax = axes[i]
            data = combined[col].dropna().values
            stats.probplot(data, dist="norm", plot=ax)
            clean_title = col.replace("_smooth_z", "").replace("_z", "")
            ax.set_title(f"Q-Q Plot: {clean_title}", fontsize=10, fontweight="bold")
            ax.grid(True, linestyle=":", alpha=0.6)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Landmark Features Z-Score Normal Q-Q Plots", fontsize=14, fontweight="bold", y=1.01)
        out_path = self.output_dir / self.plot_qq_fn
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        log.info(f"✓ Q-Q プロット保存: {out_path}")
        return out_path

    def plot_subject_boxplots(self, dfs: Dict[str, pd.DataFrame]) -> Path:
        """被験者ごとの Zスコア分布の均一性を箱ひげ図で視覚比較"""
        subjects = sorted(list(dfs.keys()))
        first_sub = subjects[0]
        z_cols = self.extract_z_columns(dfs[first_sub])
        if not z_cols:
            return self.output_dir / self.plot_box_fn

        n_features = len(z_cols)
        n_cols_plot = 3
        n_rows_plot = math.ceil(n_features / n_cols_plot)
        fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(15, 3.5 * n_rows_plot), tight_layout=True)
        if n_features == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, col in enumerate(z_cols):
            ax = axes[i]
            sub_data = []
            for s in subjects:
                df_s = dfs[s]
                if col in df_s.columns:
                    sub_data.append(df_s[col].dropna().values)
                else:
                    sub_data.append(np.array([]))

            try:
                bp = ax.boxplot(sub_data, tick_labels=subjects, patch_artist=True, showfliers=False, medianprops=dict(color="red", linewidth=1.5))
            except TypeError:
                bp = ax.boxplot(sub_data, labels=subjects, patch_artist=True, showfliers=False, medianprops=dict(color="red", linewidth=1.5))
            for box in bp['boxes']:
                box.set_facecolor('#d9e6f2')
                box.set_edgecolor('#1f77b4')

            ax.axhline(0.0, color="gray", linestyle="--", alpha=0.7)
            ax.axhline(1.0, color="red", linestyle=":", alpha=0.5)
            ax.axhline(-1.0, color="red", linestyle=":", alpha=0.5)
            clean_name = col.replace("_smooth_z", "").replace("_z", "")
            ax.set_title(f"Subject Comparison: {clean_name}", fontsize=10, fontweight="bold")
            ax.set_ylabel("Z-score (IQR)")
            ax.set_ylim(-3.5, 3.5)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.tick_params(axis='x', rotation=45, labelsize=7)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Cross-Subject Z-Score Consistency (Median & IQR Boxplots)", fontsize=14, fontweight="bold", y=1.01)
        out_path = self.output_dir / self.plot_box_fn
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        log.info(f"✓ 被験者間分布比較プロット保存: {out_path}")
        return out_path

    def plot_timeline_preview(self, dfs: Dict[str, pd.DataFrame]) -> Path:
        """代表被験者の Zスコア時系列波形プレビューを描画"""
        if not dfs:
            return self.output_dir / self.plot_time_fn

        # 最初の被験者を選択
        first_sub = sorted(list(dfs.keys()))[0]
        df = dfs[first_sub]
        z_cols = self.extract_z_columns(df)
        if not z_cols:
            return self.output_dir / self.plot_time_fn

        time_col = "time" if "time" in df.columns else None
        if time_col:
            x_time = df[time_col].values
            x_label = "Time (sec)"
        else:
            x_time = np.arange(len(df))
            x_label = "Frame Index"

        # 最大6つの主要特徴量を選択して描画
        selected_cols = z_cols[:6]
        fig, axes = plt.subplots(len(selected_cols), 1, figsize=(14, 2.5 * len(selected_cols)), sharex=True, tight_layout=True)
        if len(selected_cols) == 1:
            axes = [axes]

        for i, col in enumerate(selected_cols):
            ax = axes[i]
            ax.plot(x_time, df[col].values, color="#1f77b4", lw=1.2)
            ax.axhline(0.0, color="black", linestyle="--", alpha=0.5, lw=0.8)
            ax.axhline(2.0, color="orange", linestyle=":", alpha=0.7, lw=0.8)
            ax.axhline(-2.0, color="orange", linestyle=":", alpha=0.7, lw=0.8)
            clean_name = col.replace("_smooth_z", "").replace("_z", "")
            ax.set_ylabel(f"{clean_name}\n[Z-score]", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.6)

        axes[-1].set_xlabel(x_label, fontsize=10)
        fig.suptitle(f"Representative Landmark Z-Score Timeline Preview (Subject: {first_sub})", fontsize=13, fontweight="bold", y=1.01)
        out_path = self.output_dir / self.plot_time_fn
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        log.info(f"✓ 時系列プレビュープロット保存: {out_path}")
        return out_path

    def run_all(self, dfs: Dict[str, pd.DataFrame]) -> Dict[str, Union[pd.DataFrame, Path]]:
        """全分布確認・統計算出・プロット描画を実行"""
        log.info(f"▶ Zスコア分布確認・可視化開始 (被験者数: {len(dfs)})")
        sub_metrics, overall_metrics = self.compute_distribution_metrics(dfs)

        output_paths = {
            "subject_metrics_df": sub_metrics,
            "overall_metrics_df": overall_metrics,
        }

        if self.save_plots:
            output_paths["dist_plot"] = self.plot_distributions(dfs)
            output_paths["qq_plot"] = self.plot_qq_plots(dfs)
            output_paths["box_plot"] = self.plot_subject_boxplots(dfs)
            output_paths["timeline_plot"] = self.plot_timeline_preview(dfs)

        return output_paths
