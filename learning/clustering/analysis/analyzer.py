import sys
from pathlib import Path
import numpy as np
import pandas as pd

# パス設定（単体実行時のフォールバック）
_LEARNING_DIR = Path(__file__).resolve().parent.parent
if str(_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(_LEARNING_DIR))

from config_loader import load_config, log

try:
    from .metrics import DEFAULT_FEATURE_COLS, attach_physical_metrics_to_df
    from .statistics import compute_cross_tabulation, aggregate_cluster_profiles
    from .plotter import plot_cluster_features
    from .reporter import generate_cluster_report
except (ImportError, ValueError):
    try:
        from metrics import DEFAULT_FEATURE_COLS, attach_physical_metrics_to_df
        from statistics import compute_cross_tabulation, aggregate_cluster_profiles
        from plotter import plot_cluster_features
        from reporter import generate_cluster_report
    except ImportError:
        from clustering.analysis.metrics import DEFAULT_FEATURE_COLS, attach_physical_metrics_to_df
        from clustering.analysis.statistics import compute_cross_tabulation, aggregate_cluster_profiles
        from clustering.analysis.plotter import plot_cluster_features
        from clustering.analysis.reporter import generate_cluster_report


class ClusterAnalyzer:
    """
    GMM クラスタリングで得られた幾何学的クラスタ（cluster 0, 1, ...）に対して、
    心拍数（BPM）、瞬目比率（EAR）、頭部・視線の動き（分散）などの生体シグナル平均値および
    実験時タスク（集中タスク区間 / 休憩区間）とのクロス集計を行い、
    クラスタごとの特徴量を集計・可視化・レポート出力するクラス。
    （※完全リークフリー設計のため、クロス集計および意味付けは split_dfs["train"] のみに限定）
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.paths = cfg["paths"]
        self.dataset_cfg = cfg.get("dataset", {})
        self.clustering_cfg = cfg.get("clustering", {})
        self.analysis_cfg = cfg.get("cluster_analysis", {})

        self.window_size = self.dataset_cfg.get("window_size", "10s")
        data_root = self.paths.get("data_root", Path(__file__).resolve().parent.parent.parent / "data")
        self.data_dir = Path(data_root) / self.window_size
        self.data_dir.mkdir(parents=True, exist_ok=True)
        input_data_root = self.paths.get("input_data_root", Path(__file__).resolve().parent.parent.parent.parent / "preprocess" / "data" / "windowed")
        self.input_data_dir = Path(input_data_root) / self.window_size

        # 特徴量列名設定
        self.feature_cols = self.analysis_cfg.get("feature_cols", DEFAULT_FEATURE_COLS)
        self.metrics_cfg = self.analysis_cfg.get("metrics", {})
        self.bpm_col = self.metrics_cfg.get("bpm_col", "bpm")
        self.ear_col = self.metrics_cfg.get("ear_col", "ear_ratio")
        self.head_pose_cols = self.metrics_cfg.get(
            "head_pose_cols",
            ["head_pitch_diff", "head_yaw_diff", "head_roll_diff"],
        )
        self.gaze_cols = self.metrics_cfg.get(
            "gaze_cols",
            ["gaze_v_deg_diff", "gaze_h_deg_diff"],
        )

        # 出力ファイル設定
        self.save_plot = self.analysis_cfg.get("save_plot", True)
        self.plot_filename = self.analysis_cfg.get(
            "plot_filename", f"cluster_physical_interpretation_{self.window_size}.png"
        )
        self.report_filename = self.analysis_cfg.get(
            "report_filename", f"cluster_analysis_report_{self.window_size}.md"
        )
        self.profiles_filename = self.analysis_cfg.get(
            "profiles_filename", f"cluster_profiles_{self.window_size}.csv"
        )

    # =========================================================================
    # データ読み込み
    # =========================================================================

    def load_clustered_data(self) -> dict[str, pd.DataFrame] | None:
        """
        Train, Val, Test のクラスタリング済みメタデータ (meta_*_clustered.csv) をロードします。
        """
        split_names = ["train", "val", "test"]
        split_dfs: dict[str, pd.DataFrame] = {}

        for s_name in split_names:
            meta_path = self.data_dir / f"meta_{s_name}_clustered.csv"
            if meta_path.exists():
                df = pd.read_csv(meta_path)
                if "cluster" in df.columns:
                    split_dfs[s_name] = df
                    log.info(f"ロード完了: {meta_path.name} (行数: {len(df)})")
                else:
                    log.warning(f"⚠️ {meta_path.name} に 'cluster' 列が存在しません。")
            else:
                log.debug(f"ファイル未存在: {meta_path.name}")

        if "train" not in split_dfs:
            # フォールバック 1: PLS フォルダ内の Train 専用クラスタリング済みデータを探索
            learning_root = Path(__file__).resolve().parent.parent
            pls_dir = learning_root / "data" / "pls"
            pls_train_cand = pls_dir / f"pls_features_{self.window_size}_train_clustered.csv"
            if pls_train_cand.exists():
                df = pd.read_csv(pls_train_cand)
                if "cluster" in df.columns:
                    split_dfs["train"] = df
                    log.info(f"✓ PLS Train クラスタリング済みデータをロード: {pls_train_cand} (行数: {len(df)})")

            if "train" not in split_dfs:
                pls_cands = [
                    pls_dir / f"pls_features_{self.window_size}_clustered.csv",
                    pls_dir / f"meta_train_clustered.csv",
                    pls_dir / "pls_features_clustered.csv",
                ]
                for p_cand in pls_cands:
                    if p_cand.exists():
                        df = pd.read_csv(p_cand)
                        if "cluster" in df.columns:
                            # 結合版 CSV に split 列がある場合は Train のみを厳格に抽出
                            if "split" in df.columns and (df["split"] == "train").any():
                                df = df[df["split"] == "train"].reset_index(drop=True)
                            split_dfs["train"] = df
                            log.info(f"✓ PLS クラスタリング済みデータ (Train限定) をロード: {p_cand} (行数: {len(df)})")
                            break

        if not split_dfs:
            log.error(
                f"❌ クラスタリング済みデータ (meta_train_clustered.csv) が見つかりません: {self.data_dir}\n"
                f"   先に 'python main.py --stage clustering' を実行してください。"
            )
            return None

        return split_dfs

    def load_raw_windows(self) -> tuple[np.ndarray | None, dict | None, pd.DataFrame | None]:
        """
        X_windows.npy と split_indices.npz、および元 metadata.csv をロードします。
        生体指標（BPM, EAR, 分散）の算出に使用します。
        """
        x_cands = [
            self.input_data_dir / "X_windows.npy",
            self.input_data_dir.parent / "combined" / self.window_size / "X_windows.npy",
            self.data_dir / "X_windows.npy",
            self.data_dir.parent / "combined" / self.window_size / "X_windows.npy",
        ]
        x_path = next((p for p in x_cands if p.exists()), x_cands[0])

        m_cands = [
            self.input_data_dir / "metadata.csv",
            self.input_data_dir.parent / "combined" / self.window_size / "metadata.csv",
            self.data_dir / "metadata.csv",
            self.data_dir.parent / "combined" / self.window_size / "metadata.csv",
        ]
        meta_path = next((p for p in m_cands if p.exists()), m_cands[0])

        idx_cands = [
            self.data_dir / "split_indices.npz",
            self.input_data_dir / "split_indices.npz",
        ]
        indices_path = next((p for p in idx_cands if p.exists()), idx_cands[0])

        X = None
        split_indices = None
        meta_raw = None

        if x_path.exists():
            try:
                X = np.load(x_path)
                log.info(f"X_windows.npy ロード完了: shape = {X.shape}")
            except Exception as e:
                log.warning(f"X_windows.npy の読み込みに失敗しました: {e}")

        if meta_path.exists():
            try:
                meta_raw = pd.read_csv(meta_path)
                log.info(f"元 metadata.csv ロード完了: shape = {meta_raw.shape}")
            except Exception as e:
                log.warning(f"元 metadata.csv の読み込みに失敗しました: {e}")

        if indices_path.exists():
            try:
                split_indices = dict(np.load(indices_path))
                log.info("split_indices.npz ロード完了")
            except Exception as e:
                log.warning(f"split_indices.npz の読み込みに失敗しました: {e}")

        return X, split_indices, meta_raw

    # =========================================================================
    # メイン実行メソッド
    # =========================================================================

    def run(self) -> bool:
        """
        GMM クラスタの特徴量集計・分析を実行します。
        【厳格なデータリークフリー設計】
        「集中/非集中の意味付け」のためのクロス集計・生体プロファイル集計対象は
        split_dfs["train"] のみに限定し、Val / Test データが意思決定に一切混入しないようにします。
        """
        log.info("=" * 60)
        log.info(f"▶ GMM Cluster Analysis (クラスタ別特徴量集計・分析) 開始 [Window: {self.window_size}]")
        log.info("=" * 60)

        # 1. クラスタリング済みメタデータの読み込み
        split_dfs = self.load_clustered_data()
        if split_dfs is None or "train" not in split_dfs:
            log.error("❌ Train スプリットのクラスタリングデータ (split_dfs['train']) が存在しません。")
            return False

        # Train スプリットのみに厳格限定 (Test / Val を完全除外)
        train_only_dfs = {"train": split_dfs["train"].copy()}
        log.info(
            f"🔒 [Leakage-Free] 集中/非集中の意味付け・クロス集計対象を split_dfs['train'] のみ "
            f"(N={len(train_only_dfs['train'])}) に限定します (Test/Val データ完全除外)"
        )

        # 2. X_windows.npy からの生体指標の算出（存在する場合）および Train データへの結合
        X_all, split_indices, meta_raw = self.load_raw_windows()
        df_train = attach_physical_metrics_to_df(
            split_dfs=train_only_dfs,
            X_all=X_all,
            split_indices=split_indices,
            feature_cols=self.feature_cols,
            bpm_col=self.bpm_col,
            ear_col=self.ear_col,
            head_pose_cols=self.head_pose_cols,
            gaze_cols=self.gaze_cols,
            meta_raw=meta_raw,
        )

        clusters = sorted(df_train["cluster"].unique())
        n_clusters = len(clusters)
        log.info(f"対象クラスタ数: k = {n_clusters} (クラスタ: {clusters}) [Train N={len(df_train)}]")

        # 3. 実験タスクとのクロス集計 (split_dfs["train"] のみ)
        log.info("実験タスク (集中区間 / 休憩区間) とのクロス集計を実行中 (split_dfs['train'] のみ)...")
        crosstab, metrics_summary = compute_cross_tabulation(df_train)
        if not crosstab.empty:
            log.info("\n" + crosstab.to_string())
            log.info(
                f"評価メトリクス (Train): Purity = {metrics_summary.get('purity')}, "
                f"ARI = {metrics_summary.get('ari')}, NMI = {metrics_summary.get('nmi')}"
            )

        # 4. 生体指標プロファイルの集計 (split_dfs["train"] のみ)
        log.info("生体指標 (心拍数, EAR, 頭部・視線分散) のクラスタ別プロファイルを集計中 (split_dfs['train'] のみ)...")
        profiles = aggregate_cluster_profiles(df_train)
        pls_dir = Path(__file__).resolve().parent.parent / "data" / "pls"

        if not profiles.empty:
            log.info("\n" + profiles.to_string())

            # クラスタプロファイル CSV の保存
            profiles_path = self.data_dir / self.profiles_filename
            profiles.to_csv(profiles_path)
            log.info(f"💾 クラスタ別特徴量プロファイル CSV を保存しました: {profiles_path}")
            if pls_dir.exists():
                profiles.to_csv(pls_dir / f"cluster_profiles_{self.window_size}.csv")

        # 5. 可視化グラフの保存 (split_dfs["train"] のみ)
        if self.save_plot:
            plot_path = self.data_dir / self.plot_filename
            plot_cluster_features(
                df_combined=df_train,
                crosstab=crosstab,
                profiles=profiles,
                save_path=plot_path,
                window_size=self.window_size,
            )
            if pls_dir.exists():
                plot_cluster_features(
                    df_combined=df_train,
                    crosstab=crosstab,
                    profiles=profiles,
                    save_path=pls_dir / f"cluster_physical_interpretation_{self.window_size}.png",
                    window_size=self.window_size,
                )

        # 6. 詳細 Markdown レポートの保存
        report_path = self.data_dir / self.report_filename
        generate_cluster_report(
            crosstab=crosstab,
            profiles=profiles,
            metrics_summary=metrics_summary,
            save_path=report_path,
            window_size=self.window_size,
        )
        if pls_dir.exists():
            generate_cluster_report(
                crosstab=crosstab,
                profiles=profiles,
                metrics_summary=metrics_summary,
                save_path=pls_dir / f"cluster_analysis_report_{self.window_size}.md",
                window_size=self.window_size,
            )

        log.info("=" * 60)
        log.info("📊 【クラスタ別特徴量の集計が完了しました (Train スプリット限定)】")
        log.info(f"   各クラスタへのラベル付与は、レポート ({report_path.name}) や")
        log.info("   可視化プロットを確認の上、ユーザ自身で決定してください。")
        log.info("=" * 60)

        return True


def run_cluster_analysis(cfg: dict = None) -> bool:
    """外部モジュールから呼び出し可能なヘルパー関数"""
    if cfg is None:
        cfg = load_config()
    analyzer = ClusterAnalyzer(cfg)
    return analyzer.run()


if __name__ == "__main__":
    cfg = load_config()
    run_cluster_analysis(cfg)
