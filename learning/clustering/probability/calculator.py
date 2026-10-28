"""
確率ラベル（ソフトターゲット y ∈ [0, 1]）算出オーケストレータ (Probability Label Calculator)
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

_LEARNING_DIR = Path(__file__).resolve().parent.parent.parent
if str(_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(_LEARNING_DIR))

try:
    from config_loader import load_config, log
except ImportError:
    import logging
    log = logging.getLogger("probability_label.calculator")
    def load_config(config_filename: str = "config.yaml") -> dict:
        return {}

from .core import compute_soft_labels, generate_default_weights
from .loader import load_probabilities_for_split
from .plotter import plot_soft_label_distribution


class ProbabilityLabelCalculator:
    """
    GMM クラスタ所属確率とクラスタ重みから、サンプルごとの期待値ソフトターゲット (y ∈ [0, 1]) を
    各 Split (Train, Val, Test) に対して一括算出・保存するオーケストレータクラス。
    """

    def __init__(self, cfg: dict | None = None, weights: list[float] | np.ndarray | None = None):
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.paths = cfg.get("paths", {})
        self.dataset_cfg = cfg.get("dataset", {})
        self.soft_cfg = cfg.get("soft_labeling", {})

        self.window_size = self.dataset_cfg.get("window_size", "10s")
        data_root = self.paths.get("data_root", _LEARNING_DIR / "data")
        self.data_dir = Path(data_root) / self.window_size
        self.data_dir.mkdir(parents=True, exist_ok=True)
        input_data_root = self.paths.get("input_data_root", _LEARNING_DIR.parent / "preprocess" / "data" / "windowed")
        self.input_data_dir = Path(input_data_root) / self.window_size

        # モデル保存ディレクトリ
        self.model_dir = Path(self.paths.get("model_dir", _LEARNING_DIR / "models"))

        # 重み設定 (引数 > config.yaml > None)
        if weights is not None:
            self.weights = np.asarray(weights, dtype=np.float64)
        elif "weights" in self.soft_cfg and self.soft_cfg["weights"] is not None:
            self.weights = np.asarray(self.soft_cfg["weights"], dtype=np.float64)
        else:
            self.weights = None

        # 保存設定
        self.target_col = self.soft_cfg.get("target_col", "y_soft")
        self.save_npy = self.soft_cfg.get("save_npy", True)
        self.save_csv = self.soft_cfg.get("save_csv", True)
        self.save_plot = self.soft_cfg.get("save_plot", True)
        self.plot_filename = self.soft_cfg.get(
            "plot_filename", f"soft_label_distribution_{self.window_size}.png"
        )

    def attach_explicit_features(self, df_meta: pd.DataFrame, split_name: str = "") -> pd.DataFrame:
        """
        GMM ソフトラベル算出後に、人間が解釈可能な明示的物理統計量 (20次元) を集約・算出し、
        下流タスクで直接利用できるようにメタデータ DataFrame へ結合します。
        """
        try:
            from features import compute_explicit_statistics
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

            if not x_path.exists():
                return df_meta

            X_all = np.load(x_path)
            meta_all = pd.read_csv(meta_path) if meta_path.exists() else None

            stat_matrix, stat_names = compute_explicit_statistics(X_all, meta_df=meta_all)
            stat_df = pd.DataFrame(stat_matrix, columns=stat_names)

            existing_stat_cols = [c for c in stat_names if c in df_meta.columns]
            if len(existing_stat_cols) == len(stat_names):
                return df_meta

            if meta_all is not None and "username" in meta_all.columns and "username" in df_meta.columns:
                stat_df["username"] = meta_all["username"].astype(str)
                pls_users = df_meta["username"].astype(str).unique()
                matched_stat = []
                for u in pls_users:
                    u_count = (df_meta["username"].astype(str) == u).sum()
                    u_stat = stat_df[stat_df["username"] == u].iloc[:u_count]
                    matched_stat.append(u_stat[stat_names])
                if matched_stat:
                    matched_stat_df = pd.concat(matched_stat, ignore_index=True)
                    if len(matched_stat_df) == len(df_meta):
                        for col in stat_names:
                            df_meta[col] = matched_stat_df[col].values
                        log.info(f"  ✓ GMM疑似クラスタリング後に明示的物理特徴量 ({len(stat_names)}次元) を結合しました")
                        return df_meta

            if len(stat_df) == len(df_meta):
                for col in stat_names:
                    df_meta[col] = stat_df[col].values
                log.info(f"  ✓ GMM疑似クラスタリング後に明示的物理特徴量 ({len(stat_names)}次元) を結合しました")
        except Exception as e:
            log.warning(f"⚠️ 明示的特徴量の結合中に警告が発生しました: {e}")

        return df_meta

    def _load_and_compute_initial(
        self,
        split_name: str,
        weights: np.ndarray | list[float] | None = None,
    ) -> dict | None:
        """指定された Split の所属確率を読み込み、初期ソフトラベルを算出します"""
        df_meta, probs, prob_cols = load_probabilities_for_split(
            data_dir=self.data_dir,
            split_name=split_name,
            model_dir=self.model_dir,
            window_size=self.window_size,
        )
        if probs is None or df_meta is None:
            return None

        n_clusters = probs.shape[1]

        if weights is not None and len(weights) == n_clusters:
            w = np.asarray(weights, dtype=np.float64)
        elif self.weights is not None and len(self.weights) == n_clusters:
            w = self.weights
        else:
            w = generate_default_weights(n_clusters)
            log.info(
                f"ℹ️ 重み (weights) が未指定または要素数が不一致のため、k={n_clusters} の等間隔デフォルト重みを適用します: "
                f"{[round(x, 3) for x in w.tolist()]}"
            )

        w_init = np.array([float(np.min(w)), float(np.max(w))], dtype=np.float64) if n_clusters == 2 else w
        y_soft = compute_soft_labels(probs, w_init)

        return {
            "split": split_name,
            "y_soft": y_soft,
            "df_meta": df_meta,
            "probs": probs,
            "weights": w_init,
            "n_clusters": n_clusters,
        }

    def _finalize_and_save_split(self, res: dict) -> dict:
        """アライメント後のソフトラベルをメタデータおよび NPY/CSV に保存します"""
        split_name = res["split"]
        y_soft = res["y_soft"]
        df_meta = res["df_meta"]
        probs = res["probs"]
        w = res["weights"]
        n_clusters = res["n_clusters"]

        log.info(f"--- [{split_name}] 期待値ソフトターゲット y の計算 ---")
        log.info(f"  クラスタ数: K = {n_clusters}")
        log.info(f"  適用重み: {w.tolist()}")
        log.info(
            f"  ✓ {split_name} y_soft 計算完了: 件数={len(y_soft)}, "
            f"平均={y_soft.mean():.4f}, 標準偏差={y_soft.std():.4f}, "
            f"最小={y_soft.min():.4f}, 中央値={np.median(y_soft):.4f}, 最大={y_soft.max():.4f}"
        )

        df_meta[self.target_col] = y_soft
        df_meta["prob_label"] = y_soft

        # GMM による疑似ソフトラベル確定後に、明示的物理特徴量 (20次元) を結合
        df_meta = self.attach_explicit_features(df_meta, split_name=split_name)
        res["df_meta"] = df_meta

        # 保存処理
        if self.save_csv:
            out_csv_path = self.data_dir / f"meta_{split_name}_clustered.csv"
            df_meta.to_csv(out_csv_path, index=False)
            log.info(f"  💾 メタデータCSV更新: {out_csv_path.name} (列 '{self.target_col}' を追加)")

            soft_csv_path = self.data_dir / f"meta_{split_name}_soft_labeled.csv"
            df_meta.to_csv(soft_csv_path, index=False)
            log.info(f"  💾 ソフトラベル付与CSV保存: {soft_csv_path.name}")

            pls_dir = _LEARNING_DIR / "data" / "pls"
            if pls_dir.exists():
                pls_split_csv = pls_dir / f"pls_features_{self.window_size}_{split_name}_soft_labeled.csv"
                df_meta.to_csv(pls_split_csv, index=False)
                log.info(f"  💾 PLS フォルダにもソフトラベル付与CSV保存 ({split_name}): {pls_split_csv.name}")

        if self.save_npy:
            y_soft_path = self.data_dir / f"y_soft_{split_name}.npy"
            np.save(y_soft_path, y_soft)
            log.info(f"  💾 ソフトラベル配列保存: {y_soft_path.name} {y_soft.shape}")

            y_std_path = self.data_dir / f"y_{split_name}.npy"
            np.save(y_std_path, y_soft)
            log.info(f"  💾 標準ターゲット配列保存: {y_std_path.name} {y_soft.shape}")

            pls_dir = _LEARNING_DIR / "data" / "pls"
            if pls_dir.exists():
                np.save(pls_dir / f"y_soft_{self.window_size}_{split_name}.npy", y_soft)
                log.info(f"  💾 PLS フォルダにも y_soft 配列保存 ({split_name}): y_soft_{self.window_size}_{split_name}.npy")

        return {
            "split": split_name,
            "y_soft": y_soft,
            "df_meta": df_meta,
            "probs": probs,
            "weights": w,
        }

    def process_split(
        self,
        split_name: str,
        weights: np.ndarray | list[float] | None = None,
    ) -> dict | None:
        """単一 Split のソフトラベル計算・保存"""
        res = self._load_and_compute_initial(split_name, weights=weights)
        if res is None:
            return None
        return self._finalize_and_save_split(res)

    def run(self, weights: list[float] | np.ndarray | None = None) -> dict[str, dict]:
        """
        Train, Val, Test の全 Split に対して確率ラベル計算を実行します。
        （Train データのみを参照した GMM クラスタ自動アライメント付き）
        """
        log.info("=" * 65)
        log.info(f"▶ サンプルごとの確率ラベル (ソフトターゲット) 計算開始 [Window: {self.window_size}]")
        log.info("=" * 65)

        if weights is not None:
            self.weights = np.asarray(weights, dtype=np.float64)

        raw_results = {}
        for s_name in ["train", "val", "test"]:
            res = self._load_and_compute_initial(s_name, weights=self.weights)
            if res is not None:
                raw_results[s_name] = res

        if not raw_results:
            log.warning("⚠️ 処理対象のデータが見つかりませんでした。")
            return {}

        # GMMクラスタ自動アライメント（Trainデータのみ参照で完全リークフリー）
        if "train" in raw_results and raw_results["train"]["n_clusters"] == 2:
            df_train_meta = raw_results["train"]["df_meta"]
            probs_train = raw_results["train"]["probs"]
            label_col = (
                "label"
                if "label" in df_train_meta.columns
                else ("label_mean" if "label_mean" in df_train_meta.columns else None)
            )
            if label_col is not None:
                y_train_task = df_train_meta[label_col].values.astype(float)
                clusters_train = (
                    df_train_meta["cluster"].values
                    if "cluster" in df_train_meta.columns
                    else np.argmax(probs_train, axis=1)
                )
                c1_mask = clusters_train == 1
                c0_mask = clusters_train == 0
                c1_ratio = float(np.mean(y_train_task[c1_mask])) if np.any(c1_mask) else 0.0
                c0_ratio = float(np.mean(y_train_task[c0_mask])) if np.any(c0_mask) else 0.0

                y_soft_train = raw_results["train"]["y_soft"]
                y_soft_val = raw_results["val"]["y_soft"] if "val" in raw_results else None
                y_soft_test = raw_results["test"]["y_soft"] if "test" in raw_results else None

                # クラスタ0の方が集中タスク割合が高い場合、クラスタ定義が逆転している
                inverted = bool(c0_ratio > c1_ratio)
                if c0_ratio > c1_ratio:
                    # クラスタ1が「集中」になるよう、全スプリットのソフトラベルを反転
                    y_soft_train = 1.0 - y_soft_train
                    if y_soft_val is not None:
                        y_soft_val = 1.0 - y_soft_val
                    if y_soft_test is not None:
                        y_soft_test = 1.0 - y_soft_test

                raw_results["train"]["y_soft"] = y_soft_train
                if y_soft_val is not None:
                    raw_results["val"]["y_soft"] = y_soft_val
                if y_soft_test is not None:
                    raw_results["test"]["y_soft"] = y_soft_test

                log.info(
                    f"GMM Alignment: c1_ratio={c1_ratio:.3f}, c0_ratio={c0_ratio:.3f} -> (Inverted: {inverted})"
                )

        results = {}
        for s_name, res in raw_results.items():
            results[s_name] = self._finalize_and_save_split(res)

        # 可視化プロット保存
        if self.save_plot:
            plot_path = self.data_dir / self.plot_filename
            plot_soft_label_distribution(results, plot_path, window_size=self.window_size)
            pls_dir = _LEARNING_DIR / "data" / "pls"
            if pls_dir.exists():
                plot_soft_label_distribution(results, pls_dir / f"soft_label_distribution_{self.window_size}.png", window_size=self.window_size)

        # 全結合版 PLS ソフトラベル付き CSV (明示的特徴量統合済み) を保存
        pls_dir = _LEARNING_DIR / "data" / "pls"
        if pls_dir.exists() and results:
            all_dfs = [res["df_meta"] for res in results.values() if "df_meta" in res and res["df_meta"] is not None]
            if all_dfs:
                df_all = pd.concat(all_dfs, ignore_index=True)
                pls_all_csv = pls_dir / f"pls_features_{self.window_size}_soft_labeled.csv"
                df_all.to_csv(pls_all_csv, index=False)
                log.info(f"  💾 全結合版 PLS ソフトラベル付きCSV保存 (明示的特徴量統合済み): {pls_all_csv.name}")

        log.info("🎉 全サンプルの確率ラベル（ソフトターゲット y）の計算・保存が正常に完了しました！")
        return results
