"""
クラスタリング用データローダーモジュール (Clustering Data Loader)

【責務】
- previous_task で抽出・被験者別 (Subject-wise) Zスコア標準化済みの PLS 潜在特徴量
  (pls_train_fused_standardized.npy 等) の読み込み
- フォールバックとして OOF 形式 PLS 特徴量 (X_pls_oof_*.npy) および
  TS2Vec 埋め込み (emb_train_visual.npy + emb_train_vital.npy / emb_train.npy) の探索・読み込み
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import sys
_LEARNING_DIR = Path(__file__).resolve().parent.parent
if str(_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(_LEARNING_DIR))

try:
    from config_loader import log
except ImportError:
    import logging
    log = logging.getLogger("clustering.data_loader")


class ClusteringDataLoader:
    """クラスタリングおよびクラスタ数決定向けの特徴量・メタデータローダークラス"""

    def __init__(self, cfg: dict, window_size: str | None = None):
        self.cfg = cfg
        self.paths = cfg.get("paths", {})
        self.dataset_cfg = cfg.get("dataset", {})
        self.window_size = window_size or self.dataset_cfg.get("window_size", "10s")

        self.learning_root = _LEARNING_DIR
        data_root = self.paths.get("data_root", self.learning_root / "data")
        if not Path(data_root).is_absolute():
            data_root = (self.learning_root / data_root).resolve()
        self.data_dir = Path(data_root) / self.window_size
        self.data_dir.mkdir(parents=True, exist_ok=True)

        input_data_root = self.paths.get("input_data_root", self.learning_root.parent / "preprocess" / "data" / "windowed")
        if not Path(input_data_root).is_absolute():
            input_data_root = (self.learning_root / input_data_root).resolve()
        self.input_data_dir = Path(input_data_root) / self.window_size

    def load_split_data(self) -> dict | None:
        """
        Train, Val, Test の各特徴量ベクトルおよびメタデータCSVをロードします。
        (※GMM 直前の被験者別 Zスコア標準化は previous_task にて完了済みのファイルを最優先で読み込みます)

        優先順位:
          1. previous_task で標準化済みの Split 別 PLS 特徴量 (pls_*_fused_standardized.npy / pls_*_fused.npy)
          2. OOF 形式 PLS 特徴量 (X_pls_oof_*.npy & pls_features_*.csv) + GroupShuffleSplit
          3. TS2Vec 埋め込み (emb_train_visual.npy + emb_train_vital.npy の自動結合、または emb_train.npy)
        """
        # 1. Split 別 PLS 特徴量 (previous_task で標準化済みの pls_*_fused_standardized.npy を最優先)
        pls_train_fused_std = self.data_dir / "pls_train_fused_standardized.npy"
        pls_train_fused_raw = self.data_dir / "pls_train_fused.npy"
        pls_train_fused = pls_train_fused_std if pls_train_fused_std.exists() else pls_train_fused_raw
        meta_train_path = (
            (self.data_dir / "meta_train.csv")
            if (self.data_dir / "meta_train.csv").exists()
            else (self.input_data_dir / "meta_train.csv")
        )
        if pls_train_fused.exists() and meta_train_path.exists():
            log.info(f"✓ previous_task 生成の Split 別 PLS 特徴量を検出・読み込みます: {pls_train_fused.name}")
            data = {
                "train": {
                    "emb": np.load(pls_train_fused),
                    "meta": pd.read_csv(meta_train_path),
                }
            }
            for split_k in ["val", "test"]:
                p_std = self.data_dir / f"pls_{split_k}_fused_standardized.npy"
                p_raw = self.data_dir / f"pls_{split_k}_fused.npy"
                p_target = p_std if p_std.exists() else p_raw
                m_target = (
                    (self.data_dir / f"meta_{split_k}.csv")
                    if (self.data_dir / f"meta_{split_k}.csv").exists()
                    else (self.input_data_dir / f"meta_{split_k}.csv")
                )
                if p_target.exists() and m_target.exists():
                    data[split_k] = {
                        "emb": np.load(p_target),
                        "meta": pd.read_csv(m_target),
                    }
            return data

        # 2. PLS 特徴量の探索 (OOF 形式テーブル: pls_features_*.csv & X_pls_oof_*.npy)
        pls_npy_cands = [
            self.learning_root / "data" / "pls" / f"X_pls_oof_{self.window_size}.npy",
            self.learning_root / "data" / "pls" / self.window_size / "X_pls_oof.npy",
            self.learning_root / "data" / "pls" / "X_pls_oof.npy",
            self.data_dir / f"X_pls_oof_{self.window_size}.npy",
            self.data_dir / "X_pls_oof.npy",
        ]
        pls_csv_cands = [
            self.learning_root / "data" / "pls" / f"pls_features_{self.window_size}.csv",
            self.learning_root / "data" / "pls" / self.window_size / "pls_features.csv",
            self.learning_root / "data" / "pls" / "pls_features.csv",
            self.data_dir / f"pls_features_{self.window_size}.csv",
            self.data_dir / "pls_features.csv",
        ]

        found_npy = next((p for p in pls_npy_cands if p.exists()), None)
        found_csv = next((p for p in pls_csv_cands if p.exists()), None)

        if found_npy and found_csv:
            log.info(f"✓ PLS 次元削減済み特徴量を検出・読み込みます:\n  - NPY: {found_npy}\n  - CSV: {found_csv}")
            arr = np.load(found_npy)
            df = pd.read_csv(found_csv)

            group_col = self.dataset_cfg.get("group_col", "username")
            test_size = float(self.dataset_cfg.get("test_size", 0.15))
            val_size = float(self.dataset_cfg.get("val_size", 0.1764))
            random_state = int(self.dataset_cfg.get("random_state", 42))

            if "split" in df.columns:
                splits = {}
                for s in ["train", "val", "test"]:
                    mask = (df["split"] == s).values
                    if mask.any():
                        splits[s] = {"emb": arr[mask], "meta": df[mask].reset_index(drop=True)}
                if "train" in splits:
                    return splits

            if group_col in df.columns and len(df[group_col].unique()) > 1 and test_size > 0:
                groups = df[group_col].values
                gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
                train_val_idx, test_idx = next(gss_test.split(arr, groups=groups))

                if val_size > 0:
                    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
                    tv_groups = groups[train_val_idx]
                    tr_sub_idx, val_sub_idx = next(gss_val.split(arr[train_val_idx], groups=tv_groups))
                    train_idx = train_val_idx[tr_sub_idx]
                    val_idx = train_val_idx[val_sub_idx]
                else:
                    train_idx = train_val_idx
                    val_idx = np.array([], dtype=int)

                splits = {
                    "train": {"emb": arr[train_idx], "meta": df.iloc[train_idx].reset_index(drop=True)},
                    "test": {"emb": arr[test_idx], "meta": df.iloc[test_idx].reset_index(drop=True)},
                }
                if len(val_idx) > 0:
                    splits["val"] = {"emb": arr[val_idx], "meta": df.iloc[val_idx].reset_index(drop=True)}

                n_val = len(splits["val"]["emb"]) if "val" in splits else 0
                log.info(
                    f"🔒 [Leakage-Free] 被験者グループ ({group_col}) で分割完了: "
                    f"Train={len(splits['train']['emb'])}, Val={n_val}, Test={len(splits['test']['emb'])}"
                )
                return splits

            return {"train": {"emb": arr, "meta": df}}

        # 3. TS2Vec 埋め込み・メタデータ
        emb_train_path = self.data_dir / "emb_train.npy"
        meta_train_path = (
            (self.data_dir / "meta_train.csv")
            if (self.data_dir / "meta_train.csv").exists()
            else (self.input_data_dir / "meta_train.csv")
        )

        vis_tr = (
            (self.data_dir / "emb_train_visual.npy")
            if (self.data_dir / "emb_train_visual.npy").exists()
            else (self.input_data_dir / "emb_train_visual.npy")
        )
        bio_tr = (
            (self.data_dir / "emb_train_vital.npy")
            if (self.data_dir / "emb_train_vital.npy").exists()
            else (self.input_data_dir / "emb_train_vital.npy")
        )
        if not emb_train_path.exists() and vis_tr.exists() and bio_tr.exists():
            log.info("✓ Visual/Vital 埋め込みを検出しました。自動結合して使用します")
            emb_train_arr = np.hstack([np.load(vis_tr), np.load(bio_tr)])
        elif emb_train_path.exists():
            emb_train_arr = np.load(emb_train_path)
        else:
            emb_train_arr = None

        if emb_train_arr is None or not meta_train_path.exists():
            log.warning(f"⚠️ 必要なデータが見つかりません:\n  Emb: {emb_train_path}\n  Meta: {meta_train_path}")
            return None

        data = {
            "train": {
                "emb": emb_train_arr,
                "meta": pd.read_csv(meta_train_path),
            }
        }

        # Val
        emb_val_path = self.data_dir / "emb_val.npy"
        meta_val_path = self.data_dir / "meta_val.csv"
        vis_val = self.data_dir / "emb_val_visual.npy"
        bio_val = self.data_dir / "emb_val_vital.npy"
        if emb_val_path.exists() and meta_val_path.exists():
            data["val"] = {"emb": np.load(emb_val_path), "meta": pd.read_csv(meta_val_path)}
        elif vis_val.exists() and bio_val.exists() and meta_val_path.exists():
            data["val"] = {"emb": np.hstack([np.load(vis_val), np.load(bio_val)]), "meta": pd.read_csv(meta_val_path)}

        # Test
        emb_test_path = self.data_dir / "emb_test.npy"
        meta_test_path = self.data_dir / "meta_test.csv"
        vis_test = self.data_dir / "emb_test_visual.npy"
        bio_test = self.data_dir / "emb_test_vital.npy"
        if emb_test_path.exists() and meta_test_path.exists():
            data["test"] = {"emb": np.load(emb_test_path), "meta": pd.read_csv(meta_test_path)}
        elif vis_test.exists() and bio_test.exists() and meta_test_path.exists():
            data["test"] = {"emb": np.hstack([np.load(vis_test), np.load(bio_test)]), "meta": pd.read_csv(meta_test_path)}

        return data

    def load_train_embeddings(self) -> np.ndarray | None:
        """クラスタ数探索用に Train 特徴量ベクトルのみをリークフリーで抽出して返します"""
        split_data = self.load_split_data()
        if split_data is None or "train" not in split_data:
            return None
        arr_train = split_data["train"]["emb"]
        log.info(f"🔒 [Leakage-Free] Train 特徴量ベクトル抽出完了: 形状 {arr_train.shape}")
        return arr_train


def load_clustering_data(cfg: dict, window_size: str | None = None) -> dict | None:
    loader = ClusteringDataLoader(cfg, window_size=window_size)
    return loader.load_split_data()


def load_train_embeddings(cfg: dict, window_size: str | None = None) -> np.ndarray | None:
    loader = ClusteringDataLoader(cfg, window_size=window_size)
    return loader.load_train_embeddings()
