"""
クラスタ所属確率・データ読み込み & GMMモデル推論ローダーモジュール (Data Loader & Inference)
"""

from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd

import sys
_LEARNING_DIR = Path(__file__).resolve().parent.parent
if str(_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(_LEARNING_DIR))

try:
    from config_loader import log
except ImportError:
    import logging
    log = logging.getLogger("probability_label.loader")


def find_latest_gmm_model(model_dir: Path, window_size: str = "10s") -> Path | None:
    """models ディレクトリから最新の GMM モデルファイルを探索します"""
    if not model_dir.exists():
        return None
    candidates = list(model_dir.glob(f"gmm_model_{window_size}_*.joblib"))
    if not candidates:
        candidates = list(model_dir.glob("gmm_model_*.joblib"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_probabilities_for_split(
    data_dir: Path,
    split_name: str,
    model_dir: Path | None = None,
    window_size: str = "10s",
) -> tuple[pd.DataFrame | None, np.ndarray | None, list[str]]:
    """
    指定された Split (train, val, test) のメタデータおよび所属確率行列を取得します。

    1. meta_{split}_clustered.csv から cluster_prob_* 列を優先的に取得
    2. なければ GMM モデルと emb_{split}.npy から predict_proba を推論して補完

    Returns:
        meta_df (pd.DataFrame | None): メタデータ DataFrame
        probs (np.ndarray | None): 所属確率行列 [N, K]
        prob_cols (list[str]): 所属確率カラム名リスト
    """
    meta_clustered_path = data_dir / f"meta_{split_name}_clustered.csv"
    if not meta_clustered_path.exists():
        pls_cand = _LEARNING_DIR / "data" / "pls" / f"pls_features_{window_size}_{split_name}_clustered.csv"
        if pls_cand.exists():
            meta_clustered_path = pls_cand
        else:
            pls_cand2 = _LEARNING_DIR / "data" / "pls" / f"pls_features_{window_size}_clustered.csv"
            if pls_cand2.exists():
                meta_clustered_path = pls_cand2
            else:
                pls_cand3 = _LEARNING_DIR / "data" / "pls" / f"meta_{split_name}_clustered.csv"
                if pls_cand3.exists():
                    meta_clustered_path = pls_cand3

    emb_path = data_dir / f"emb_{split_name}.npy"

    # 1. meta_*_clustered.csv から cluster_prob_* 列を優先取得
    if meta_clustered_path.exists():
        df_meta = pd.read_csv(meta_clustered_path)
        prob_cols = sorted(
            [c for c in df_meta.columns if re.match(r"^cluster_prob_\d+$", c)],
            key=lambda c: int(c.split("_")[-1]),
        )
        if prob_cols:
            probs = df_meta[prob_cols].values
            log.info(
                f"[{split_name}] meta_{split_name}_clustered.csv より {len(prob_cols)} クラスの所属確率をロード: "
                f"shape={probs.shape}"
            )
            return df_meta, probs, prob_cols

    # 2. cluster_prob_* 列がない、または CSV がない場合は GMM モデルと埋め込みから推論
    if emb_path.exists() and model_dir is not None:
        emb_data = np.load(emb_path)
        gmm_model_path = find_latest_gmm_model(model_dir, window_size=window_size)
        if gmm_model_path and gmm_model_path.exists():
            log.info(f"[{split_name}] GMM モデル ({gmm_model_path.name}) から predict_proba を算出します...")
            gmm = joblib.load(gmm_model_path)
            probs = gmm.predict_proba(emb_data)
            n_k = probs.shape[1]
            prob_cols = [f"cluster_prob_{k}" for k in range(n_k)]

            meta_raw_path = data_dir / f"meta_{split_name}.csv"
            if meta_clustered_path.exists():
                df_meta = pd.read_csv(meta_clustered_path)
            elif meta_raw_path.exists():
                df_meta = pd.read_csv(meta_raw_path)
            else:
                df_meta = pd.DataFrame({"index": np.arange(len(probs))})

            for k_idx, c_name in enumerate(prob_cols):
                df_meta[c_name] = probs[:, k_idx]

            log.info(f"[{split_name}] GMM 推論完了: {n_k} クラス, shape={probs.shape}")
            return df_meta, probs, prob_cols

    log.warning(f"⚠️ [{split_name}] 所属確率データを取得できませんでした: {data_dir}")
    return None, None, []
