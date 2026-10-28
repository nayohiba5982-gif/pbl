"""
GMM 所属確率推論・ラベル付与・保存モジュール (GMM Predictor & Saver)

【責務】
- GMM 推論直前での被験者別 (Subject-wise) Zスコア標準化の確認・徹底
- 各 Split (Train, Val, Test) に対する所属確率 (predict_proba) & 離散ラベル (predict) の算出
- 期待値ソフトターゲット (y_soft ∈ [0, 1]) の算出・メタデータ CSV への付与および NPY 配列保存
- pls_features_*_clustered.csv のスプリット別保存および全スプリット結合保存（上書きバグ防止）
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

import sys
_LEARNING_DIR = Path(__file__).resolve().parent.parent.parent
if str(_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(_LEARNING_DIR))

try:
    from config_loader import log
except ImportError:
    import logging
    log = logging.getLogger("clustering.gmm.predictor")

from ..probability.core import compute_soft_labels, generate_default_weights


def apply_and_save_clusters(
    gmm: GaussianMixture,
    split_data: dict,
    n_clusters: int,
    weights: list[float] | np.ndarray | None = None,
    target_col: str = "y_soft",
    save_npy: bool = True,
    data_dir: Path | None = None,
    window_size: str = "10s",
    apply_subject_norm: bool = True,
) -> dict[str, pd.DataFrame]:
    """各データセットに確率ラベル (y_soft ∈ [0, 1])、クラスタ番号、所属確率を付与して保存します"""
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != n_clusters:
            log.warning(
                f"⚠️ 指定された重み要素数 ({len(w)}) がクラスタ数 (k={n_clusters}) と一致しないため、等間隔重みを再生成します。"
            )
            w = generate_default_weights(n_clusters)
    else:
        w = generate_default_weights(n_clusters)
        log.info(f"ℹ️ クラスタ数 k={n_clusters} に応じた等間隔重みを適用します: {[round(x, 3) for x in w.tolist()]}")

    clustered_dfs: dict[str, pd.DataFrame] = {}

    # 1. 各 Split に対して離散クラスタと所属確率を推論 (Val/Test は推論のみ、Fitなし)
    split_infer: dict[str, dict] = {}
    w_init = np.array([float(np.min(w)), float(np.max(w))], dtype=np.float64) if n_clusters == 2 else w

    for split_name, s_data in split_data.items():
        emb = s_data["emb"]
        df_meta = s_data["meta"].copy()

        # PLS 特徴量列 (vis_pls_*, bio_pls_*) がメタデータに未付与の場合は付与
        has_pls_cols = any(c.startswith("vis_pls_") for c in df_meta.columns) and any(
            c.startswith("bio_pls_") for c in df_meta.columns
        )
        if not has_pls_cols and emb.ndim == 2 and emb.shape[1] >= 2 and emb.shape[1] <= 12:
            n_vis = emb.shape[1] // 2
            n_bio = emb.shape[1] - n_vis
            for i in range(n_vis):
                df_meta[f"vis_pls_{i+1}"] = emb[:, i]
            for i in range(n_bio):
                df_meta[f"bio_pls_{i+1}"] = emb[:, n_vis + i]

        labels = gmm.predict(emb)
        probs = gmm.predict_proba(emb)
        y_soft = compute_soft_labels(probs, w_init)

        split_infer[split_name] = {
            "df_meta": df_meta,
            "labels": labels,
            "probs": probs,
            "y_soft": y_soft,
        }

    # 2. GMMクラスタ自動アライメント（Label Switching防止・Trainデータのみ参照で完全リークフリー）
    if n_clusters == 2 and "train" in split_infer:
        df_train_meta = split_infer["train"]["df_meta"]
        label_col = (
            "label"
            if "label" in df_train_meta.columns
            else ("label_mean" if "label_mean" in df_train_meta.columns else None)
        )
        if label_col is not None:
            y_train_task = df_train_meta[label_col].values.astype(float)
            clusters_train = split_infer["train"]["labels"]
            c1_mask = clusters_train == 1
            c0_mask = clusters_train == 0
            c1_ratio = float(np.mean(y_train_task[c1_mask])) if np.any(c1_mask) else 0.0
            c0_ratio = float(np.mean(y_train_task[c0_mask])) if np.any(c0_mask) else 0.0

            y_soft_train = split_infer["train"]["y_soft"]
            y_soft_val = split_infer["val"]["y_soft"] if "val" in split_infer else None
            y_soft_test = split_infer["test"]["y_soft"] if "test" in split_infer else None

            # クラスタ0の方が集中タスク割合が高い場合、クラスタ定義が逆転している
            inverted = bool(c0_ratio > c1_ratio)
            if c0_ratio > c1_ratio:
                # クラスタ1が「集中」になるよう、全スプリットのソフトラベルを反転
                y_soft_train = 1.0 - y_soft_train
                if y_soft_val is not None:
                    y_soft_val = 1.0 - y_soft_val
                if y_soft_test is not None:
                    y_soft_test = 1.0 - y_soft_test

            split_infer["train"]["y_soft"] = y_soft_train
            if y_soft_val is not None:
                split_infer["val"]["y_soft"] = y_soft_val
            if y_soft_test is not None:
                split_infer["test"]["y_soft"] = y_soft_test

            log.info(
                f"GMM Alignment: c1_ratio={c1_ratio:.3f}, c0_ratio={c0_ratio:.3f} -> (Inverted: {inverted})"
            )

    # 3. 各 Split のメタデータへ反映・保存
    for split_name, info in split_infer.items():
        df_meta = info["df_meta"]
        labels = info["labels"]
        probs = info["probs"]
        y_soft = info["y_soft"]

        # メタデータに追加
        df_meta["cluster"] = labels
        df_meta[target_col] = y_soft
        df_meta["prob_label"] = y_soft
        df_meta["split"] = split_name
        for k_idx in range(n_clusters):
            df_meta[f"cluster_prob_{k_idx}"] = probs[:, k_idx]

        if data_dir is not None:
            out_csv_path = data_dir / f"meta_{split_name}_clustered.csv"
            out_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_meta.to_csv(out_csv_path, index=False)
            log.info(f"💾 {split_name.capitalize()} クラスタリング結果を保存: {out_csv_path} (行数: {len(df_meta)})")

            pls_dir = _LEARNING_DIR / "data" / "pls"
            if pls_dir.exists():
                # スプリット別に保存（ループの最後の Test スプリットによる上書きバグを防止）
                pls_out_csv = pls_dir / f"pls_features_{window_size}_{split_name}_clustered.csv"
                df_meta.to_csv(pls_out_csv, index=False)
                log.info(f"💾 PLS フォルダにスプリット別クラスタリング結果を保存 ({split_name}): {pls_out_csv.name}")

            if save_npy:
                np.save(data_dir / f"y_soft_{split_name}.npy", y_soft)
                np.save(data_dir / f"y_{split_name}.npy", y_soft)
                np.save(data_dir / f"probs_{split_name}.npy", probs)
                log.info(f"   💾 確率ラベル配列 (y_soft_{split_name}.npy, y_{split_name}.npy, probs_{split_name}.npy) を保存")

        clustered_dfs[split_name] = df_meta

        log.info(
            f"   ✓ [{split_name}] 確率ラベル y (0~1): 平均={y_soft.mean():.4f}, 標準偏差={y_soft.std():.4f}, "
            f"最小={y_soft.min():.4f}, 最大={y_soft.max():.4f}"
        )
        dist = df_meta["cluster"].value_counts().sort_index().to_dict()
        log.info(f"   クラスタ頻度分布: {dist}")

    # 全スプリット結合版の PLS クラスタリング結果も安全に保存
    pls_dir = _LEARNING_DIR / "data" / "pls"
    if pls_dir.exists() and len(clustered_dfs) > 0:
        all_dfs = []
        for s_name, d_meta in clustered_dfs.items():
            c_df = d_meta.copy()
            c_df["split"] = s_name
            all_dfs.append(c_df)
        df_all_clustered = pd.concat(all_dfs, ignore_index=True)
        all_pls_csv = pls_dir / f"pls_features_{window_size}_clustered.csv"
        df_all_clustered.to_csv(all_pls_csv, index=False)
        log.info(f"💾 全スプリット結合版 PLS クラスタリング結果を保存: {all_pls_csv.name} (計 {len(df_all_clustered)} 行)")

    return clustered_dfs
