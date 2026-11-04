"""
下流タスク LOSO-CV 共通データローダー & モダリティ特徴量分離モジュール
(learning/downstream_task/data_loader.py)

PLS 特徴量 (6d: vis_pls_1..3, bio_pls_1..3) または TS2Vec 埋め込み (32d + 32d)、
メタデータ、明示的物理統計量 (20d) の読み込み・被験者別Zスコア標準化・
モダリティ別 (Visual / Vital) 特徴量分離を行います。
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd


VITAL_KEYWORDS = (
    "bpm",
    "bvp",
    "hr",
    "ibi",
    "rmssd",
    "sdnn",
    "pnn",
    "lf",
    "hf",
    "vital",
    "pulse",
)


def standardize_subject_wise(
    features: np.ndarray,
    usernames: Sequence,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    被験者 (username) ごとに平均0・分散1 (標準偏差1) となるよう Zスコア標準化を適用します。
    """
    normed = np.zeros_like(features, dtype=np.float32)
    user_arr = np.asarray(usernames).astype(str)
    for u in np.unique(user_arr):
        u_mask = user_arr == u
        u_mean = np.mean(features[u_mask], axis=0, keepdims=True)
        u_std = np.std(features[u_mask], axis=0, keepdims=True)
        u_std = np.where(u_std < eps, 1.0, u_std)
        normed[u_mask] = (features[u_mask] - u_mean) / u_std
    return normed


def split_modality_feature_indices(
    n_vis_pls: int = 3,
    n_bio_pls: int = 3,
    explicit_names: Optional[Sequence[str]] = None,
    n_explicit: int = 0,
) -> Tuple[List[int], List[int]]:
    """
    下流タスク入力特徴量 X = [vis_pls (n_vis_pls), bio_pls (n_bio_pls), explicit (n_explicit)] を
    Visual 系特徴量インデックスと Vital 系特徴量インデックスに厳密に分離します。

    - Visual: vis_pls + 視覚系明示的特徴量 (EAR, 頭部姿勢, 視線など)
    - Vital : bio_pls + 生体系明示的特徴量 (BPM, BVPなど)
    """
    vis_indices: List[int] = list(range(0, n_vis_pls))
    vital_indices: List[int] = list(range(n_vis_pls, n_vis_pls + n_bio_pls))
    offset = n_vis_pls + n_bio_pls

    if n_explicit > 0:
        if explicit_names is not None and len(explicit_names) == n_explicit:
            for i, col_name in enumerate(explicit_names):
                col_lower = str(col_name).lower()
                if any(kw in col_lower for kw in VITAL_KEYWORDS):
                    vital_indices.append(offset + i)
                else:
                    vis_indices.append(offset + i)
        else:
            vital_local_positions = {6, 7, 14, 15}
            for i in range(n_explicit):
                if i in vital_local_positions:
                    vital_indices.append(offset + i)
                else:
                    vis_indices.append(offset + i)

    return vis_indices, vital_indices


def build_modality_feature_names(
    n_vis_pls: int = 3,
    n_bio_pls: int = 3,
    explicit_names: Optional[Sequence[str]] = None,
    n_explicit: int = 0,
) -> Tuple[List[str], List[str], List[str]]:
    """
    全体の特徴量名リスト、Visual用特徴量名リスト、Vital用特徴量名リストを返します。
    """
    all_names = [f"vis_pls_{i+1}" for i in range(n_vis_pls)] + [
        f"bio_pls_{i+1}" for i in range(n_bio_pls)
    ]
    if n_explicit > 0:
        if explicit_names is not None and len(explicit_names) == n_explicit:
            all_names.extend([str(c) for c in explicit_names])
        else:
            all_names.extend([f"explicit_{i+1}" for i in range(n_explicit)])

    vis_idx, vital_idx = split_modality_feature_indices(
        n_vis_pls=n_vis_pls,
        n_bio_pls=n_bio_pls,
        explicit_names=explicit_names,
        n_explicit=n_explicit,
    )
    vis_names = [all_names[i] for i in vis_idx]
    vital_names = [all_names[i] for i in vital_idx]
    return all_names, vis_names, vital_names


def load_embeddings_and_meta(
    data_dir: Path,
    input_data_dir: Path,
    learning_root: Path,
    window_size: str,
    log,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """全被験者の PLS 潜在特徴量 (3d + 3d) または TS2Vec 32次元埋め込み (視覚 & 生体) およびメタデータを読み込み"""
    # パターン 1: PLS 特徴量 CSV
    pls_dir = learning_root / "data" / "pls"
    cand_pls_list = [
        pls_dir / f"pls_features_{window_size}_soft_labeled.csv",
        pls_dir / f"pls_features_{window_size}_clustered.csv",
        pls_dir / f"pls_features_{window_size}.csv",
    ]
    cand_pls = next((p for p in cand_pls_list if p.exists()), None)
    if cand_pls is not None:
        p_df = pd.read_csv(cand_pls)
        vis_cols = [c for c in p_df.columns if c.startswith("vis_pls_")]
        bio_cols = [c for c in p_df.columns if c.startswith("bio_pls_")]
        if vis_cols and bio_cols:
            log.info(f"✓ PLS 特徴量 CSV を検出・読み込みます: {cand_pls.name}")
            log.info(
                f"  ✓ 読み込み完了: サンプル数={len(p_df)}, 視覚PLS={len(vis_cols)}d, 生体PLS={len(bio_cols)}d, "
                f"全被験者数={len(p_df['username'].unique()) if 'username' in p_df.columns else 'N/A'}"
            )
            return p_df[vis_cols].values.astype(np.float32), p_df[bio_cols].values.astype(np.float32), p_df

    # パターン 2: Split ごとの TS2Vec ファイルまたは PLS ファイルを探索 (最新の input_data_dir を優先)
    search_dirs = [input_data_dir, data_dir]
    src_dir = next(
        (d for d in search_dirs if (d / "meta_train.csv").exists() and (d / "emb_train_visual.npy").exists()),
        None
    )
    if src_dir is not None:
        tr_m = src_dir / "meta_train.csv"
        va_m = src_dir / "meta_val.csv"
        te_m = src_dir / "meta_test.csv"
        tr_v = src_dir / "emb_train_visual.npy"
        tr_b = src_dir / "emb_train_vital.npy"

        log.info("✓ Split 別 (Train/Val/Test) の TS2Vec 埋め込み・メタデータを検出しました。全被験者データを結合します")
        meta_dfs = [pd.read_csv(tr_m)]
        vis_arrs = [np.load(tr_v)]
        bio_arrs = [np.load(tr_b)]

        if va_m.exists() and (src_dir / "emb_val_visual.npy").exists():
            meta_dfs.append(pd.read_csv(va_m))
            vis_arrs.append(np.load(src_dir / "emb_val_visual.npy"))
            bio_arrs.append(np.load(src_dir / "emb_val_vital.npy"))

        if te_m.exists() and (src_dir / "emb_test_visual.npy").exists():
            meta_dfs.append(pd.read_csv(te_m))
            vis_arrs.append(np.load(src_dir / "emb_test_visual.npy"))
            bio_arrs.append(np.load(src_dir / "emb_test_vital.npy"))

        meta_df = pd.concat(meta_dfs, ignore_index=True)
        emb_vis = np.concatenate(vis_arrs, axis=0).astype(np.float32)
        emb_bio = np.concatenate(bio_arrs, axis=0).astype(np.float32)

        log.info(
            f"  ✓ 結合完了: サンプル数={len(meta_df)}, 視覚={emb_vis.shape}, 生体={emb_bio.shape}, "
            f"全被験者数={len(meta_df['username'].unique()) if 'username' in meta_df.columns else 'N/A'}"
        )
        return emb_vis, emb_bio, meta_df

    # パターン 2b: Split 別 PLS ファイル (pls_train_visual.npy, pls_train_vital.npy, meta_train_soft_labeled.csv / meta_train.csv)
    pls_src_dir = next(
        (d for d in search_dirs if (d / "pls_train_visual.npy").exists() and (d / "pls_train_vital.npy").exists()),
        None
    )
    if pls_src_dir is not None:
        log.info("✓ Split 別 (Train/Val/Test) の PLS 潜在特徴量・メタデータを検出しました。全被験者データを結合します")
        meta_dfs = []
        vis_arrs = []
        bio_arrs = []
        for split_k in ["train", "val", "test"]:
            v_p = pls_src_dir / f"pls_{split_k}_visual.npy"
            b_p = pls_src_dir / f"pls_{split_k}_vital.npy"
            m_cands = [
                pls_src_dir / f"meta_{split_k}_soft_labeled.csv",
                pls_src_dir / f"meta_{split_k}_clustered.csv",
                pls_src_dir / f"meta_{split_k}.csv",
            ]
            m_p = next((p for p in m_cands if p.exists()), None)
            if v_p.exists() and b_p.exists() and m_p is not None:
                meta_dfs.append(pd.read_csv(m_p))
                vis_arrs.append(np.load(v_p))
                bio_arrs.append(np.load(b_p))

        if meta_dfs:
            meta_df = pd.concat(meta_dfs, ignore_index=True)
            emb_vis = np.concatenate(vis_arrs, axis=0).astype(np.float32)
            emb_bio = np.concatenate(bio_arrs, axis=0).astype(np.float32)
            log.info(
                f"  ✓ 結合完了: サンプル数={len(meta_df)}, 視覚={emb_vis.shape}, 生体={emb_bio.shape}, "
                f"全被験者数={len(meta_df['username'].unique()) if 'username' in meta_df.columns else 'N/A'}"
            )
            return emb_vis, emb_bio, meta_df

    # パターン 3: 一括 metadata.csv および一括埋め込みファイルの探索
    m_cands = [
        input_data_dir / "metadata.csv",
        input_data_dir.parent / "combined" / window_size / "metadata.csv",
        data_dir / "metadata.csv",
        data_dir.parent / "combined" / window_size / "metadata.csv",
    ]
    meta_path = next((p for p in m_cands if p.exists()), m_cands[0])

    if meta_path.exists():
        meta_df = pd.read_csv(meta_path)
        vis_cands = [
            data_dir / "emb_visual_all.npy",
            data_dir / "emb_all_visual.npy",
            data_dir / "visual" / "emb_all.npy",
            data_dir / "emb_visual.npy",
            data_dir / "visual" / "X_windows.npy",
        ]
        bio_cands = [
            data_dir / "emb_vital_all.npy",
            data_dir / "emb_all_vital.npy",
            data_dir / "vital" / "emb_all.npy",
            data_dir / "emb_vital.npy",
            data_dir / "vital" / "X_windows.npy",
        ]
        found_vis = next((p for p in vis_cands if p.exists()), None)
        found_bio = next((p for p in bio_cands if p.exists()), None)
        if found_vis and found_bio:
            emb_vis = np.load(found_vis)
            emb_bio = np.load(found_bio)
            if emb_vis.ndim == 3:
                emb_vis = emb_vis.mean(axis=1)
            if emb_bio.ndim == 3:
                emb_bio = emb_bio.mean(axis=1)
            return emb_vis.astype(np.float32), emb_bio.astype(np.float32), meta_df

    raise FileNotFoundError(
        f"PLS 特徴量または TS2Vec 潜在埋め込みベクトルが "
        f"{data_dir} に見つかりません。先に特徴量抽出を実行してください。"
    )


def load_explicit_features(
    meta_df: pd.DataFrame,
    input_data_dir: Path,
    data_dir: Path,
    window_size: str,
    log,
) -> Tuple[Optional[np.ndarray], List[str]]:
    """明示的物理特徴量 (20次元: EAR, 頭部姿勢, 視線, BPM統計量) をロードまたは算出"""
    # 1. すでに meta_df に明示的特徴量が含まれている場合
    exclude_cols = {
        "username", "subject_id", "label", "label_mean", "target_label",
        "cluster", "y_soft", "prob_label", "split", "index", "time",
        "window_sec", "start_idx", "end_idx", "start_time", "end_time",
        "bpm_base_10s", "bpm_mean",
    }
    stat_cols = [
        c for c in meta_df.columns
        if c not in exclude_cols
        and not c.startswith("vis_pls_")
        and not c.startswith("bio_pls_")
        and not c.startswith("cluster_prob_")
    ]
    if len(stat_cols) >= 20:
        log.info(f"✓ メタデータから直接 {len(stat_cols)} 次元の明示的物理特徴量を検出しました: {stat_cols[:4]} ...")
        return meta_df[stat_cols].values.astype(np.float32), stat_cols

    # 1b. Split 別 meta_*_soft_labeled.csv に統計量列が含まれている場合
    soft_meta_dfs = []
    for split_k in ["train", "val", "test"]:
        p = data_dir / f"meta_{split_k}_soft_labeled.csv"
        if p.exists():
            soft_meta_dfs.append(pd.read_csv(p))
    if soft_meta_dfs:
        combined_soft = pd.concat(soft_meta_dfs, ignore_index=True)
        if len(combined_soft) == len(meta_df):
            s_cols = [
                c for c in combined_soft.columns
                if c not in exclude_cols
                and not c.startswith("vis_pls_")
                and not c.startswith("bio_pls_")
                and not c.startswith("cluster_prob_")
            ]
            if len(s_cols) >= 20:
                log.info(f"✓ meta_*_soft_labeled.csv から {len(s_cols)} 次元の明示的物理特徴量を検出しました")
                return combined_soft[s_cols].values.astype(np.float32), s_cols

    # 2. X_windows.npy から動的算出 (被験者単位でのアライメント対応)
    try:
        from features import compute_explicit_statistics
        x_cands = [
            input_data_dir / "X_windows.npy",
            input_data_dir.parent / "combined" / window_size / "X_windows.npy",
            data_dir / "X_windows.npy",
            data_dir.parent / "combined" / window_size / "X_windows.npy",
        ]
        m_cands = [
            input_data_dir / "metadata.csv",
            input_data_dir.parent / "combined" / window_size / "metadata.csv",
            data_dir / "metadata.csv",
            data_dir.parent / "combined" / window_size / "metadata.csv",
        ]
        x_path = next((p for p in x_cands if p.exists()), None)
        meta_raw_path = next((p for p in m_cands if p.exists()), None)
        if x_path is not None:
            X_all = np.load(x_path)
            meta_raw = pd.read_csv(meta_raw_path) if meta_raw_path is not None else None
            stat_matrix, stat_names = compute_explicit_statistics(
                X_all, meta_df=meta_raw if meta_raw is not None and len(meta_raw) == len(X_all) else meta_df
            )
            if len(stat_matrix) == len(meta_df):
                log.info(f"✓ X_windows.npy から {len(stat_names)} 次元の明示的物理特徴量を算出しました")
                return stat_matrix.astype(np.float32), stat_names
            elif (
                meta_raw is not None
                and len(meta_raw) == len(stat_matrix)
                and "username" in meta_raw.columns
                and "username" in meta_df.columns
            ):
                stat_df = pd.DataFrame(stat_matrix, columns=stat_names)
                stat_df["_u_str"] = meta_raw["username"].astype(str).values
                matched_list = []
                for u in meta_df["username"].astype(str).unique():
                    u_count = int((meta_df["username"].astype(str) == u).sum())
                    u_stat = stat_df[stat_df["_u_str"] == u].iloc[:u_count]
                    matched_list.append(u_stat[stat_names])
                if matched_list:
                    matched_df = pd.concat(matched_list, ignore_index=True)
                    if len(matched_df) == len(meta_df):
                        log.info(
                            f"✓ X_windows.npy から被験者マッチングにより {len(stat_names)} 次元の明示的物理特徴量を算出しました"
                        )
                        return matched_df.values.astype(np.float32), stat_names
    except Exception as e:
        log.warning(f"明示的特徴量の取得をスキップします: {e}")

    return None, []
