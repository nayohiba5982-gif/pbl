"""
PLS 学習 & 潜在特徴量抽出・保存モジュール
(learning/previous_task/pls_extractor.py)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression


def run_pls_feature_extraction(
    d_dir: Path,
    pls_model_dir: Path,
    vis_train_path: Path | None,
    bio_train_path: Path | None,
    meta_train_path: Path | None,
    resolve_split_files_fn: Callable[[str, str], Tuple[Path | None, Path | None, Path | None]],
    n_components_visual: int,
    n_components_vital: int,
    ws: str,
    log,
) -> bool:
    """
    指定されたウィンドウサイズと成分数 (n_components_visual, n_components_vital) で
    Train データのみを用いて PLS を fit し、Train / Val / Test の潜在特徴量を抽出・保存します。
    """
    if not (vis_train_path and bio_train_path and meta_train_path):
        log.warning(f"⚠️ [{ws}] Train 埋め込みファイルが見つかりません。PLS 学習をスキップします。")
        return False

    X_train_vis = np.load(vis_train_path)
    X_train_bio = np.load(bio_train_path)
    meta_train = pd.read_csv(meta_train_path)
    label_col = "label" if "label" in meta_train.columns else "label_mean"
    y_train = meta_train[label_col].values.astype(float)

    n_vis = n_components_visual
    n_bio = n_components_vital

    log.info("=" * 70)
    log.info(f"▶ PLS 学習開始 (Train データのみ使用) [Window: {ws}]")
    log.info(f"  - Visual 設定成分数: {n_vis} 次元 (入力: {X_train_vis.shape})")
    log.info(f"  - Vital  設定成分数: {n_bio} 次元 (入力: {X_train_bio.shape})")
    log.info(f"  - 目的変数 y: {label_col} (サンプル数: {len(y_train)})")
    log.info("=" * 70)

    # Train のみで PLS モデルを fit
    pls_vis = PLSRegression(n_components=n_vis, scale=True).fit(X_train_vis, y_train)
    pls_bio = PLSRegression(n_components=n_bio, scale=True).fit(X_train_bio, y_train)

    # Train の潜在スコア抽出
    scores_train_vis = np.asarray(pls_vis.transform(X_train_vis), dtype=np.float32)
    scores_train_bio = np.asarray(pls_bio.transform(X_train_bio), dtype=np.float32)
    scores_train_fused = np.hstack([scores_train_vis, scores_train_bio]).astype(np.float32)

    r2_train_vis = float(pls_vis.score(X_train_vis, y_train))
    r2_train_bio = float(pls_bio.score(X_train_bio, y_train))

    log.info("【Train データへのフィッティング結果】")
    log.info(f"  ✓ Visual PLS (成分数 {n_vis}): R² = {r2_train_vis:.4f}")
    log.info(f"  ✓ Vital  PLS (成分数 {n_bio}): R² = {r2_train_bio:.4f}")
    log.info(
        f"  ✓ 抽出された Train 特徴量 shape: Visual={scores_train_vis.shape}, "
        f"Vital={scores_train_bio.shape}, Fused={scores_train_fused.shape} (計 {n_vis + n_bio} 次元)"
    )

    # 学習済み PLS モデルの保存
    vis_model_file = pls_model_dir / f"pls_model_vis_{ws}_c{n_vis}.joblib"
    bio_model_file = pls_model_dir / f"pls_model_bio_{ws}_c{n_bio}.joblib"
    joblib.dump(pls_vis, vis_model_file)
    joblib.dump(pls_bio, bio_model_file)
    log.info(f"✓ 学習済み PLS モデルを保存しました: {vis_model_file.name}, {bio_model_file.name}")

    # Train 特徴量の保存
    np.save(d_dir / "pls_train_visual.npy", scores_train_vis)
    np.save(d_dir / "pls_train_vital.npy", scores_train_bio)
    np.save(d_dir / "pls_train_fused.npy", scores_train_fused)

    # Val / Test データが存在すれば Train の PLS モデルを適用して変換・保存 (fit せず transform のみ)
    for split in ["val", "test"]:
        v_path, b_path, _ = resolve_split_files_fn(ws, split)
        if v_path is not None and b_path is not None:
            x_v = np.load(v_path)
            x_b = np.load(b_path)
            sc_v = np.asarray(pls_vis.transform(x_v), dtype=np.float32)
            sc_b = np.asarray(pls_bio.transform(x_b), dtype=np.float32)
            sc_fused = np.hstack([sc_v, sc_b]).astype(np.float32)

            np.save(d_dir / f"pls_{split}_visual.npy", sc_v)
            np.save(d_dir / f"pls_{split}_vital.npy", sc_b)
            np.save(d_dir / f"pls_{split}_fused.npy", sc_fused)
            log.info(f"✓ {split.upper()} データも Train 基準で PLS 変換完了 (shape: {sc_fused.shape})")

    log.info(f"🎉 [{ws}] の PLS 特徴量抽出および保存が完了しました。")
    return True
