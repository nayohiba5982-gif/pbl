r"""
確率ラベル計算コアロジックモジュール (Core Logic for Soft Label Calculation)

数式:
    y_i = \sum_{k=0}^{K-1} w_k \cdot P(\text{Cluster}=k \mid x_i)
"""

import numpy as np


def compute_soft_labels(
    probs: np.ndarray, weights: np.ndarray | list[float]
) -> np.ndarray:
    r"""
    各サンプルのクラスタ所属確率とクラスタ重みの内積を計算し、
    連続値ソフトラベル（期待値 y ∈ [0, 1]）を算出します。

    数式:
        y_i = \sum_{k=0}^{K-1} w_k \cdot P(\text{Cluster}=k \mid x_i)

    Args:
        probs (np.ndarray): クラスタ所属確率行列 [N, K] (各行の合計は 1.0)
        weights (np.ndarray | list[float]): 各クラスタに対する重みベクトル [K] (例: [1.0, 0.5, 0.0])

    Returns:
        np.ndarray: 各サンプルの期待値ベクトル [N] (0.0 〜 1.0)
    """
    weights = np.asarray(weights, dtype=np.float64)

    if probs.ndim != 2:
        raise ValueError(f"所属確率行列は2次元 [N, K] である必要があります。現在の形状: {probs.shape}")

    n_samples, n_clusters = probs.shape
    if n_clusters != len(weights):
        raise ValueError(
            f"クラスタ数 (K={n_clusters}) と指定された重みベクトルの長さ ({len(weights)}) が一致しません。\n"
            f"クラスタ数: {n_clusters}, 重み: {weights}"
        )

    # 確率の行方向正規化（数値誤差防止）
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    normalized_probs = probs / row_sums

    # 期待値の計算: y = probs ・ weights
    y_soft = np.dot(normalized_probs, weights)

    # [0.0, 1.0] にクリッピング
    y_soft = np.clip(y_soft, 0.0, 1.0)
    return y_soft


def generate_default_weights(
    n_clusters: int, user_weights: list[float] | np.ndarray | None = None
) -> np.ndarray:
    """
    指定された重みがあればバリデーションを行い、未指定の場合は 1.0 から 0.0 までの等間隔重みを生成します。
    例: K=3 ➔ [1.0, 0.5, 0.0], K=2 ➔ [1.0, 0.0]
    """
    if user_weights is not None:
        w = np.asarray(user_weights, dtype=np.float64)
        if len(w) != n_clusters:
            raise ValueError(
                f"指定された重みの要素数 ({len(w)}) がクラスタ数 (K={n_clusters}) と一致しません。"
            )
        return w

    return np.linspace(1.0, 0.0, n_clusters)
