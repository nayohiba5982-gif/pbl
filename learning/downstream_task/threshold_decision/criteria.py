"""
閾値決定（Threshold Tuning）の数理基準・評価関数モジュール
(learning/downstream_task/threshold_decision/criteria.py)

検証データ（Val）に対してデータリークなしで最適カットオフ値 th* ∈ [0, 1] を探索するための
3大基準を提供します：
  1. Youden's J index (`youden_j`):
     J(th) = Recall(th) + Specificity(th) - 1 = 2 * Balanced_Accuracy(th) - 1
  2. F1-Macro 最大化 (`f1_macro`):
     th* = argmax_th F1-Macro(th) = 0.5 * (F1_pos(th) + F1_neg(th))
  3. Precision-Recall のバランス (`pr_balance` / `pr_min_diff` / `pr_product`):
     適合率 Precision(th) と再現率 Recall(th) の差が最小 (|P - R| 最小) または積 (P * R) が最大となる点
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd


SUPPORTED_CRITERIA = (
    "youden_j",
    "f1_macro",
    "pr_balance",
    "pr_min_diff",
    "pr_product",
)

CRITERION_ALIASES = {
    "youden": "youden_j",
    "youden_j": "youden_j",
    "j_index": "youden_j",
    "balanced_acc": "youden_j",
    "balanced_accuracy": "youden_j",
    "f1": "f1_macro",
    "f1_macro": "f1_macro",
    "macro_f1": "f1_macro",
    "pr": "pr_balance",
    "pr_balance": "pr_balance",
    "pr_diff": "pr_min_diff",
    "pr_min_diff": "pr_min_diff",
    "pr_prod": "pr_product",
    "pr_product": "pr_product",
}

CRITERION_LABELS = {
    "youden_j": "Youden's J Index (Balanced Accuracy 最大化)",
    "f1_macro": "F1-Macro 最大化 (少数派クラス重視)",
    "pr_balance": "Precision-Recall バランス",
    "pr_min_diff": "Precision-Recall 差最小化 (|P - R| → min)",
    "pr_product": "Precision-Recall 積最大化 (P × R → max)",
}


def normalize_criterion_name(criterion: str) -> str:
    """基準名を正規化します。"""
    key = str(criterion or "youden_j").strip().lower().replace("-", "_").replace(" ", "_")
    if key not in CRITERION_ALIASES:
        raise ValueError(
            f"未知の閾値決定基準です: '{criterion}'. "
            f"利用可能: {list(SUPPORTED_CRITERIA)}"
        )
    return CRITERION_ALIASES[key]


def build_candidate_thresholds(
    y_pred: np.ndarray,
    search_range: Sequence[float] = (0.05, 0.95),
    search_step: float = 0.005,
    use_exact_curve_points: bool = True,
) -> np.ndarray:
    """
    探索対象の閾値候補配列 th ∈ [min_th, max_th] を構築します。
    等間隔グリッドに加え、オプションで予測値分布の分位点（正確なROC/PR変化点）を統合します。
    """
    low = float(max(0.001, min(0.999, search_range[0])))
    high = float(max(low + 1e-4, min(0.999, search_range[1])))
    step = float(max(1e-4, search_step))

    grid = np.arange(low, high + step * 0.5, step, dtype=np.float64)
    grid = np.clip(grid, low, high)

    if use_exact_curve_points and len(y_pred) > 0:
        valid_preds = np.asarray(y_pred, dtype=np.float64)
        valid_preds = valid_preds[np.isfinite(valid_preds)]
        if len(valid_preds) > 0:
            # 予測値の範囲が [low, high] 内で変化するユニーク点・分位点を追加
            in_range = valid_preds[(valid_preds >= low) & (valid_preds <= high)]
            if len(in_range) > 200:
                quantiles = np.linspace(0.01, 0.99, 199)
                curve_pts = np.quantile(in_range, quantiles)
            else:
                curve_pts = np.unique(in_range)
            grid = np.unique(np.concatenate([grid, curve_pts, [0.5]]))

    return np.sort(np.unique(np.round(grid, 6)))


def evaluate_threshold_candidates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: np.ndarray,
    true_label_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    ベクトル演算により全閾値候補に対する混同行列および3大基準のスコアを一括算出します。

    Returns:
        pd.DataFrame: 各行が1つの閾値候補 th に対応する評価指標テーブル
            - threshold
            - tp, fp, tn, fn
            - recall (sensitivity / TPR)
            - specificity (TNR)
            - precision (PPV)
            - npv
            - accuracy
            - balanced_acc
            - youden_j
            - f1_pos (集中クラス F1)
            - f1_neg (休憩クラス F1)
            - f1_macro
            - pr_diff (|Precision - Recall|)
            - pr_product (Precision * Recall)
            - pr_balance_diff_score (1 - |P - R|)
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_arr = y_true_arr[mask]
    y_pred_arr = y_pred_arr[mask]

    y_bin = (y_true_arr >= true_label_threshold).astype(np.int32)
    ths = np.asarray(thresholds, dtype=np.float64).reshape(-1, 1)  # [M, 1]
    preds_row = y_pred_arr.reshape(1, -1)                          # [1, N]
    y_row = y_bin.reshape(1, -1)                                   # [1, N]

    pred_pos = preds_row >= ths                                    # [M, N] bool
    pred_neg = ~pred_pos
    true_pos = y_row == 1
    true_neg = y_row == 0

    tp = np.sum(pred_pos & true_pos, axis=1).astype(np.float64)
    fp = np.sum(pred_pos & true_neg, axis=1).astype(np.float64)
    tn = np.sum(pred_neg & true_neg, axis=1).astype(np.float64)
    fn = np.sum(pred_neg & true_pos, axis=1).astype(np.float64)

    n_total = np.maximum(1.0, tp + fp + tn + fn)
    pos_total = tp + fn
    neg_total = tn + fp
    pred_pos_total = tp + fp
    pred_neg_total = tn + fn

    recall = np.where(pos_total > 0, tp / np.maximum(1.0, pos_total), 0.0)
    specificity = np.where(neg_total > 0, tn / np.maximum(1.0, neg_total), 0.0)
    precision = np.where(pred_pos_total > 0, tp / np.maximum(1.0, pred_pos_total), 0.0)
    npv = np.where(pred_neg_total > 0, tn / np.maximum(1.0, pred_neg_total), 0.0)

    accuracy = (tp + tn) / n_total
    balanced_acc = 0.5 * (recall + specificity)
    youden_j = recall + specificity - 1.0  # == 2 * balanced_acc - 1

    # Class 1 (Focus) F1 & Class 0 (Rest) F1
    denom_pos = 2.0 * tp + fp + fn
    f1_pos = np.where(denom_pos > 0, (2.0 * tp) / np.maximum(1.0, denom_pos), 0.0)

    denom_neg = 2.0 * tn + fn + fp
    f1_neg = np.where(denom_neg > 0, (2.0 * tn) / np.maximum(1.0, denom_neg), 0.0)

    f1_macro = 0.5 * (f1_pos + f1_neg)

    # Precision-Recall balance metrics
    pr_diff = np.abs(precision - recall)
    pr_product = precision * recall
    # P=0, R=0 の自明な退化点を除外したバランススコア: (1 - |P - R|)
    valid_pr = (precision > 1e-6) & (recall > 1e-6)
    pr_balance_diff_score = np.where(valid_pr, 1.0 - pr_diff, -1.0)

    df_curve = pd.DataFrame({
        "threshold": ths.ravel(),
        "tp": tp.astype(int),
        "fp": fp.astype(int),
        "tn": tn.astype(int),
        "fn": fn.astype(int),
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "accuracy": accuracy,
        "balanced_acc": balanced_acc,
        "youden_j": youden_j,
        "f1_pos": f1_pos,
        "f1_neg": f1_neg,
        "f1_macro": f1_macro,
        "pr_diff": pr_diff,
        "pr_product": pr_product,
        "pr_balance_diff_score": pr_balance_diff_score,
    })
    return df_curve


def select_best_threshold_from_curve(
    df_curve: pd.DataFrame,
    criterion: str = "youden_j",
    pr_balance_mode: str = "min_diff",
    fallback_threshold: float = 0.5,
) -> Tuple[float, float, Dict]:
    """
    評価曲線 DataFrame から、指定基準を最大化する最適閾値 th* を決定します。
    同点（タイ）が発生した場合は、F1-Macro が高く、かつ 0.5 に近い安定した閾値を優先します。

    Returns:
        best_th (float): 最適カットオフ閾値 th*
        best_score (float): 基準の最大スコア値
        best_row_dict (Dict): 最適閾値時点での全評価指標辞書
    """
    if df_curve is None or df_curve.empty:
        return float(fallback_threshold), np.nan, {"threshold": float(fallback_threshold)}

    canon_crit = normalize_criterion_name(criterion)
    pr_mode = str(pr_balance_mode or "min_diff").strip().lower()

    if canon_crit == "youden_j":
        score_col = "youden_j"
        scores = df_curve[score_col].values
    elif canon_crit == "f1_macro":
        score_col = "f1_macro"
        scores = df_curve[score_col].values
    elif canon_crit == "pr_product" or (canon_crit == "pr_balance" and pr_mode in ("product", "prod", "max_product")):
        score_col = "pr_product"
        scores = df_curve[score_col].values
    else:
        # pr_min_diff または pr_balance (min_diff モード)
        score_col = "pr_balance_diff_score"
        scores = df_curve[score_col].values
        # 全候補が退化していた場合は pr_product にフォールバック
        if np.all(scores < 0):
            score_col = "pr_product"
            scores = df_curve[score_col].values

    valid_mask = np.isfinite(scores)
    if not np.any(valid_mask):
        return float(fallback_threshold), np.nan, {"threshold": float(fallback_threshold)}

    max_score = float(np.nanmax(scores[valid_mask]))
    # 浮動小数点許容誤差内で最大スコアを達成する候補を抽出
    top_indices = np.where(valid_mask & (scores >= max_score - 1e-7))[0]

    if len(top_indices) == 1:
        best_idx = int(top_indices[0])
    else:
        # タイブレーク: F1-Macro + Balanced Acc が高く、0.5 に近い点を選択
        sub = df_curve.iloc[top_indices]
        tie_breaker = (
            sub["f1_macro"].values * 0.5
            + sub["balanced_acc"].values * 0.5
            - 0.01 * np.abs(sub["threshold"].values - 0.5)
        )
        best_idx = int(top_indices[int(np.argmax(tie_breaker))])

    best_row = df_curve.iloc[best_idx].to_dict()
    best_th = float(best_row["threshold"])
    best_score = float(best_row[score_col])
    best_row["criterion"] = canon_crit
    best_row["criterion_score"] = best_score
    return best_th, best_score, best_row


def find_optimal_threshold(
    y_val_true: np.ndarray,
    y_val_pred: np.ndarray,
    criterion: str = "youden_j",
    pr_balance_mode: str = "min_diff",
    search_range: Sequence[float] = (0.05, 0.95),
    search_step: float = 0.005,
    use_exact_curve_points: bool = True,
    fallback_threshold: float = 0.5,
) -> Dict:
    """
    検証データ (y_val_true, y_val_pred) から最適閾値 th* と全3大基準の比較結果を返します。
    """
    y_true_arr = np.asarray(y_val_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_val_pred, dtype=np.float64)
    valid = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_arr = y_true_arr[valid]
    y_pred_arr = y_pred_arr[valid]

    y_bin = (y_true_arr >= 0.5).astype(int)
    if len(y_bin) == 0 or len(np.unique(y_bin)) < 2:
        # 単一クラスしか検証データに存在しない場合は fallback_threshold を返す
        return {
            "best_threshold": float(fallback_threshold),
            "best_score": np.nan,
            "criterion": normalize_criterion_name(criterion),
            "all_criteria_thresholds": {
                "youden_j": float(fallback_threshold),
                "f1_macro": float(fallback_threshold),
                "pr_balance": float(fallback_threshold),
            },
            "curve_df": pd.DataFrame(),
            "best_metrics_val": {"threshold": float(fallback_threshold)},
        }

    thresholds = build_candidate_thresholds(
        y_pred=y_pred_arr,
        search_range=search_range,
        search_step=search_step,
        use_exact_curve_points=use_exact_curve_points,
    )
    df_curve = evaluate_threshold_candidates(
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        thresholds=thresholds,
    )

    best_th, best_score, best_row = select_best_threshold_from_curve(
        df_curve=df_curve,
        criterion=criterion,
        pr_balance_mode=pr_balance_mode,
        fallback_threshold=fallback_threshold,
    )

    # 3大基準すべての最適閾値も同時に記録
    th_youden, _, row_youden = select_best_threshold_from_curve(
        df_curve, criterion="youden_j", pr_balance_mode=pr_balance_mode, fallback_threshold=fallback_threshold
    )
    th_f1, _, row_f1 = select_best_threshold_from_curve(
        df_curve, criterion="f1_macro", pr_balance_mode=pr_balance_mode, fallback_threshold=fallback_threshold
    )
    th_pr, _, row_pr = select_best_threshold_from_curve(
        df_curve, criterion="pr_balance", pr_balance_mode=pr_balance_mode, fallback_threshold=fallback_threshold
    )

    return {
        "best_threshold": best_th,
        "best_score": best_score,
        "criterion": normalize_criterion_name(criterion),
        "all_criteria_thresholds": {
            "youden_j": th_youden,
            "f1_macro": th_f1,
            "pr_balance": th_pr,
        },
        "all_criteria_val_rows": {
            "youden_j": row_youden,
            "f1_macro": row_f1,
            "pr_balance": row_pr,
        },
        "curve_df": df_curve,
        "best_metrics_val": best_row,
    }
