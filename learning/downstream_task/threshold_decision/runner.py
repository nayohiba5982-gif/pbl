"""
LOSO-CV 完全データリークフリー 閾値決定（Threshold Tuning）実行エンジン
(learning/downstream_task/threshold_decision/runner.py)

テスト被験者（Test）のデータを一切参照せず、各 LOSO Fold 内の
学習・検証データ（Train/Val）のみから 3大基準（Youden's J, F1-Macro, PR-Balance）に基づき
最適カットオフ閾値 th*_k を決定し、テスト被験者の二値化判定へ適用します。
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, train_test_split

try:
    from ..metrics import compute_fold_metrics
    from .criteria import (
        CRITERION_LABELS,
        build_candidate_thresholds,
        evaluate_threshold_candidates,
        find_optimal_threshold,
        normalize_criterion_name,
    )
except (ImportError, ValueError):
    from downstream_task.metrics import compute_fold_metrics
    from downstream_task.threshold_decision.criteria import (
        CRITERION_LABELS,
        build_candidate_thresholds,
        evaluate_threshold_candidates,
        find_optimal_threshold,
        normalize_criterion_name,
    )


def _compute_binary_classification_detail(
    y_true: np.ndarray,
    y_pred_cont: np.ndarray,
    y_pred_bin: np.ndarray,
    true_threshold: float = 0.5,
) -> Dict[str, float]:
    """二値化予測配列 y_pred_bin に対する詳細な二値分類評価指標を算出します。"""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_cont_arr = np.asarray(y_pred_cont, dtype=float)
    y_true_b = (y_true_arr >= true_threshold).astype(int)
    y_pred_b = np.asarray(y_pred_bin, dtype=int)

    n_classes = len(np.unique(y_true_b))
    tn, fp, fn, tp = confusion_matrix(y_true_b, y_pred_b, labels=[0, 1]).ravel()

    acc = float(accuracy_score(y_true_b, y_pred_b))
    recall_pos = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    precision_pos = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    if n_classes > 1 and pd.notna(recall_pos) and pd.notna(specificity):
        balanced_acc = 0.5 * (recall_pos + specificity)
        youden_j = recall_pos + specificity - 1.0
    else:
        balanced_acc = np.nan
        youden_j = np.nan

    f1_macro = float(f1_score(y_true_b, y_pred_b, average="macro", zero_division=0))
    f1_pos = float(f1_score(y_true_b, y_pred_b, pos_label=1, average="binary", zero_division=0))
    f1_neg = float(f1_score(y_true_b, y_pred_b, pos_label=0, average="binary", zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true_b, y_cont_arr)) if n_classes > 1 else np.nan
    except Exception:
        roc_auc = np.nan

    try:
        pr_auc = (
            float(average_precision_score(y_true_b, y_cont_arr))
            if (n_classes > 1 and np.sum(y_true_b == 1) > 0)
            else np.nan
        )
    except Exception:
        pr_auc = np.nan

    return {
        "accuracy": acc,
        "balanced_acc": balanced_acc,
        "youden_j": youden_j,
        "f1_macro": f1_macro,
        "f1_pos": f1_pos,
        "f1_neg": f1_neg,
        "precision": precision_pos,
        "recall": recall_pos,
        "specificity": specificity,
        "npv": npv,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def select_validation_indices_for_fold(
    train_idx: np.ndarray,
    subjects: np.ndarray,
    labels: np.ndarray,
    fold: int,
    val_split_mode: str = "oof_non_test",
    val_ratio: float = 0.2,
    group_split: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """
    LOSO Fold 内の非テストデータ (train_idx) から、閾値探索用の検証データ (Val) インデックスを抽出します。
    テスト被験者のデータは一切含まれません。

    Modes:
      - "oof_non_test" / "all_non_test":
          テスト被験者を除いた残り全被験者 (U-1名) の OOF 予測値を検証データとして使用（最も安定）
      - "group_val" / "inner_val":
          Early Stopping と同じ被験者単位 GroupShuffleSplit (val_ratio) で抽出した検証被験者を使用
          （抽出された Val に単一クラスしか含まれない場合は安全のため oof_non_test に自動フォールバック）
    """
    mode = str(val_split_mode or "oof_non_test").strip().lower()
    n_tr = len(train_idx)

    if mode in ("group_val", "inner_val", "val_split") and 0.0 < val_ratio < 0.9 and n_tr >= 10:
        train_sub_users = subjects[train_idx]
        unique_tr_users = np.unique(train_sub_users)
        if group_split and len(unique_tr_users) >= 3:
            gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed + fold)
            _, val_sub = next(gss.split(np.zeros(n_tr), groups=train_sub_users))
        else:
            _, val_sub = train_test_split(
                np.arange(n_tr), test_size=val_ratio, random_state=seed + fold
            )
        val_global_idx = train_idx[val_sub]
        val_y_bin = (labels[val_global_idx] >= 0.5).astype(int)
        if len(np.unique(val_y_bin)) >= 2:
            return val_global_idx

    # デフォルト / フォールバック: テスト被験者を除いた全非テスト被験者の OOF データを検証用に使用
    return train_idx


def run_loso_threshold_tuning(
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    subjects: Sequence,
    criterion: str = "youden_j",
    pr_balance_mode: str = "min_diff",
    search_range: Sequence[float] = (0.05, 0.95),
    search_step: float = 0.005,
    use_exact_curve_points: bool = True,
    val_split_mode: str = "oof_non_test",
    val_ratio: float = 0.2,
    group_split: bool = True,
    fallback_threshold: float = 0.5,
    baseline_threshold: float = 0.5,
    seed: int = 42,
    val_predictions_per_fold: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict:
    """
    LOSO-CV の各 Fold 内で検証データ (Val) を用いて最適閾値 th*_k を決定し、
    テスト被験者 (Test) に適用して評価指標を集計します。

    Parameters:
        y_true_all: 全サンプルの正解ラベル配列 [N]
        y_pred_all: 全サンプルの OOF 連続予測値配列 [N]
        subjects: 全サンプルの被験者ID配列 [N]
        criterion: 最適化基準 ("youden_j", "f1_macro", "pr_balance", "pr_min_diff", "pr_product")
        pr_balance_mode: PRバランス基準の計算モード ("min_diff" または "product")
        search_range: 閾値探索範囲 [min_th, max_th]
        search_step: グリッド刻み幅
        use_exact_curve_points: ROC/PR変化点を候補に追加するか
        val_split_mode: Fold内検証データの選択方式 ("oof_non_test" または "group_val")
        val_ratio: group_val 時の分割率
        group_split: 被験者単位分割を行うか
        fallback_threshold: 最適化不能時のデフォルト閾値 (0.5)
        baseline_threshold: 比較用ベースライン固定閾値 (0.5)
        seed: 乱数シード
        val_predictions_per_fold: 各 Fold 固有の (y_val_true, y_val_pred) 辞書がある場合に直接指定可能

    Returns:
        Dict:
            - criterion: 正規化後の基準名
            - criterion_label: 日本語説明ラベル
            - oof_binary_tuned: 各Foldの最適閾値 th*_k で二値化した予測配列 [N]
            - oof_binary_baseline: 固定閾値 (0.5) で二値化した予測配列 [N]
            - oof_thresholds: 各サンプルに適用された Fold 最適閾値配列 [N]
            - fold_thresholds: Fold ごとの最適閾値リスト
            - mean_threshold, std_threshold, median_threshold
            - tuned_metrics: 閾値調整後の全体 OOF & Fold Mean±Std 評価指標
            - baseline_metrics: 固定閾値 (0.5) の全体 OOF & Fold Mean±Std 評価指標
            - improvement: 固定閾値 (0.5) からの改善幅 (ΔBalanced Acc, ΔF1-Macro, ΔAccuracy 等)
            - fold_df: Fold 別詳細テーブル (th*_k, Valスコア, Test指標[0.5 vs Tuned])
            - global_val_curve_df: 全体検証曲線 DataFrame (可視化プロット用)
            - global_all_criteria_thresholds: 全体での3大基準それぞれの最適閾値辞書
    """
    canon_crit = normalize_criterion_name(criterion)
    y_true_arr = np.asarray(y_true_all, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred_all, dtype=np.float64)
    subj_arr = np.asarray(subjects).astype(str)
    n_samples = len(y_true_arr)

    oof_binary_tuned = np.zeros(n_samples, dtype=np.int32)
    oof_binary_base = (y_pred_arr >= baseline_threshold).astype(np.int32)
    oof_thresholds = np.full(n_samples, fallback_threshold, dtype=np.float32)

    fold_records: List[Dict] = []
    fold_thresholds: List[float] = []

    logo = LeaveOneGroupOut()
    for fold, (train_idx, test_idx) in enumerate(logo.split(y_pred_arr, y_true_arr, groups=subj_arr), 1):
        test_user = str(subj_arr[test_idx[0]])

        # 1. Fold 内の検証データ (Val) 取得（Test被験者は完全に除外）
        if val_predictions_per_fold is not None and fold in val_predictions_per_fold:
            y_val_true, y_val_pred = val_predictions_per_fold[fold]
        else:
            val_idx = select_validation_indices_for_fold(
                train_idx=train_idx,
                subjects=subj_arr,
                labels=y_true_arr,
                fold=fold,
                val_split_mode=val_split_mode,
                val_ratio=val_ratio,
                group_split=group_split,
                seed=seed,
            )
            y_val_true = y_true_arr[val_idx]
            y_val_pred = y_pred_arr[val_idx]

        # 2. Val データのみで最適閾値 th*_k を探索
        opt_res = find_optimal_threshold(
            y_val_true=y_val_true,
            y_val_pred=y_val_pred,
            criterion=canon_crit,
            pr_balance_mode=pr_balance_mode,
            search_range=search_range,
            search_step=search_step,
            use_exact_curve_points=use_exact_curve_points,
            fallback_threshold=fallback_threshold,
        )
        th_k = float(opt_res["best_threshold"])
        val_best_row = opt_res["best_metrics_val"]
        all_crit_ths = opt_res["all_criteria_thresholds"]

        fold_thresholds.append(th_k)
        oof_thresholds[test_idx] = th_k

        # 3. 決定した閾値 th*_k をテスト被験者 (Test) に適用して二値化
        y_test_true = y_true_arr[test_idx]
        y_test_pred = y_pred_arr[test_idx]
        y_test_bin_tuned = (y_test_pred >= th_k).astype(np.int32)
        y_test_bin_base = (y_test_pred >= baseline_threshold).astype(np.int32)

        oof_binary_tuned[test_idx] = y_test_bin_tuned

        # 4. Test 被験者での評価（固定0.5 vs 最適化閾値 th*_k）
        base_fold_m = compute_fold_metrics(
            fold=fold,
            test_user=test_user,
            y_true=y_test_true,
            y_pred=y_test_pred,
            threshold=baseline_threshold,
        )
        tuned_detail = _compute_binary_classification_detail(
            y_true=y_test_true,
            y_pred_cont=y_test_pred,
            y_pred_bin=y_test_bin_tuned,
            true_threshold=0.5,
        )
        base_detail = _compute_binary_classification_detail(
            y_true=y_test_true,
            y_pred_cont=y_test_pred,
            y_pred_bin=y_test_bin_base,
            true_threshold=0.5,
        )

        fold_records.append({
            "fold": int(fold),
            "test_user": test_user,
            "n_test": int(len(test_idx)),
            "n_val": int(len(y_val_true)),
            "optimal_threshold": round(th_k, 4),
            "th_youden_j": round(float(all_crit_ths["youden_j"]), 4),
            "th_f1_macro": round(float(all_crit_ths["f1_macro"]), 4),
            "th_pr_balance": round(float(all_crit_ths["pr_balance"]), 4),
            "val_criterion_score": round(float(opt_res["best_score"]), 4) if pd.notna(opt_res["best_score"]) else np.nan,
            "val_balanced_acc": round(float(val_best_row.get("balanced_acc", np.nan)), 4),
            "val_f1_macro": round(float(val_best_row.get("f1_macro", np.nan)), 4),
            # 回帰・曲線指標 (閾値不変)
            "mae": base_fold_m["mae"],
            "rmse": base_fold_m["rmse"],
            "r2": base_fold_m["r2"],
            "pearson_r": base_fold_m["pearson_r"],
            "spearman_rho": base_fold_m["spearman_rho"],
            "roc_auc": base_fold_m["roc_auc"],
            "pr_auc": base_fold_m["pr_auc"],
            # Tuned 二値指標 (th = th*_k)
            "acc_binary": round(tuned_detail["accuracy"], 4),
            "balanced_acc": round(tuned_detail["balanced_acc"], 4) if pd.notna(tuned_detail["balanced_acc"]) else np.nan,
            "youden_j": round(tuned_detail["youden_j"], 4) if pd.notna(tuned_detail["youden_j"]) else np.nan,
            "f1_macro": round(tuned_detail["f1_macro"], 4),
            "f1_pos": round(tuned_detail["f1_pos"], 4),
            "f1_neg": round(tuned_detail["f1_neg"], 4),
            "precision": round(tuned_detail["precision"], 4),
            "recall": round(tuned_detail["recall"], 4) if pd.notna(tuned_detail["recall"]) else np.nan,
            "specificity": round(tuned_detail["specificity"], 4) if pd.notna(tuned_detail["specificity"]) else np.nan,
            # Baseline 二値指標 (th = 0.5)
            "acc_base_05": round(base_detail["accuracy"], 4),
            "balanced_acc_base_05": round(base_detail["balanced_acc"], 4) if pd.notna(base_detail["balanced_acc"]) else np.nan,
            "f1_macro_base_05": round(base_detail["f1_macro"], 4),
            "f1_neg_base_05": round(base_detail["f1_neg"], 4),
            "specificity_base_05": round(base_detail["specificity"], 4) if pd.notna(base_detail["specificity"]) else np.nan,
            # 改善幅 (Tuned - Baseline 0.5)
            "delta_balanced_acc": (
                round(tuned_detail["balanced_acc"] - base_detail["balanced_acc"], 4)
                if pd.notna(tuned_detail["balanced_acc"]) and pd.notna(base_detail["balanced_acc"])
                else np.nan
            ),
            "delta_f1_macro": round(tuned_detail["f1_macro"] - base_detail["f1_macro"], 4),
            "delta_acc": round(tuned_detail["accuracy"] - base_detail["accuracy"], 4),
        })

    fold_df = pd.DataFrame(fold_records)

    # 全体 OOF 評価 (Tuned vs Baseline 0.5)
    oof_tuned_detail = _compute_binary_classification_detail(
        y_true=y_true_arr,
        y_pred_cont=y_pred_arr,
        y_pred_bin=oof_binary_tuned,
        true_threshold=0.5,
    )
    oof_base_detail = _compute_binary_classification_detail(
        y_true=y_true_arr,
        y_pred_cont=y_pred_arr,
        y_pred_bin=oof_binary_base,
        true_threshold=0.5,
    )

    def _mean_std(col: str) -> Tuple[float, float]:
        if col not in fold_df.columns:
            return np.nan, np.nan
        s = fold_df[col].dropna()
        if len(s) == 0:
            return np.nan, np.nan
        return float(s.mean()), (float(s.std(ddof=1)) if len(s) > 1 else 0.0)

    tuned_stats = {
        "accuracy": _mean_std("acc_binary"),
        "balanced_acc": _mean_std("balanced_acc"),
        "youden_j": _mean_std("youden_j"),
        "f1_macro": _mean_std("f1_macro"),
        "f1_pos": _mean_std("f1_pos"),
        "f1_neg": _mean_std("f1_neg"),
        "precision": _mean_std("precision"),
        "recall": _mean_std("recall"),
        "specificity": _mean_std("specificity"),
        "roc_auc": _mean_std("roc_auc"),
        "pr_auc": _mean_std("pr_auc"),
        "mae": _mean_std("mae"),
        "rmse": _mean_std("rmse"),
        "r2": _mean_std("r2"),
        "pearson_r": _mean_std("pearson_r"),
        "spearman_rho": _mean_std("spearman_rho"),
    }
    base_stats = {
        "accuracy": _mean_std("acc_base_05"),
        "balanced_acc": _mean_std("balanced_acc_base_05"),
        "f1_macro": _mean_std("f1_macro_base_05"),
        "f1_neg": _mean_std("f1_neg_base_05"),
        "specificity": _mean_std("specificity_base_05"),
    }

    # 全体曲線（可視化・プロファイル用）
    global_opt = find_optimal_threshold(
        y_val_true=y_true_arr,
        y_val_pred=y_pred_arr,
        criterion=canon_crit,
        pr_balance_mode=pr_balance_mode,
        search_range=search_range,
        search_step=search_step,
        use_exact_curve_points=use_exact_curve_points,
        fallback_threshold=fallback_threshold,
    )

    th_arr = np.asarray(fold_thresholds, dtype=float)
    mean_th = float(np.mean(th_arr)) if len(th_arr) > 0 else float(fallback_threshold)
    std_th = float(np.std(th_arr, ddof=1)) if len(th_arr) > 1 else 0.0
    median_th = float(np.median(th_arr)) if len(th_arr) > 0 else float(fallback_threshold)

    improvement = {
        "oof_balanced_acc_diff": float(oof_tuned_detail["balanced_acc"] - oof_base_detail["balanced_acc"]),
        "oof_f1_macro_diff": float(oof_tuned_detail["f1_macro"] - oof_base_detail["f1_macro"]),
        "oof_f1_neg_diff": float(oof_tuned_detail["f1_neg"] - oof_base_detail["f1_neg"]),
        "oof_specificity_diff": float(oof_tuned_detail["specificity"] - oof_base_detail["specificity"]),
        "oof_accuracy_diff": float(oof_tuned_detail["accuracy"] - oof_base_detail["accuracy"]),
        "fold_mean_balanced_acc_diff": float(tuned_stats["balanced_acc"][0] - base_stats["balanced_acc"][0]),
        "fold_mean_f1_macro_diff": float(tuned_stats["f1_macro"][0] - base_stats["f1_macro"][0]),
    }

    return {
        "criterion": canon_crit,
        "criterion_label": CRITERION_LABELS.get(canon_crit, canon_crit),
        "pr_balance_mode": pr_balance_mode,
        "val_split_mode": val_split_mode,
        "oof_binary_tuned": oof_binary_tuned,
        "oof_binary_baseline": oof_binary_base,
        "oof_thresholds": oof_thresholds,
        "fold_thresholds": fold_thresholds,
        "mean_threshold": mean_th,
        "std_threshold": std_th,
        "median_threshold": median_th,
        "tuned_oof_detail": oof_tuned_detail,
        "baseline_oof_detail": oof_base_detail,
        "tuned_fold_stats": tuned_stats,
        "baseline_fold_stats": base_stats,
        "improvement": improvement,
        "fold_df": fold_df,
        "global_val_curve_df": global_opt["curve_df"],
        "global_all_criteria_thresholds": global_opt["all_criteria_thresholds"],
    }
