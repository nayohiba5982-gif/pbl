"""
共通評価指標算出モジュール
(learning/downstream_task/metrics.py)

回帰指標 (MAE, RMSE, R², Pearson r, Spearman ρ) および
二値判定指標 (ROC-AUC, PR-AUC, Accuracy, Balanced Accuracy, F1-Macro) の計算と
Fold 間統計量 (Mean ± Std) の集計を提供します。
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


METRIC_KEYS = [
    "mae",
    "rmse",
    "r2",
    "pearson_r",
    "spearman_rho",
    "accuracy",
    "balanced_acc",
    "roc_auc",
    "pr_auc",
    "f1_macro",
]


def compute_fold_metrics(
    fold: int,
    test_user: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """単一 Fold (被験者 1 名) の回帰および二値分類評価指標を算出します。"""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    n_test = len(y_true_arr)

    y_true_b = (y_true_arr >= threshold).astype(int)
    y_pred_b = (y_pred_arr >= threshold).astype(int)
    n_unique_y = len(np.unique(y_true_arr))
    n_classes = len(np.unique(y_true_b))

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))

    try:
        r2_val = float(r2_score(y_true_arr, y_pred_arr)) if n_unique_y > 1 else np.nan
    except Exception:
        r2_val = np.nan

    try:
        if n_unique_y > 1 and np.std(y_pred_arr) > 1e-12:
            r_val = float(pearsonr(y_true_arr, y_pred_arr)[0])
        else:
            r_val = np.nan
    except Exception:
        r_val = np.nan

    try:
        if n_unique_y > 1 and np.std(y_pred_arr) > 1e-12:
            rho_val = float(spearmanr(y_true_arr, y_pred_arr)[0])
        else:
            rho_val = np.nan
    except Exception:
        rho_val = np.nan

    acc = float(accuracy_score(y_true_b, y_pred_b))
    try:
        balanced_acc = float(balanced_accuracy_score(y_true_b, y_pred_b)) if n_classes > 1 else np.nan
    except Exception:
        balanced_acc = np.nan

    try:
        auc_val = float(roc_auc_score(y_true_b, y_pred_arr)) if n_classes > 1 else np.nan
    except Exception:
        auc_val = np.nan

    try:
        pr_auc_val = (
            float(average_precision_score(y_true_b, y_pred_arr))
            if (n_classes > 1 and np.sum(y_true_b == 1) > 0)
            else np.nan
        )
    except Exception:
        pr_auc_val = np.nan

    try:
        f1_macro_val = float(f1_score(y_true_b, y_pred_b, average="macro", zero_division=0))
    except Exception:
        f1_macro_val = np.nan

    return {
        "fold": int(fold),
        "test_user": str(test_user),
        "n_test": int(n_test),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2_val, 4) if not np.isnan(r2_val) else np.nan,
        "pearson_r": round(r_val, 4) if not np.isnan(r_val) else np.nan,
        "spearman_rho": round(rho_val, 4) if not np.isnan(rho_val) else np.nan,
        "acc_binary": round(acc, 4),
        "balanced_acc": round(balanced_acc, 4) if not np.isnan(balanced_acc) else np.nan,
        "roc_auc": round(auc_val, 4) if not np.isnan(auc_val) else np.nan,
        "pr_auc": round(pr_auc_val, 4) if not np.isnan(pr_auc_val) else np.nan,
        "f1_macro": round(f1_macro_val, 4) if not np.isnan(f1_macro_val) else np.nan,
    }


def compute_oof_and_fold_summary(
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    fold_records: List[Dict],
    threshold: float = 0.5,
) -> Dict:
    """全体 Out-Of-Fold (OOF) 指標および Fold 間 Mean ± Std を集計します。"""
    y_true_arr = np.asarray(y_true_all, dtype=float)
    y_pred_arr = np.asarray(y_pred_all, dtype=float)

    mae_total = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse_total = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    r2_total = float(r2_score(y_true_arr, y_pred_arr))

    if np.std(y_true_arr) > 1e-12 and np.std(y_pred_arr) > 1e-12:
        r_total, p_val = pearsonr(y_true_arr, y_pred_arr)
        rho_total, _ = spearmanr(y_true_arr, y_pred_arr)
        r_total = float(r_total)
        p_val = float(p_val)
        rho_total = float(rho_total)
    else:
        r_total, p_val, rho_total = 0.0, 1.0, 0.0

    y_true_bin = (y_true_arr >= threshold).astype(int)
    y_pred_bin = (y_pred_arr >= threshold).astype(int)
    acc_total = float(accuracy_score(y_true_bin, y_pred_bin))
    balanced_acc_total = float(balanced_accuracy_score(y_true_bin, y_pred_bin))
    f1_macro_total = float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0))

    try:
        auc_total = float(roc_auc_score(y_true_bin, y_pred_arr))
    except Exception:
        auc_total = np.nan

    try:
        pr_auc_total = float(average_precision_score(y_true_bin, y_pred_arr))
    except Exception:
        pr_auc_total = np.nan

    fold_df = pd.DataFrame(fold_records)

    def calc_mean_std(col_name: str) -> Tuple[float, float]:
        if col_name not in fold_df.columns:
            return np.nan, np.nan
        s = fold_df[col_name].dropna()
        if len(s) == 0:
            return np.nan, np.nan
        m_val = float(s.mean())
        s_val = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        return m_val, s_val

    stats_dict = {
        "mae": calc_mean_std("mae"),
        "rmse": calc_mean_std("rmse"),
        "r2": calc_mean_std("r2"),
        "pearson_r": calc_mean_std("pearson_r"),
        "spearman_rho": calc_mean_std("spearman_rho"),
        "accuracy": calc_mean_std("acc_binary"),
        "balanced_acc": calc_mean_std("balanced_acc"),
        "roc_auc": calc_mean_std("roc_auc"),
        "pr_auc": calc_mean_std("pr_auc"),
        "f1_macro": calc_mean_std("f1_macro"),
    }

    return {
        "mae": mae_total,
        "rmse": rmse_total,
        "r2": r2_total,
        "pearson_r": r_total,
        "pearson_p": p_val,
        "spearman_rho": rho_total,
        "accuracy": acc_total,
        "balanced_acc": balanced_acc_total,
        "roc_auc": auc_total,
        "pr_auc": pr_auc_total,
        "f1_macro": f1_macro_total,
        "stats_dict": stats_dict,
        "fold_df": fold_df,
    }


def build_summary_row(model_name: str, eval_res: Dict, extra_fields: Dict | None = None) -> Dict:
    """モデル比較テーブル用の1行分の辞書を構築します。"""
    st = eval_res["stats_dict"]
    row = {"model": model_name}
    if extra_fields:
        row.update(extra_fields)

    row.update({
        "roc_auc": eval_res["roc_auc"],
        "roc_auc_mean": st["roc_auc"][0],
        "roc_auc_std": st["roc_auc"][1],
        "pr_auc": eval_res["pr_auc"],
        "pr_auc_mean": st["pr_auc"][0],
        "pr_auc_std": st["pr_auc"][1],
        "pearson_r": eval_res["pearson_r"],
        "pearson_r_mean": st["pearson_r"][0],
        "pearson_r_std": st["pearson_r"][1],
        "spearman_rho": eval_res["spearman_rho"],
        "spearman_rho_mean": st["spearman_rho"][0],
        "spearman_rho_std": st["spearman_rho"][1],
        "mae": eval_res["mae"],
        "mae_mean": st["mae"][0],
        "mae_std": st["mae"][1],
        "rmse": eval_res["rmse"],
        "rmse_mean": st["rmse"][0],
        "rmse_std": st["rmse"][1],
        "r2": eval_res["r2"],
        "r2_mean": st["r2"][0],
        "r2_std": st["r2"][1],
        "accuracy": eval_res["accuracy"],
        "accuracy_mean": st["accuracy"][0],
        "accuracy_std": st["accuracy"][1],
        "balanced_acc": eval_res["balanced_acc"],
        "balanced_acc_mean": st["balanced_acc"][0],
        "balanced_acc_std": st["balanced_acc"][1],
        "f1_macro": eval_res["f1_macro"],
        "f1_macro_mean": st["f1_macro"][0],
        "f1_macro_std": st["f1_macro"][1],
    })
    return row
