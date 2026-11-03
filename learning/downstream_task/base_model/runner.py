"""
ベースモデル (Visual / Vital) LOSO-CV 学習・評価・モデル保存実行エンジン
(learning/downstream_task/base_model/runner.py)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, train_test_split

try:
    from ..data_loader import (
        build_modality_feature_names,
        split_modality_feature_indices,
        standardize_subject_wise,
    )
    from ..metrics import compute_fold_metrics, compute_oof_and_fold_summary
    from ..pls import DualStreamPLS
    from ..threshold_decision.criteria import find_optimal_threshold
    from .models import create_base_model, normalize_base_model_name
    from .reporter import generate_visualizations, save_markdown_report
except (ImportError, ValueError):
    from downstream_task.data_loader import (
        build_modality_feature_names,
        split_modality_feature_indices,
        standardize_subject_wise,
    )
    from downstream_task.metrics import compute_fold_metrics, compute_oof_and_fold_summary
    from downstream_task.pls import DualStreamPLS
    from downstream_task.threshold_decision.criteria import find_optimal_threshold
    from downstream_task.base_model.models import create_base_model, normalize_base_model_name
    from downstream_task.base_model.reporter import generate_visualizations, save_markdown_report


@dataclass
class LOSOFoldData:
    """単一 LOSO Fold のリークフリー前処理済みデータコンテナ"""
    fold: int
    test_user: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    inner_tr_mask: np.ndarray
    inner_val_mask: np.ndarray
    X_train_visual: np.ndarray
    X_test_visual: np.ndarray
    X_train_vital: np.ndarray
    X_test_vital: np.ndarray
    y_soft_train: np.ndarray
    y_soft_test: np.ndarray
    sample_weights_train: np.ndarray
    sample_weights_test: np.ndarray
    y_true_test: np.ndarray


def prepare_loso_folds(
    emb_vis: np.ndarray,
    emb_bio: np.ndarray,
    meta_df: pd.DataFrame,
    explicit_feats: Optional[np.ndarray],
    explicit_names: Optional[List[str]],
    cfg: Optional[dict],
    log,
    n_vis_pls: int = 3,
    n_bio_pls: int = 3,
    cov_type: str = "full",
    weights: Tuple[float, float] = (1.0, 0.0),
    apply_subject_norm: bool = True,
) -> Tuple[List[LOSOFoldData], List[str], List[str], np.ndarray, np.ndarray]:
    """
    全 LOSO Fold について、テスト被験者を完全除外したリークフリーな手順で
    1. DualStreamPLS 次元削減
    2. 被験者別 Z スコア標準化
    3. GMM (k=2) による連続値ソフトラベル y_soft & サンプル重み算出
    4. Visual / Vital モダリティ特徴量分離
    5. Early Stopping 用の Fold 内 Train / Val 分割
    を一括構築します。
    """
    subject_col = "username" if "username" in meta_df.columns else "subject_id"
    label_col = "label" if "label" in meta_df.columns else "label_mean"

    dt_cfg = cfg.get("downstream_task", {}) if cfg else {}
    loso_cfg = cfg.get("loso_cv", {}) if cfg else {}
    bm_cfg = dt_cfg.get("base_model", {})
    es_cfg = bm_cfg.get("early_stopping", {})

    seed = int(loso_cfg.get("random_state", dt_cfg.get("random_state", 42)))
    imbalance_penalty = float(dt_cfg.get("imbalance_penalty", 4.0))
    use_explicit = bool(dt_cfg.get("use_explicit_features", True))
    val_ratio = float(es_cfg.get("val_ratio", 0.2))
    group_split = bool(es_cfg.get("group_split", True))

    subjects = meta_df[subject_col].astype(str).values
    labels = meta_df[label_col].values.astype(float)

    actual_vis_comps = min(n_vis_pls, emb_vis.shape[1])
    actual_bio_comps = min(n_bio_pls, emb_bio.shape[1])

    use_exp_arr = explicit_feats if (use_explicit and explicit_feats is not None) else None
    n_explicit = use_exp_arr.shape[1] if use_exp_arr is not None else 0

    vis_indices, vital_indices = split_modality_feature_indices(
        n_vis_pls=actual_vis_comps,
        n_bio_pls=actual_bio_comps,
        explicit_names=explicit_names,
        n_explicit=n_explicit,
    )
    _, vis_feat_names, vital_feat_names = build_modality_feature_names(
        n_vis_pls=actual_vis_comps,
        n_bio_pls=actual_bio_comps,
        explicit_names=explicit_names,
        n_explicit=n_explicit,
    )

    logo = LeaveOneGroupOut()
    folds_data: List[LOSOFoldData] = []
    n_samples = len(meta_df)
    oof_gmm_soft = np.full(n_samples, np.nan, dtype=np.float32)
    oof_sample_weight = np.full(n_samples, np.nan, dtype=np.float32)

    for fold, (train_idx, test_idx) in enumerate(logo.split(emb_vis, labels, groups=subjects), 1):
        test_user = str(subjects[test_idx[0]])
        train_sub_users = subjects[train_idx]
        test_sub_users = subjects[test_idx]

        # A. Fold 内 DualStream PLS
        if actual_vis_comps < emb_vis.shape[1] or actual_bio_comps < emb_bio.shape[1]:
            pls = DualStreamPLS(n_vis=actual_vis_comps, n_bio=actual_bio_comps)
            pls.fit(emb_vis[train_idx], emb_bio[train_idx], labels[train_idx])
            z_train = pls.transform(emb_vis[train_idx], emb_bio[train_idx])
            z_test = pls.transform(emb_vis[test_idx], emb_bio[test_idx])
        else:
            z_train = np.concatenate([emb_vis[train_idx], emb_bio[train_idx]], axis=1).astype(np.float32)
            z_test = np.concatenate([emb_vis[test_idx], emb_bio[test_idx]], axis=1).astype(np.float32)

        # B. 被験者別 Zスコア標準化
        if apply_subject_norm:
            z_train_norm = standardize_subject_wise(z_train, train_sub_users)
            z_test_norm = standardize_subject_wise(z_test, test_sub_users)
        else:
            z_train_norm = z_train
            z_test_norm = z_test

        # C. Fold 内 GMM fit & ソフトラベル算出
        gmm = GaussianMixture(n_components=2, covariance_type=cov_type, random_state=seed)
        gmm.fit(z_train_norm)
        gmm_probs_train = gmm.predict_proba(z_train_norm)
        gmm_probs_test = gmm.predict_proba(z_test_norm)

        r0 = np.corrcoef(gmm_probs_train[:, 0], labels[train_idx])[0, 1]
        r1 = np.corrcoef(gmm_probs_train[:, 1], labels[train_idx])[0, 1]
        r0 = 0.0 if np.isnan(r0) else r0
        r1 = 0.0 if np.isnan(r1) else r1

        focus_c = 1 if r1 > r0 else 0
        relax_c = 1 - focus_c

        y_soft_train = (gmm_probs_train[:, focus_c] * weights[0] + gmm_probs_train[:, relax_c] * weights[1]).astype(np.float32)
        y_soft_test = (gmm_probs_test[:, focus_c] * weights[0] + gmm_probs_test[:, relax_c] * weights[1]).astype(np.float32)

        conf_w_train = np.max(gmm_probs_train, axis=1)
        imb_w_train = y_soft_train * 1.0 + (1.0 - y_soft_train) * imbalance_penalty
        sample_weights_train = (conf_w_train * imb_w_train).astype(np.float32)

        conf_w_test = np.max(gmm_probs_test, axis=1)
        imb_w_test = y_soft_test * 1.0 + (1.0 - y_soft_test) * imbalance_penalty
        sample_weights_test = (conf_w_test * imb_w_test).astype(np.float32)

        oof_gmm_soft[test_idx] = y_soft_test
        oof_sample_weight[test_idx] = sample_weights_test

        # D. 明示的物理特徴量の結合 & モダリティ分離
        if use_exp_arr is not None:
            if apply_subject_norm:
                stat_train_norm = standardize_subject_wise(use_exp_arr[train_idx], train_sub_users)
                stat_test_norm = standardize_subject_wise(use_exp_arr[test_idx], test_sub_users)
            else:
                stat_train_norm = use_exp_arr[train_idx]
                stat_test_norm = use_exp_arr[test_idx]
            X_train_full = np.concatenate([z_train_norm, stat_train_norm], axis=1)
            X_test_full = np.concatenate([z_test_norm, stat_test_norm], axis=1)
        else:
            X_train_full = z_train_norm
            X_test_full = z_test_norm

        X_train_visual = X_train_full[:, vis_indices].astype(np.float32)
        X_test_visual = X_test_full[:, vis_indices].astype(np.float32)
        X_train_vital = X_train_full[:, vital_indices].astype(np.float32)
        X_test_vital = X_test_full[:, vital_indices].astype(np.float32)

        # E. Early Stopping 用の Fold 内 Train / Val 分割 (被験者単位 GroupShuffleSplit 優先)
        n_tr = len(train_idx)
        inner_tr_mask = np.ones(n_tr, dtype=bool)
        inner_val_mask = np.zeros(n_tr, dtype=bool)

        if 0.0 < val_ratio < 0.9 and n_tr >= 10:
            unique_tr_users = np.unique(train_sub_users)
            if group_split and len(unique_tr_users) >= 3:
                gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed + fold)
                tr_sub, val_sub = next(gss.split(np.zeros(n_tr), groups=train_sub_users))
            else:
                tr_sub, val_sub = train_test_split(
                    np.arange(n_tr), test_size=val_ratio, random_state=seed + fold
                )
            inner_tr_mask[:] = False
            inner_tr_mask[tr_sub] = True
            inner_val_mask[val_sub] = True

        folds_data.append(
            LOSOFoldData(
                fold=fold,
                test_user=test_user,
                train_idx=train_idx,
                test_idx=test_idx,
                inner_tr_mask=inner_tr_mask,
                inner_val_mask=inner_val_mask,
                X_train_visual=X_train_visual,
                X_test_visual=X_test_visual,
                X_train_vital=X_train_vital,
                X_test_vital=X_test_vital,
                y_soft_train=y_soft_train,
                y_soft_test=y_soft_test,
                sample_weights_train=sample_weights_train,
                sample_weights_test=sample_weights_test,
                y_true_test=labels[test_idx],
            )
        )

    return folds_data, vis_feat_names, vital_feat_names, oof_gmm_soft, oof_sample_weight


def run_single_base_model_loso(
    modality: str,
    model_name: str,
    folds_data: List[LOSOFoldData],
    feature_names: List[str],
    meta_df: pd.DataFrame,
    oof_gmm_soft: np.ndarray,
    oof_sample_weight: np.ndarray,
    window_size: str,
    modality_output_dir: Path,
    cfg: Optional[dict],
    log,
) -> Dict:
    """
    特定モダリティ (visual または vital) の単一ベースモデルについて
    LOSO-CV (Early Stopping 付き) を実行し、評価結果・予測値・学習済みモデルを保存します。
    """
    canonical_name = normalize_base_model_name(model_name)
    modality = modality.lower().strip()
    assert modality in ("visual", "vital"), f"modality must be 'visual' or 'vital', got {modality}"

    dt_cfg = cfg.get("downstream_task", {}) if cfg else {}
    loso_cfg = cfg.get("loso_cv", {}) if cfg else {}
    bm_cfg = dt_cfg.get("base_model", {})
    es_cfg = bm_cfg.get("early_stopping", {})
    td_cfg = dt_cfg.get("threshold_decision", {})

    seed = int(loso_cfg.get("random_state", dt_cfg.get("random_state", 42)))
    save_plots = bool(dt_cfg.get("save_plots", True))
    save_models = bool(dt_cfg.get("save_models", True))
    es_enabled = bool(es_cfg.get("enabled", True))
    stop_rounds = int(es_cfg.get("stopping_rounds", dt_cfg.get("early_stopping_rounds", 30))) if es_enabled else 0

    td_enabled = bool(td_cfg.get("enabled", True))
    td_criterion = str(td_cfg.get("default_criterion", "youden_j"))
    td_pr_mode = str(td_cfg.get("pr_balance_mode", "min_diff"))
    td_range = list(td_cfg.get("search_range", [0.05, 0.95]))
    td_step = float(td_cfg.get("search_step", 0.005))
    td_fallback = float(td_cfg.get("fallback_threshold", 0.5))

    label_col = "label" if "label" in meta_df.columns else "label_mean"
    labels = meta_df[label_col].values.astype(float)
    n_samples = len(meta_df)
    n_folds = len(folds_data)

    oof_preds = np.full(n_samples, np.nan, dtype=np.float32)
    oof_thresholds = np.full(n_samples, td_fallback, dtype=np.float32)
    fold_records: List[Dict] = []
    fold_models = []
    best_iterations: List[int] = []
    importance_accum = np.zeros(len(feature_names), dtype=np.float64)

    modality_output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{modality}_{canonical_name}"

    log.info("-" * 75)
    log.info(
        f"▶ [Base Model: {modality.upper()} / {canonical_name.upper()}] "
        f"LOSO-CV 開始 (Folds={n_folds}, Features={len(feature_names)}d, EarlyStopping={stop_rounds})"
    )
    start_time = time.time()

    for f_data in folds_data:
        X_tr_all = f_data.X_train_visual if modality == "visual" else f_data.X_train_vital
        X_te = f_data.X_test_visual if modality == "visual" else f_data.X_test_vital
        y_tr_all = f_data.y_soft_train
        w_tr_all = f_data.sample_weights_train

        if es_enabled and stop_rounds > 0 and f_data.inner_val_mask.any():
            tr_m = f_data.inner_tr_mask
            val_m = f_data.inner_val_mask
            X_tr, y_tr, w_tr = X_tr_all[tr_m], y_tr_all[tr_m], w_tr_all[tr_m]
            X_val, y_val, w_val = X_tr_all[val_m], y_tr_all[val_m], w_tr_all[val_m]
        else:
            X_tr, y_tr, w_tr = X_tr_all, y_tr_all, w_tr_all
            X_val, y_val, w_val = None, None, None

        reg = create_base_model(
            model_name=canonical_name,
            modality=modality,
            cfg=cfg,
            random_state=seed,
            early_stopping_rounds=stop_rounds,
        )
        reg.fit(
            X_train=X_tr,
            y_train=y_tr,
            sample_weight=w_tr,
            X_val=X_val,
            y_val=y_val,
            sample_weight_val=w_val,
            early_stopping_rounds=stop_rounds,
            feature_names=feature_names,
        )

        y_pred_test = reg.predict(X_te).astype(np.float32)
        oof_preds[f_data.test_idx] = y_pred_test

        # Fold 内の検証データ (Val) で最適カットオフ閾値 th* を決定（Testリーク完全防止）
        th_fold = td_fallback
        if td_enabled:
            if X_val is not None and len(X_val) > 0:
                y_val_true_hard = labels[f_data.train_idx[f_data.inner_val_mask]]
                if len(np.unique((y_val_true_hard >= 0.5).astype(int))) >= 2:
                    y_val_pred = reg.predict(X_val).astype(np.float32)
                else:
                    y_val_true_hard = labels[f_data.train_idx]
                    y_val_pred = reg.predict(X_tr_all).astype(np.float32)
            else:
                y_val_true_hard = labels[f_data.train_idx]
                y_val_pred = reg.predict(X_tr_all).astype(np.float32)

            th_opt_res = find_optimal_threshold(
                y_val_true=y_val_true_hard,
                y_val_pred=y_val_pred,
                criterion=td_criterion,
                pr_balance_mode=td_pr_mode,
                search_range=td_range,
                search_step=td_step,
                fallback_threshold=td_fallback,
            )
            th_fold = float(th_opt_res["best_threshold"])

        oof_thresholds[f_data.test_idx] = th_fold

        best_it = reg.best_iteration_ or 0
        best_iterations.append(int(best_it))
        fold_models.append(reg)

        imp_df = reg.get_feature_importance(feature_names=feature_names)
        if not imp_df.empty:
            imp_map = dict(zip(imp_df["feature"], imp_df["importance"]))
            for idx_f, fname in enumerate(feature_names):
                importance_accum[idx_f] += float(imp_map.get(fname, 0.0))

        f_metrics = compute_fold_metrics(
            fold=f_data.fold,
            test_user=f_data.test_user,
            y_true=f_data.y_true_test,
            y_pred=y_pred_test,
            threshold=th_fold,
        )
        f_metrics["optimal_threshold"] = round(th_fold, 4)
        f_metrics["best_iteration"] = int(best_it)
        fold_records.append(f_metrics)

        auc_str = f"{f_metrics['roc_auc']:.3f}" if pd.notna(f_metrics["roc_auc"]) else "N/A"
        pr_str = f"{f_metrics['pr_auc']:.3f}" if pd.notna(f_metrics["pr_auc"]) else "N/A"
        r_str = f"{f_metrics['pearson_r']:.3f}" if pd.notna(f_metrics["pearson_r"]) else "N/A"
        log.info(
            f"  Fold [{f_data.fold:2d}/{n_folds:2d}] User: {f_data.test_user:<13} | "
            f"BestIter: {best_it:3d} | th*: {th_fold:.3f} | MAE: {f_metrics['mae']:.4f} | r: {r_str} | "
            f"AUC: {auc_str} | PR-AUC: {pr_str}"
        )

    elapsed = time.time() - start_time
    avg_best_iter = float(np.mean(best_iterations)) if best_iterations else 0.0

    # 全体 OOF 指標と Fold Mean ± Std の集計
    eval_res = compute_oof_and_fold_summary(
        y_true_all=labels,
        y_pred_all=oof_preds,
        fold_records=fold_records,
        threshold=oof_thresholds if td_enabled else td_fallback,
    )
    fold_df = eval_res["fold_df"]
    stats_dict = eval_res["stats_dict"]

    log.info(
        f"✅ [{modality.upper()} / {canonical_name.upper()}] 完了 ({elapsed:.2f}s) | "
        f"ROC-AUC={eval_res['roc_auc']:.4f} | PR-AUC={eval_res['pr_auc']:.4f} | "
        f"Pearson r={eval_res['pearson_r']:.4f} | MAE={eval_res['mae']:.4f} | AvgBestIter={avg_best_iter:.1f}"
    )

    # 1. Fold 別詳細 CSV 保存
    fold_csv = modality_output_dir / f"loso_{tag}_folds.csv"
    fold_df.to_csv(fold_csv, index=False)

    # 2. OOF 予測値 CSV 保存 (メタモデル学習でも直接再利用可能)
    oof_df = meta_df.copy()
    oof_df["y_gmm_soft_oof"] = oof_gmm_soft
    oof_df["sample_weight_oof"] = oof_sample_weight
    oof_df[f"y_pred_{modality}_oof"] = oof_preds
    oof_df["y_pred_oof"] = oof_preds
    oof_csv = modality_output_dir / f"loso_{tag}_oof_predictions.csv"
    oof_df.to_csv(oof_csv, index=False)

    # 3. 特徴量重要度 CSV 保存
    mean_importance = importance_accum / max(1, n_folds)
    imp_out_df = (
        pd.DataFrame({"feature": feature_names, "importance": mean_importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp_csv = modality_output_dir / f"{canonical_name}_feature_importance.csv"
    imp_out_df.to_csv(imp_csv, index=False)

    # 4. 学習済みモデルの保存 (全Foldモデル群 + 代表モデル + メタ情報)
    model_path = modality_output_dir / f"{canonical_name}_model.joblib"
    if save_models:
        model_bundle = {
            "model_name": canonical_name,
            "modality": modality,
            "window_size": window_size,
            "feature_names": feature_names,
            "early_stopping_rounds": stop_rounds,
            "avg_best_iteration": avg_best_iter,
            "best_iterations_per_fold": best_iterations,
            "fold_models": [
                {
                    "fold": f_data.fold,
                    "test_user": f_data.test_user,
                    "best_iteration": m.best_iteration_,
                    "estimator": m.model,
                }
                for f_data, m in zip(folds_data, fold_models)
            ],
            "metrics": {
                "roc_auc": eval_res["roc_auc"],
                "pr_auc": eval_res["pr_auc"],
                "pearson_r": eval_res["pearson_r"],
                "spearman_rho": eval_res["spearman_rho"],
                "mae": eval_res["mae"],
                "rmse": eval_res["rmse"],
                "r2": eval_res["r2"],
                "accuracy": eval_res["accuracy"],
                "balanced_acc": eval_res["balanced_acc"],
                "f1_macro": eval_res["f1_macro"],
            },
        }
        joblib.dump(model_bundle, model_path)
        log.info(f"💾 学習済みベースモデルを保存しました: {model_path}")

    # 5. 可視化プロット保存
    plot_path = None
    if save_plots:
        plot_path = generate_visualizations(
            output_dir=modality_output_dir,
            y_true=labels,
            y_pred=oof_preds,
            fold_df=fold_df,
            model_name=tag,
            mae=eval_res["mae"],
            rmse=eval_res["rmse"],
            r2=eval_res["r2"],
            r_val=eval_res["pearson_r"],
            acc=eval_res["accuracy"],
            balanced_acc=eval_res["balanced_acc"],
            auc=eval_res["roc_auc"],
            pr_auc=eval_res["pr_auc"],
            log=log,
        )

    # 6. 個別 Markdown レポート保存
    report_path = save_markdown_report(
        output_dir=modality_output_dir,
        window_size=window_size,
        model_name=tag,
        n_users=n_folds,
        mae=eval_res["mae"],
        rmse=eval_res["rmse"],
        r2=eval_res["r2"],
        r_val=eval_res["pearson_r"],
        rho_val=eval_res["spearman_rho"],
        acc=eval_res["accuracy"],
        balanced_acc=eval_res["balanced_acc"],
        auc=eval_res["roc_auc"],
        pr_auc=eval_res["pr_auc"],
        f1_macro=eval_res["f1_macro"],
        stats_dict=stats_dict,
        fold_df=fold_df,
        log=log,
    )

    eval_res.update({
        "model_name": canonical_name,
        "modality": modality,
        "avg_best_iteration": avg_best_iter,
        "oof_preds": oof_preds,
        "oof_csv": str(oof_csv),
        "model_path": str(model_path) if save_models else None,
        "plot_path": str(plot_path) if plot_path else None,
        "report_path": str(report_path),
    })
    return eval_res
