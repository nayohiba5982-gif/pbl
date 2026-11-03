"""
閾値決定（Threshold Decision）統合パイプライン
(learning/downstream_task/threshold_decision/pipeline.py)

LOSO-CV の各 Fold 内で検証データ（Val）からデータリークなしで最適カットオフ閾値 th*_k を探索し、
Base Model (Visual / Vital) および Meta Model の二値判定精度（Balanced Accuracy, F1-Macro, PRバランス）を
最適化・比較評価・レポート出力します。
"""

from __future__ import annotations
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

_CURRENT_DIR = Path(__file__).resolve().parent
_DOWNSTREAM_DIR = _CURRENT_DIR.parent
_LEARNING_ROOT = _DOWNSTREAM_DIR.parent
for _p in (_CURRENT_DIR, _DOWNSTREAM_DIR, _LEARNING_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from config_loader import load_config, log
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("threshold_decision")

    def load_config(cfg_file: str = "config.yaml") -> dict:
        return {}

try:
    from .criteria import normalize_criterion_name
    from .reporter import (
        generate_threshold_visualizations,
        save_single_threshold_report,
        save_threshold_decision_summary_report,
    )
    from .runner import run_loso_threshold_tuning
except (ImportError, ValueError):
    from downstream_task.threshold_decision.criteria import normalize_criterion_name
    from downstream_task.threshold_decision.reporter import (
        generate_threshold_visualizations,
        save_single_threshold_report,
        save_threshold_decision_summary_report,
    )
    from downstream_task.threshold_decision.runner import run_loso_threshold_tuning


THREE_MAJOR_CRITERIA = ["youden_j", "f1_macro", "pr_balance"]


class ThresholdDecisionPipeline:
    """
    Step 7-3: 検証データ（Val）に基づく LOSO-CV リークフリー閾値決定パイプライン
    """

    def __init__(self, cfg: Optional[dict] = None):
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.paths = cfg.get("paths", {})
        self.dataset_cfg = cfg.get("dataset", {})
        self.downstream_cfg = cfg.get("downstream_task", {})
        self.loso_cfg = cfg.get("loso_cv", {})
        self.td_cfg = self.downstream_cfg.get("threshold_decision", {})

        self.window_size = self.dataset_cfg.get("window_size", "10s")
        self.learning_root = _LEARNING_ROOT

        model_root = Path(self.paths.get("model_dir", "models"))
        if not model_root.is_absolute():
            model_root = self.learning_root / model_root

        self.window_model_dir = model_root / self.window_size
        self.base_model_dir = self.window_model_dir / "base_model"
        self.meta_model_dir = self.window_model_dir / "meta_model"
        self.output_dir = self.window_model_dir / "threshold_decision"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _collect_oof_targets(
        self,
        target_scope: str = "both",
        models: Optional[Sequence[str]] = None,
        modalities: Optional[Sequence[str]] = None,
    ) -> List[Dict]:
        """評価対象の OOF 予測 CSV ファイル一覧を収集します。"""
        scope = str(target_scope or "both").strip().lower()
        targets: List[Dict] = []

        # 1. Meta Model OOF CSV
        if scope in ("both", "all", "meta", "meta_model", "meta_models"):
            if self.meta_model_dir.exists():
                for csv_path in sorted(self.meta_model_dir.glob("loso_meta_*_oof_predictions.csv")):
                    m_name = csv_path.name.replace("loso_meta_", "").replace("_oof_predictions.csv", "")
                    if models and m_name not in models and f"meta_{m_name}" not in models:
                        continue
                    targets.append({
                        "scope": "meta_model",
                        "modality": "meta",
                        "model": m_name,
                        "tag": f"meta_{m_name}",
                        "oof_csv": csv_path,
                    })

        # 2. Base Model OOF CSV (visual / vital)
        if scope in ("both", "all", "base", "base_model", "base_models"):
            target_mods = list(modalities) if modalities else ["visual", "vital"]
            for mod in target_mods:
                mod_dir = self.base_model_dir / mod
                if not mod_dir.exists():
                    continue
                for csv_path in sorted(mod_dir.glob(f"loso_{mod}_*_oof_predictions.csv")):
                    m_name = csv_path.name.replace(f"loso_{mod}_", "").replace("_oof_predictions.csv", "")
                    if models and m_name not in models and f"{mod}_{m_name}" not in models:
                        continue
                    targets.append({
                        "scope": f"base_{mod}",
                        "modality": mod,
                        "model": m_name,
                        "tag": f"{mod}_{m_name}",
                        "oof_csv": csv_path,
                    })

        return targets

    def run(
        self,
        criterion: Optional[str] = None,
        target_scope: Optional[str] = None,
        models: Optional[Sequence[str]] = None,
        modalities: Optional[Sequence[str]] = None,
        compare_all_criteria: Optional[bool] = None,
        val_split_mode: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        指定されたモデル群・基準に対して LOSO-CV リークフリー閾値決定を実行し、
        比較 CSV・可視化プロット・Markdown レポートを保存します。
        """
        default_crit = normalize_criterion_name(
            criterion or self.td_cfg.get("default_criterion", "youden_j")
        )
        scope = target_scope or self.td_cfg.get("target_scope", "both")
        do_compare_all = (
            compare_all_criteria
            if compare_all_criteria is not None
            else bool(self.td_cfg.get("compare_all_criteria", True))
        )
        pr_balance_mode = str(self.td_cfg.get("pr_balance_mode", "min_diff"))
        search_range = list(self.td_cfg.get("search_range", [0.05, 0.95]))
        search_step = float(self.td_cfg.get("search_step", 0.005))
        use_exact = bool(self.td_cfg.get("use_exact_curve_points", True))
        v_mode = str(val_split_mode or self.td_cfg.get("val_split_mode", "oof_non_test"))
        val_ratio = float(self.td_cfg.get("val_ratio", 0.2))
        group_split = bool(self.td_cfg.get("group_split", True))
        fallback_th = float(self.td_cfg.get("fallback_threshold", 0.5))
        save_plots = bool(self.td_cfg.get("save_plots", self.downstream_cfg.get("save_plots", True)))
        seed = int(self.loso_cfg.get("random_state", self.downstream_cfg.get("random_state", 42)))

        criteria_to_run = list(THREE_MAJOR_CRITERIA) if do_compare_all else [default_crit]
        if default_crit not in criteria_to_run:
            criteria_to_run.insert(0, default_crit)

        targets = self._collect_oof_targets(
            target_scope=scope,
            models=models,
            modalities=modalities,
        )
        if not targets:
            log.warning(
                f"⚠️ [{self.window_size}] 閾値決定対象の OOF 予測 CSV が見つかりません "
                f"(検索先: {self.meta_model_dir}, {self.base_model_dir})。"
            )
            return pd.DataFrame()

        log.info("=" * 78)
        log.info(f"🎯 [Step 7-3: Threshold Decision パイプライン開始] [Window: {self.window_size}]")
        log.info(f"   デフォルト基準 (Primary) : {default_crit}")
        log.info(f"   評価基準一覧             : {criteria_to_run}")
        log.info(f"   検証データ抽出モード     : {v_mode} (Test被験者リーク完全防止)")
        log.info(f"   対象モデル数             : {len(targets)}")
        log.info(f"   保存先ディレクトリ       : {self.output_dir}")
        log.info("=" * 78)

        start_t = time.time()
        comparison_rows: List[Dict] = []

        for item in targets:
            tag = item["tag"]
            oof_csv = item["oof_csv"]
            df_oof = pd.read_csv(oof_csv)

            subject_col = "username" if "username" in df_oof.columns else "subject_id"
            label_col = "label" if "label" in df_oof.columns else "label_mean"
            pred_col = "y_pred_oof"
            if pred_col not in df_oof.columns:
                cands = [c for c in df_oof.columns if c.startswith("y_pred_") and "binary" not in c]
                if not cands:
                    log.warning(f"⚠️ {oof_csv.name} に予測列が見つからないためスキップします。")
                    continue
                pred_col = cands[-1]

            subjects = df_oof[subject_col].astype(str).values
            y_true = df_oof[label_col].values.astype(np.float64)
            y_pred = df_oof[pred_col].values.astype(np.float64)

            df_tuned_out = df_oof.copy()
            primary_res = None

            for crit in criteria_to_run:
                res = run_loso_threshold_tuning(
                    y_true_all=y_true,
                    y_pred_all=y_pred,
                    subjects=subjects,
                    criterion=crit,
                    pr_balance_mode=pr_balance_mode,
                    search_range=search_range,
                    search_step=search_step,
                    use_exact_curve_points=use_exact,
                    val_split_mode=v_mode,
                    val_ratio=val_ratio,
                    group_split=group_split,
                    fallback_threshold=fallback_th,
                    seed=seed,
                )

                df_tuned_out[f"th_opt_{crit}"] = res["oof_thresholds"]
                df_tuned_out[f"y_pred_binary_{crit}"] = res["oof_binary_tuned"]

                if crit == default_crit:
                    primary_res = res
                    df_tuned_out["optimal_threshold"] = res["oof_thresholds"]
                    df_tuned_out["y_pred_binary_tuned"] = res["oof_binary_tuned"]

                # Fold 詳細 CSV 保存
                fold_csv = self.output_dir / f"loso_{tag}_threshold_{crit}_folds.csv"
                res["fold_df"].to_csv(fold_csv, index=False)

                # Primary 基準または compare_all_criteria 時にレポート・プロット保存
                if crit == default_crit or do_compare_all:
                    save_single_threshold_report(
                        output_dir=self.output_dir,
                        window_size=self.window_size,
                        model_tag=tag,
                        tuning_res=res,
                        log=log if crit == default_crit else None,
                    )
                    if save_plots and crit == default_crit:
                        generate_threshold_visualizations(
                            output_dir=self.output_dir,
                            model_tag=tag,
                            y_true=y_true,
                            y_pred=y_pred,
                            tuning_res=res,
                            log=log,
                        )

                base_m = res["baseline_oof_detail"]
                tuned_m = res["tuned_oof_detail"]
                imp = res["improvement"]

                comparison_rows.append({
                    "scope": item["scope"],
                    "modality": item["modality"],
                    "model": item["model"],
                    "tag": tag,
                    "criterion": crit,
                    "mean_th": round(res["mean_threshold"], 4),
                    "std_th": round(res["std_threshold"], 4),
                    "median_th": round(res["median_threshold"], 4),
                    "balanced_acc_05": round(base_m["balanced_acc"], 4),
                    "balanced_acc_tuned": round(tuned_m["balanced_acc"], 4),
                    "delta_balanced_acc": round(imp["oof_balanced_acc_diff"], 4),
                    "f1_macro_05": round(base_m["f1_macro"], 4),
                    "f1_macro_tuned": round(tuned_m["f1_macro"], 4),
                    "delta_f1_macro": round(imp["oof_f1_macro_diff"], 4),
                    "f1_neg_05": round(base_m["f1_neg"], 4),
                    "f1_neg_tuned": round(tuned_m["f1_neg"], 4),
                    "delta_f1_neg": round(imp["oof_f1_neg_diff"], 4),
                    "specificity_05": round(base_m["specificity"], 4),
                    "specificity_tuned": round(tuned_m["specificity"], 4),
                    "recall_tuned": round(tuned_m["recall"], 4),
                    "precision_tuned": round(tuned_m["precision"], 4),
                    "accuracy_05": round(base_m["accuracy"], 4),
                    "accuracy_tuned": round(tuned_m["accuracy"], 4),
                    "roc_auc": round(tuned_m["roc_auc"], 4) if pd.notna(tuned_m["roc_auc"]) else np.nan,
                    "pr_auc": round(tuned_m["pr_auc"], 4) if pd.notna(tuned_m["pr_auc"]) else np.nan,
                })

            if primary_res is not None:
                b_m = primary_res["baseline_oof_detail"]
                t_m = primary_res["tuned_oof_detail"]
                log.info(
                    f"✅ [{tag:<26}] ({default_crit}) "
                    f"th*={primary_res['mean_threshold']:.3f}±{primary_res['std_threshold']:.3f} | "
                    f"BalAcc: {b_m['balanced_acc']:.4f} ➔ {t_m['balanced_acc']:.4f} ({t_m['balanced_acc']-b_m['balanced_acc']:+.4f}) | "
                    f"F1-Macro: {b_m['f1_macro']:.4f} ➔ {t_m['f1_macro']:.4f} ({t_m['f1_macro']-b_m['f1_macro']:+.4f})"
                )

            # Tuned OOF 予測 CSV 保存
            tuned_oof_path = self.output_dir / f"loso_{tag}_oof_predictions_tuned.csv"
            df_tuned_out.to_csv(tuned_oof_path, index=False)

        df_comp = pd.DataFrame(comparison_rows)
        if not df_comp.empty:
            comp_csv = self.output_dir / "threshold_criteria_comparison.csv"
            df_comp.to_csv(comp_csv, index=False)
            save_threshold_decision_summary_report(
                output_dir=self.output_dir,
                window_size=self.window_size,
                default_criterion=default_crit,
                comparison_df=df_comp,
                log=log,
            )

        # final_inference_predictions.csv がある場合は最適閾値による二値判定列も追加更新
        final_csv = self.meta_model_dir / "final_inference_predictions.csv"
        if final_csv.exists():
            try:
                df_final = pd.read_csv(final_csv)
                if "final_prediction" in df_final.columns:
                    subj_col = "username" if "username" in df_final.columns else "subject_id"
                    lbl_col = "label" if "label" in df_final.columns else "label_mean"
                    fin_res = run_loso_threshold_tuning(
                        y_true_all=df_final[lbl_col].values,
                        y_pred_all=df_final["final_prediction"].values,
                        subjects=df_final[subj_col].astype(str).values,
                        criterion=default_crit,
                        pr_balance_mode=pr_balance_mode,
                        search_range=search_range,
                        search_step=search_step,
                        use_exact_curve_points=use_exact,
                        val_split_mode=v_mode,
                        val_ratio=val_ratio,
                        group_split=group_split,
                        fallback_threshold=fallback_th,
                        seed=seed,
                    )
                    df_final["optimal_threshold"] = fin_res["oof_thresholds"]
                    df_final["final_binary_prediction_tuned"] = fin_res["oof_binary_tuned"]
                    df_final.to_csv(final_csv, index=False)
                    tuned_final_copy = self.output_dir / "final_inference_predictions_tuned.csv"
                    df_final.to_csv(tuned_final_copy, index=False)
                    log.info(f"💾 最終推論 CSV に最適閾値二値化列を付与しました: {tuned_final_copy}")
            except Exception as e:
                log.warning(f"⚠️ final_inference_predictions.csv の更新中にエラーが発生しました: {e}")

        log.info(f"🎉 [{self.window_size}] 閾値決定パイプライン完了 ({time.time() - start_t:.2f}s)")
        return df_comp
