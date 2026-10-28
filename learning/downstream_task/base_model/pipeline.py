"""
ベースモデル (Visual / Vital) LOSO-CV 学習・評価・保存統括パイプライン
(learning/downstream_task/base_model/pipeline.py)
"""

from __future__ import annotations
import copy
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union
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
    log = logging.getLogger("base_model")

    def load_config(cfg_file: str = "config.yaml") -> dict:
        return {}

try:
    from ..data_loader import load_embeddings_and_meta, load_explicit_features
    from ..metrics import build_summary_row
    from .models import DEFAULT_BASE_MODELS, normalize_base_model_name
    from .reporter import (
        plot_dual_modality_comparison,
        plot_model_comparison,
        save_modality_comparison_report,
        save_overall_base_model_report,
    )
    from .runner import prepare_loso_folds, run_single_base_model_loso
except (ImportError, ValueError):
    from downstream_task.data_loader import load_embeddings_and_meta, load_explicit_features
    from downstream_task.metrics import build_summary_row
    from downstream_task.base_model.models import DEFAULT_BASE_MODELS, normalize_base_model_name
    from downstream_task.base_model.reporter import (
        plot_dual_modality_comparison,
        plot_model_comparison,
        save_modality_comparison_report,
        save_overall_base_model_report,
    )
    from downstream_task.base_model.runner import prepare_loso_folds, run_single_base_model_loso


class BaseModelPipeline:
    """
    Visual / Vital モダリティ別ベースモデル (RandomForest, XGBoost, LightGBM, CatBoost, AdaBoost)
    を Early Stopping 付き LOSO-CV で学習・評価・保存し、report.md を出力するパイプライン。
    """

    def __init__(self, cfg: Optional[dict] = None):
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.paths = cfg.get("paths", {})
        self.dataset_cfg = cfg.get("dataset", {})
        self.clustering_cfg = cfg.get("clustering", {})
        self.gmm_cfg = self.clustering_cfg.get("gmm", {})
        self.downstream_cfg = cfg.get("downstream_task", {})
        self.base_model_cfg = self.downstream_cfg.get("base_model", {})
        self.loso_cfg = cfg.get("loso_cv", {})

        self.window_size = self.dataset_cfg.get("window_size", "10s")
        self.learning_root = _LEARNING_ROOT

        data_root = Path(self.paths.get("data_root", "data"))
        if not data_root.is_absolute():
            data_root = self.learning_root / data_root
        self.data_dir = data_root / self.window_size
        self.data_dir.mkdir(parents=True, exist_ok=True)

        input_data_root = Path(self.paths.get("input_data_root", "../preprocess/data/windowed"))
        if not input_data_root.is_absolute():
            input_data_root = self.learning_root / input_data_root
        self.input_data_dir = input_data_root / self.window_size

        model_root = Path(self.paths.get("model_dir", "models"))
        if not model_root.is_absolute():
            model_root = self.learning_root / model_root
        self.output_dir = model_root / self.window_size / "base_model"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.default_cov = (
            self.loso_cfg.get("covariance_type")
            or self.gmm_cfg.get("covariance_type", "full")
        )

    def run(
        self,
        modalities: Optional[Sequence[str]] = None,
        models: Optional[Sequence[str]] = None,
        weights: Optional[Union[Tuple[float, float], List[float]]] = None,
        covariance_type: Optional[str] = None,
        apply_subject_norm: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        指定されたモダリティ (visual, vital) とベースモデル群に対して
        一括で LOSO-CV 学習・評価・保存・レポート生成を実行します。
        """
        target_modalities = list(
            modalities
            or self.base_model_cfg.get("modalities")
            or ["visual", "vital"]
        )
        raw_models = list(
            models
            or self.base_model_cfg.get("models")
            or DEFAULT_BASE_MODELS
        )
        target_models = [normalize_base_model_name(m) for m in raw_models]

        cov_type = covariance_type or self.default_cov
        vis_pls = int(self.loso_cfg.get("n_vis_pls", 3))
        bio_pls = int(self.loso_cfg.get("n_bio_pls", 3))
        subj_norm = (
            apply_subject_norm
            if apply_subject_norm is not None
            else bool(self.loso_cfg.get("apply_subject_norm", True))
        )

        if weights is not None and len(weights) >= 2:
            w_tuple = (float(weights[0]), float(weights[1]))
        else:
            default_weights = self.loso_cfg.get("weights", [1.0, 0.0])
            w_tuple = (float(default_weights[0]), float(default_weights[1]))

        es_cfg = self.base_model_cfg.get("early_stopping", {})
        es_rounds = (
            int(es_cfg.get("stopping_rounds", self.downstream_cfg.get("early_stopping_rounds", 30)))
            if es_cfg.get("enabled", True)
            else 0
        )

        log.info("=" * 78)
        log.info(f"🚀 [Step 7-1: Base Model LOSO-CV パイプライン開始] [Window: {self.window_size}]")
        log.info(f"   対象モダリティ : {target_modalities}")
        log.info(f"   対象回帰モデル : {target_models}")
        log.info(f"   Early Stopping : rounds={es_rounds} (val_ratio={es_cfg.get('val_ratio', 0.2)})")
        log.info(f"   保存先ディレクトリ: {self.output_dir}")
        log.info("=" * 78)

        start_t = time.time()

        # 1. 埋め込み・メタデータ・明示的物理統計量の読み込み
        emb_vis, emb_bio, meta_df = load_embeddings_and_meta(
            data_dir=self.data_dir,
            input_data_dir=self.input_data_dir,
            learning_root=self.learning_root,
            window_size=self.window_size,
            log=log,
        )
        explicit_feats, explicit_names = load_explicit_features(
            meta_df=meta_df,
            input_data_dir=self.input_data_dir,
            data_dir=self.data_dir,
            window_size=self.window_size,
            log=log,
        )

        # 2. 全 Fold の PLS・被験者標準化・GMM ソフトラベル・内部分割を一括準備
        folds_data, vis_feat_names, vital_feat_names, oof_gmm_soft, oof_sample_weight = prepare_loso_folds(
            emb_vis=emb_vis,
            emb_bio=emb_bio,
            meta_df=meta_df,
            explicit_feats=explicit_feats,
            explicit_names=explicit_names,
            cfg=self.cfg,
            log=log,
            n_vis_pls=vis_pls,
            n_bio_pls=bio_pls,
            cov_type=cov_type,
            weights=w_tuple,
            apply_subject_norm=subj_norm,
        )

        modality_features = {
            "visual": vis_feat_names,
            "vital": vital_feat_names,
        }
        log.info(
            f"✓ モダリティ特徴量分離完了: Visual={len(vis_feat_names)}d ({vis_feat_names[:4]}...), "
            f"Vital={len(vital_feat_names)}d ({vital_feat_names})"
        )

        all_summary_rows: List[Dict] = []

        # 3. 各モダリティ × 各モデルの学習・評価
        for modality in target_modalities:
            mod_key = modality.lower().strip()
            mod_dir = self.output_dir / mod_key
            mod_dir.mkdir(parents=True, exist_ok=True)
            feat_names = modality_features[mod_key]

            log.info("=" * 78)
            log.info(f"🔹 モダリティ [{mod_key.upper()}] の全 {len(target_models)} モデル学習を開始します")
            log.info("=" * 78)

            mod_rows: List[Dict] = []
            for m_name in target_models:
                res = run_single_base_model_loso(
                    modality=mod_key,
                    model_name=m_name,
                    folds_data=folds_data,
                    feature_names=feat_names,
                    meta_df=meta_df,
                    oof_gmm_soft=oof_gmm_soft,
                    oof_sample_weight=oof_sample_weight,
                    window_size=self.window_size,
                    modality_output_dir=mod_dir,
                    cfg=self.cfg,
                    log=log,
                )
                row = build_summary_row(
                    model_name=res["model_name"],
                    eval_res=res,
                    extra_fields={
                        "modality": mod_key,
                        "avg_best_iteration": round(res["avg_best_iteration"], 1),
                        "model_path": res["model_path"],
                    },
                )
                mod_rows.append(row)
                all_summary_rows.append(row)

            df_mod = pd.DataFrame(mod_rows)
            mod_csv = mod_dir / f"{mod_key}_models_comparison.csv"
            df_mod.to_csv(mod_csv, index=False)

            if self.downstream_cfg.get("save_plots", True) and not df_mod.empty:
                plot_model_comparison(df_mod, mod_dir, log)

            save_modality_comparison_report(
                modality=mod_key,
                window_size=self.window_size,
                df_mod=df_mod,
                modality_dir=mod_dir,
                feature_names=feat_names,
                early_stopping_rounds=es_rounds,
                log=log,
            )

        # 4. Base Model 全体比較 CSV・プロット・総合 report.md 保存
        df_all = pd.DataFrame(all_summary_rows)
        all_csv = self.output_dir / "base_models_comparison.csv"
        df_all.to_csv(all_csv, index=False)
        log.info(f"💾 Base Model 全モダリティ比較 CSV を保存しました: {all_csv}")

        if self.downstream_cfg.get("save_plots", True) and not df_all.empty:
            plot_dual_modality_comparison(df_all, self.output_dir, log)

        save_overall_base_model_report(
            base_output_dir=self.output_dir,
            window_size=self.window_size,
            df_all=df_all,
            modality_features=modality_features,
            early_stopping_rounds=es_rounds,
            log=log,
        )

        elapsed_total = time.time() - start_t
        log.info("=" * 78)
        log.info(f"🎉 [Base Model Pipeline] 全処理完了 (所要時間: {elapsed_total:.2f} 秒)")
        for mod_k in target_modalities:
            sub = df_all[df_all["modality"] == mod_k]
            if not sub.empty:
                log.info(f"  【{mod_k.upper()} モダリティ結果一覧】")
                for _, r in sub.iterrows():
                    log.info(
                        f"    - {r['model']:<15}: "
                        f"ROC-AUC={r['roc_auc']:.4f} (±{r['roc_auc_std']:.3f}) | "
                        f"PR-AUC={r['pr_auc']:.4f} | "
                        f"Pearson r={r['pearson_r']:.4f} | "
                        f"MAE={r['mae']:.4f} | "
                        f"BestIter={r['avg_best_iteration']:.1f}"
                    )
        log.info("=" * 78)

        return df_all
