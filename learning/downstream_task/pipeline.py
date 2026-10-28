"""
下流タスク 2段階 (Step 7-1: Base Model [Visual/Vital] ➔ Step 7-2: Meta Model) 統合パイプライン
(learning/downstream_task/pipeline.py)
"""

from __future__ import annotations
import argparse
import copy
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

_CURRENT_DIR = Path(__file__).resolve().parent
_LEARNING_ROOT = _CURRENT_DIR.parent
for _p in (_CURRENT_DIR, _LEARNING_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from config_loader import load_config, log
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("downstream_task")

    def load_config(cfg_file: str = "config.yaml") -> dict:
        return {}

try:
    from .base_model import BaseModelPipeline, DEFAULT_BASE_MODELS
    from .meta_model import MetaModelPipeline, DEFAULT_META_MODELS
except (ImportError, ValueError):
    from downstream_task.base_model import BaseModelPipeline, DEFAULT_BASE_MODELS
    from downstream_task.meta_model import MetaModelPipeline, DEFAULT_META_MODELS


DEFAULT_DOWNSTREAM_MODELS = DEFAULT_BASE_MODELS


class DownstreamTaskPipeline:
    """
    下流タスク LOSO-CV 2段階学習・推論パイプライン:
      1. BaseModelPipeline : visual, vital 各モダリティで 5モデル (RF, XGB, LGBM, CatBoost, AdaBoost) を
                             Early Stopping 付きで学習・評価 (report.md)・モデル保存
      2. MetaModelPipeline : config.yaml で指定された visual, vital モデルを用いて
                             4種のメタモデル (LogisticRegression, Ridge, Lasso, ElasticNet) を学習・評価・最終推論
    """

    def __init__(self, cfg: Optional[dict] = None):
        if cfg is None:
            cfg = load_config()
        self.cfg = cfg
        self.downstream_cfg = cfg.get("downstream_task", {})
        self.base_pipeline = BaseModelPipeline(cfg)
        self.meta_pipeline = MetaModelPipeline(cfg)

    def run_base_models(
        self,
        modalities: Optional[Sequence[str]] = None,
        models: Optional[Sequence[str]] = None,
        weights: Optional[Union[Tuple[float, float], List[float]]] = None,
        covariance_type: Optional[str] = None,
        apply_subject_norm: Optional[bool] = None,
    ):
        """Step 7-1: Visual / Vital 各モダリティのベースモデル群を学習・評価・保存"""
        return self.base_pipeline.run(
            modalities=modalities,
            models=models,
            weights=weights,
            covariance_type=covariance_type,
            apply_subject_norm=apply_subject_norm,
        )

    def run_meta_models(
        self,
        models: Optional[Sequence[str]] = None,
        visual_model: Optional[str] = None,
        vital_model: Optional[str] = None,
    ):
        """Step 7-2: 選択されたベースモデルを用いてメタモデル群を学習・評価・最終推論"""
        return self.meta_pipeline.run(
            models=models,
            visual_model=visual_model,
            vital_model=vital_model,
        )

    def run_downstream_task(
        self,
        stage: str = "all",
        modalities: Optional[Sequence[str]] = None,
        base_models: Optional[Sequence[str]] = None,
        meta_models: Optional[Sequence[str]] = None,
        visual_model: Optional[str] = None,
        vital_model: Optional[str] = None,
        weights: Optional[Union[Tuple[float, float], List[float]]] = None,
        covariance_type: Optional[str] = None,
        apply_subject_norm: Optional[bool] = None,
        models: Optional[Sequence[str]] = None,
        **kwargs,
    ) -> Dict:
        """
        指定ステージ ("base_model", "meta_model", または "all") に応じて実行します。
        """
        stage_norm = (stage or self.downstream_cfg.get("default_stage", "all")).lower().strip()
        results = {}

        if stage_norm in ("all", "base", "base_model", "base_models", "downstream_task", "loso_cv"):
            df_base = self.run_base_models(
                modalities=modalities,
                models=base_models or models,
                weights=weights,
                covariance_type=covariance_type,
                apply_subject_norm=apply_subject_norm,
            )
            results["base_model"] = df_base

        if stage_norm in ("all", "meta", "meta_model", "meta_models", "downstream_task", "loso_cv"):
            df_meta = self.run_meta_models(
                models=meta_models,
                visual_model=visual_model,
                vital_model=vital_model,
            )
            results["meta_model"] = df_meta

        return results

    run_all = run_downstream_task
    run_loso_regression = run_downstream_task


LOSORegressionPipeline = DownstreamTaskPipeline


def main():
    parser = argparse.ArgumentParser(
        description="下流タスク LOSO-CV 2段階パイプライン (Base Model [Visual/Vital] ➔ Meta Model)"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="設定YAMLファイル名")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "base_model", "base", "meta_model", "meta"],
        help="実行ステージ: base_model (Visual/Vitalモデル作成) / meta_model (メタモデル作成) / all (両方)"
    )
    parser.add_argument(
        "--window-size", "--window-sizes", "-w",
        nargs="+",
        default=None,
        help="ウィンドウサイズ指定 (例: 10s / 5s 10s 15s / all)"
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=["visual", "vital"],
        default=None,
        help="Base Model 対象モダリティ (visual / vital)"
    )
    parser.add_argument(
        "--base-models",
        nargs="+",
        default=None,
        help=f"対象ベースモデル (デフォルト: {DEFAULT_BASE_MODELS})"
    )
    parser.add_argument(
        "--meta-models",
        nargs="+",
        default=None,
        help=f"対象メタモデル (デフォルト: {DEFAULT_META_MODELS})"
    )
    parser.add_argument(
        "--visual-model",
        type=str,
        default=None,
        help="Meta Model で使用する Visual ベースモデル指定"
    )
    parser.add_argument(
        "--vital-model",
        type=str,
        default=None,
        help="Meta Model で使用する Vital ベースモデル指定"
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="ソフトラベル重み指定 (例: --weights 1.0 0.0)"
    )
    parser.add_argument(
        "--covariance-type", "--cov",
        type=str,
        default=None,
        choices=["full", "tied", "diag", "spherical"],
        help="GMM 共分散タイプ"
    )
    parser.add_argument(
        "--no-subj-norm",
        action="store_true",
        help="被験者別Zスコア標準化を無効化"
    )

    args = parser.parse_args()
    ALL_SUPPORTED_WINDOWS = ["5s", "10s", "15s"]
    base_cfg = load_config(args.config)

    if args.window_size:
        raw_list = [w.strip() for w in args.window_size]
        if "all" in [w.lower() for w in raw_list]:
            target_windows = ALL_SUPPORTED_WINDOWS
        else:
            target_windows = []
            for w in raw_list:
                wl = w.lower()
                if wl in ("5", "5s"):
                    target_windows.append("5s")
                elif wl in ("10", "10s"):
                    target_windows.append("10s")
                elif wl in ("15", "15s"):
                    target_windows.append("15s")
                else:
                    target_windows.append(w)
            target_windows = list(dict.fromkeys(target_windows))
    else:
        default_ws = base_cfg.get("dataset", {}).get("window_size", "10s")
        target_windows = [default_ws]

    total_start = time.time()
    for ws in target_windows:
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("dataset", {})["window_size"] = ws
        pipeline = DownstreamTaskPipeline(cfg)
        pipeline.run_downstream_task(
            stage=args.stage,
            modalities=args.modalities,
            base_models=args.base_models,
            meta_models=args.meta_models,
            visual_model=args.visual_model,
            vital_model=args.vital_model,
            weights=args.weights,
            covariance_type=args.covariance_type,
            apply_subject_norm=not args.no_subj_norm,
        )

    log.info(f"🎉 全ウィンドウ ({target_windows}) の下流タスクが完了しました ({time.time() - total_start:.2f}s)")


if __name__ == "__main__":
    main()
