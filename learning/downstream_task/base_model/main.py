"""
ベースモデル (Visual / Vital) 単体実行 CLI エントリーポイント
(learning/downstream_task/base_model/main.py)
"""

from __future__ import annotations
import argparse
import copy
from pathlib import Path
import sys
import time

_CURRENT_DIR = Path(__file__).resolve().parent
_DOWNSTREAM_DIR = _CURRENT_DIR.parent
_LEARNING_ROOT = _DOWNSTREAM_DIR.parent
for _p in (_CURRENT_DIR, _DOWNSTREAM_DIR, _LEARNING_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from config_loader import load_config, log
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("base_model")

    def load_config(config_filename: str = "config.yaml") -> dict:
        return {}

try:
    from .pipeline import BaseModelPipeline
    from .models import DEFAULT_BASE_MODELS
except (ImportError, ValueError):
    from downstream_task.base_model.pipeline import BaseModelPipeline
    from downstream_task.base_model.models import DEFAULT_BASE_MODELS


def main():
    parser = argparse.ArgumentParser(
        description="Step 7-1: Visual / Vital モダリティ別ベース回帰モデル (Early Stopping 対応) LOSO-CV 学習・評価"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="設定YAMLファイル名")
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
        help="学習対象モダリティ (デフォルト: visual vital 両方)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"学習対象ベースモデル (デフォルト: {DEFAULT_BASE_MODELS})"
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=None,
        help="Early Stopping ラウンド数の上書き指定"
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
        help="Fold 内 GMM の共分散タイプ"
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
        if args.early_stopping is not None:
            cfg.setdefault("downstream_task", {}).setdefault("base_model", {}).setdefault(
                "early_stopping", {}
            )["stopping_rounds"] = args.early_stopping

        pipeline = BaseModelPipeline(cfg)
        pipeline.run(
            modalities=args.modalities,
            models=args.models,
            weights=args.weights,
            covariance_type=args.covariance_type,
            apply_subject_norm=not args.no_subj_norm,
        )

    log.info(f"🎉 全ウィンドウ ({target_windows}) の Base Model 学習・評価が完了しました ({time.time() - total_start:.2f}s)")


if __name__ == "__main__":
    main()
