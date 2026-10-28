"""
メタモデル (Visual + Vital 統合) 単体実行 CLI エントリーポイント
(learning/downstream_task/meta_model/main.py)
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
    log = logging.getLogger("meta_model")

    def load_config(config_filename: str = "config.yaml") -> dict:
        return {}

try:
    from .pipeline import MetaModelPipeline
    from .models import DEFAULT_META_MODELS
except (ImportError, ValueError):
    from downstream_task.meta_model.pipeline import MetaModelPipeline
    from downstream_task.meta_model.models import DEFAULT_META_MODELS


def main():
    parser = argparse.ArgumentParser(
        description="Step 7-2: メタモデル (LogisticRegression, Ridge, Lasso, ElasticNet) LOSO-CV 学習・評価・最終推論"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="設定YAMLファイル名")
    parser.add_argument(
        "--window-size", "--window-sizes", "-w",
        nargs="+",
        default=None,
        help="ウィンドウサイズ指定 (例: 10s / 5s 10s 15s / all)"
    )
    parser.add_argument(
        "--visual-model",
        type=str,
        default=None,
        help="Visual 側に使用するベースモデルの上書き指定 (未指定時は config.yaml を参照)"
    )
    parser.add_argument(
        "--vital-model",
        type=str,
        default=None,
        help="Vital 側に使用するベースモデルの上書き指定 (未指定時は config.yaml を参照)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"学習対象メタモデル一覧 (デフォルト: {DEFAULT_META_MODELS})"
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
        pipeline = MetaModelPipeline(cfg)
        pipeline.run(
            models=args.models,
            visual_model=args.visual_model,
            vital_model=args.vital_model,
        )

    log.info(f"🎉 全ウィンドウ ({target_windows}) の Meta Model 学習・評価が完了しました ({time.time() - total_start:.2f}s)")


if __name__ == "__main__":
    main()
