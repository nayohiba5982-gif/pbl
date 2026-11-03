"""
閾値決定（Threshold Decision）単体実行 CLI エントリーポイント
(learning/downstream_task/threshold_decision/main.py)
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
    log = logging.getLogger("threshold_decision")

    def load_config(config_filename: str = "config.yaml") -> dict:
        return {}

try:
    from .criteria import SUPPORTED_CRITERIA
    from .pipeline import ThresholdDecisionPipeline
except (ImportError, ValueError):
    from downstream_task.threshold_decision.criteria import SUPPORTED_CRITERIA
    from downstream_task.threshold_decision.pipeline import ThresholdDecisionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Step 7-3: LOSO-CV リークフリー閾値決定 (Youden's J / F1-Macro / PR-Balance)"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="設定YAMLファイル名")
    parser.add_argument(
        "--window-size", "--window-sizes", "-w",
        nargs="+",
        default=None,
        help="ウィンドウサイズ指定 (例: 10s / 5s 10s 15s / all)"
    )
    parser.add_argument(
        "--criterion", "-c",
        type=str,
        default=None,
        help=f"閾値最適化基準 ({list(SUPPORTED_CRITERIA)})"
    )
    parser.add_argument(
        "--scope", "--target-scope",
        type=str,
        default=None,
        choices=["both", "meta_model", "base_model"],
        help="適用対象スコープ (meta_model / base_model / both)"
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=["visual", "vital"],
        default=None,
        help="Base Model 対象モダリティ"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="特定モデル名のみを対象にする場合に指定"
    )
    parser.add_argument(
        "--val-split-mode",
        type=str,
        default=None,
        choices=["oof_non_test", "group_val"],
        help="Fold内の検証データ選択方式 (oof_non_test または group_val)"
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
        pipeline = ThresholdDecisionPipeline(cfg)
        pipeline.run(
            criterion=args.criterion,
            target_scope=args.scope,
            models=args.models,
            modalities=args.modalities,
            val_split_mode=args.val_split_mode,
        )

    log.info(f"🎉 全対象ウィンドウ ({target_windows}) の閾値決定処理が完了しました ({time.time() - total_start:.2f}s)")


if __name__ == "__main__":
    main()
