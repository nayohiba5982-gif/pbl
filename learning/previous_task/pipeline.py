"""
前段タスク (TS2Vec 埋め込み検証 ➔ PLS エルボー分析 ➔ PLS 潜在特徴量抽出 ➔ GMM 直前の被験者別 Zスコア標準化) パイプライン
(learning/previous_task/pipeline.py)

【サポートステージ】
- elbow: Train データのみによる PLS 目的変数説明分散比 (累積 R²) エルボー分析
- pls: Train データのみによる PLS 学習 & 潜在特徴量抽出・保存
- standardize (norm): GMM 入力直前の被験者別 (Subject-wise) Zスコア標準化
- all: 上記全工程を順次一括実行
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time

_CURRENT_DIR = Path(__file__).resolve().parent
_LEARNING_DIR = _CURRENT_DIR.parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))
if str(_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(_LEARNING_DIR))

from config_loader import load_config, log
try:
    from .task import PreviousTask
    from .utils import normalize_window_list
except (ImportError, ValueError):
    from task import PreviousTask
    from utils import normalize_window_list


def main():
    parser = argparse.ArgumentParser(
        description="PreviousTask (PLS エルボー分析・PLS 特徴量抽出・被験者別 Zスコア標準化) パイプライン"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="設定YAMLファイル名 (デフォルト: config.yaml)",
    )
    parser.add_argument(
        "--stage",
        nargs="+",
        choices=["all", "elbow", "pls", "standardize", "norm"],
        default=["all"],
        help="実行ステージの指定 (デフォルト: all)",
    )
    parser.add_argument(
        "--window-size",
        "--window-sizes",
        "-w",
        nargs="+",
        default=None,
        help="ウィンドウサイズの指定 (未指定時は config.yaml の previous_task.selected_window を使用)",
    )
    parser.add_argument(
        "--n-components-visual",
        "--n-vis",
        type=int,
        default=None,
        help="視覚ストリーム (Visual) の PLS 成分数 (未指定時は config.yaml の設定値)",
    )
    parser.add_argument(
        "--n-components-vital",
        "--n-bio",
        type=int,
        default=None,
        help="生体ストリーム (Vital) の PLS 成分数 (未指定時は config.yaml の設定値)",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)

    task = PreviousTask(
        cfg,
        window_size=args.window_size,
        n_components_visual=args.n_components_visual,
        n_components_vital=args.n_components_vital,
    )

    log.info("=" * 70)
    log.info(f" 🚀 [PreviousTask Pipeline] 開始 [対象ステージ: {', '.join(args.stage)}]")
    log.info(f" ⏱️ 対象ウィンドウサイズ: {task.target_windows}")
    log.info(f" 📐 PLS 成分数: Visual={task.n_components_visual}d, Vital={task.n_components_vital}d")
    log.info("=" * 70)

    t0 = time.time()
    task.run(stages=set(args.stage))
    elapsed = time.time() - t0

    log.info("=" * 70)
    log.info(f" 🎉 [PreviousTask Pipeline] 完了 (総所要時間: {elapsed:.2f} 秒)")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
