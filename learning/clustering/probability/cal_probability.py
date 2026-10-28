r"""
サンプルごとの確率ラベル（期待値 y ∈ [0, 1]）算出エントリーポイント (CLI & Runner)

GMM の predict_proba で得られる各クラスタの所属確率 P(Cluster=k | x) に、
決定した重み w_k を掛け合わせた期待値（ソフトターゲット）を計算します:
    y_i = \sum_{k=0}^{K-1} w_k \cdot P(\text{Cluster}=k \mid x_i)
"""

import argparse
from pathlib import Path
import sys
import numpy as np

# パス解決（learning ディレクトリ直下を参照可能にする）
_CURRENT_DIR = Path(__file__).resolve().parent
_LEARNING_DIR = _CURRENT_DIR.parent
if str(_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(_LEARNING_DIR))

try:
    from config_loader import load_config, log
except ImportError:
    import logging
    log = logging.getLogger("probability_label")
    def load_config(config_filename: str = "config.yaml") -> dict:
        return {}

# 分割された各サブモジュールからインポート
from .core import compute_soft_labels, generate_default_weights
from .loader import find_latest_gmm_model, load_probabilities_for_split
from .plotter import plot_soft_label_distribution
from .calculator import ProbabilityLabelCalculator


def run_cal_probability(
    cfg: dict | None = None,
    weights: list[float] | np.ndarray | None = None,
) -> dict[str, dict]:
    """外部モジュールおよびパイプラインから呼び出し可能なヘルパー関数"""
    calculator = ProbabilityLabelCalculator(cfg=cfg, weights=weights)
    return calculator.run()


def main():
    parser = argparse.ArgumentParser(
        description="GMM 所属確率とクラスタ重みによるサンプル別確率ラベル（ソフトターゲット y ∈ [0, 1]）の算出"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="設定YAMLファイル名 (デフォルト: config.yaml)",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="各クラスタに対する重み w_k (例: --weights 1.0 0.5 0.0)",
    )
    parser.add_argument(
        "--window-size",
        type=str,
        default=None,
        help="ウィンドウサイズの上書き指定 (例: 5s, 10s, 15s)",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="y_soft",
        help="メタデータに追加する列名 (デフォルト: y_soft)",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.window_size:
        cfg["dataset"]["window_size"] = args.window_size

    calculator = ProbabilityLabelCalculator(cfg=cfg, weights=args.weights)
    if args.target_col:
        calculator.target_col = args.target_col

    calculator.run()


if __name__ == "__main__":
    main()
