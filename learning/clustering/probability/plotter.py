"""
確率ラベル（期待値ソフトターゲット）分布可視化プロットモジュール (Plotter)
"""

from pathlib import Path
import sys

_LEARNING_DIR = Path(__file__).resolve().parent.parent
if str(_LEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(_LEARNING_DIR))

try:
    from config_loader import log
except ImportError:
    import logging
    log = logging.getLogger("probability_label.plotter")


def plot_soft_label_distribution(
    results: dict[str, dict], save_path: Path, window_size: str = "10s"
):
    """
    各 Split (train, val, test) におけるソフトラベル期待値の分布（ヒストグラム・密度）を描画・保存します。
    """
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        colors = {"train": "royalblue", "val": "forestgreen", "test": "crimson"}

        for s_name, res in results.items():
            y_val = res["y_soft"]
            plt.hist(
                y_val,
                bins=30,
                range=(0.0, 1.0),
                alpha=0.5,
                density=True,
                label=f"{s_name.capitalize()} (n={len(y_val)}, mean={y_val.mean():.2f})",
                color=colors.get(s_name, "gray"),
                edgecolor="black",
            )

        plt.title(f"Soft Target Distribution: y = Σ (w_k * P(Cluster=k)) [{window_size}]", fontsize=12)
        plt.xlabel("Continuous Concentration Expectation (y ∈ [0.0, 1.0])", fontsize=11)
        plt.ylabel("Density", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True)
        plt.tight_layout()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        log.info(f"📊 ソフトラベル分布プロットを保存しました: {save_path.name}")
    except Exception as e:
        log.warning(f"分布プロットの作成をスキップしました ({e})")
