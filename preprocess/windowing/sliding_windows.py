"""
【後方互換性維持用 & CLIエントリーポイント】
各サブモジュールから主要関数・定数を再エクスポートします。
"""
from pathlib import Path
import sys

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

try:
    from config_loader import load_config
except ImportError:
    def load_config():
        return {}

try:
    from .trim_margins import (
        DEFAULT_TRIM_MARGINS,
        normalize_trim_margins,
    )
    from .window_generator import (
        create_sliding_windows_for_segment,
        create_sliding_windows_by_label_from_df,
        create_sliding_windows_from_df,
        create_sliding_windows_for_user_from_df,
    )
    from .batch_processor import (
        process_all_files_for_windowing,
        run_windowing,
    )
except ImportError:
    from trim_margins import (
        DEFAULT_TRIM_MARGINS,
        normalize_trim_margins,
    )
    from window_generator import (
        create_sliding_windows_for_segment,
        create_sliding_windows_by_label_from_df,
        create_sliding_windows_from_df,
        create_sliding_windows_for_user_from_df,
    )
    from batch_processor import (
        process_all_files_for_windowing,
        run_windowing,
    )

__all__ = [
    "DEFAULT_TRIM_MARGINS",
    "normalize_trim_margins",
    "create_sliding_windows_for_segment",
    "create_sliding_windows_by_label_from_df",
    "create_sliding_windows_from_df",
    "process_all_files_for_windowing",
    "run_windowing",
]

if __name__ == "__main__":
    cfg = load_config()
    run_windowing(cfg)