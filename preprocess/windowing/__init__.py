"""
windowing パッケージ
スライディングウィンドウ分割およびラベル別トリミング処理を提供します。
"""
try:
    from .trim_margins import (
        DEFAULT_TRIM_MARGINS,
        normalize_trim_margins,
    )
    from .window_generator import (
        create_sliding_windows_for_segment,
        create_sliding_windows_by_label_from_df,
        create_sliding_windows_from_df,
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
