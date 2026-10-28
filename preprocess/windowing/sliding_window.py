"""
【エイリアスモジュール (sliding_window.py)】
sliding_windows.py と同等のインターフェースを提供します。
"""
try:
    from .sliding_windows import *
    from .sliding_windows import __all__, _BASE_DIR, run_windowing, load_config
except ImportError:
    from sliding_windows import *
    from sliding_windows import __all__, _BASE_DIR, run_windowing, load_config

if __name__ == "__main__":
    cfg = load_config()
    run_windowing(cfg)
