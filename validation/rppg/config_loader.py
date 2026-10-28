"""
validation/rppg/config_loader.py
後方互換用プロキシモジュール。validation/config_loader.py を再エクスポートします。
"""

from pathlib import Path
import sys

_VALIDATION_DIR = Path(__file__).resolve().parent.parent
if str(_VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VALIDATION_DIR))

from config_loader import (  # noqa: F401
    BASE_DIR,
    PROJECT_DIR,
    RPPG_DIR,
    LM_DIR,
    PREPROCESS_DIR,
    VALIDATION_DIR,
    load_config,
    log,
)

__all__ = [
    "BASE_DIR",
    "PROJECT_DIR",
    "RPPG_DIR",
    "LM_DIR",
    "PREPROCESS_DIR",
    "VALIDATION_DIR",
    "load_config",
    "log",
]
