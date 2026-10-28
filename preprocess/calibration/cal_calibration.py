"""
cal_calibration.py (後方互換性エイリアス)

本モジュールは `lm_calibration.py` のエイリアスです。
既存コードとの互換性維持のため、`lm_calibration` の全シンボルを再エクスポートします。
"""

try:
    from .lm_calibration import *
    from .lm_calibration import (
        load_calibration_master,
        calibrate_landmarks,
        apply_calibration,
    )
except ImportError:
    from lm_calibration import *
    from lm_calibration import (
        load_calibration_master,
        calibrate_landmarks,
        apply_calibration,
    )
