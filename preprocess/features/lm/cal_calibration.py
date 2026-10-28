"""
cal_calibration.py (後方互換性エイリアス)

本モジュールは `preprocess.calibration.lm_calibration` へ移行しました。
既存コードとの互換性維持のため、`lm_calibration` の全シンボルを再エクスポートします。
今後は `from calibration.lm_calibration import ...` または `from calibration import ...` を推奨します。
"""

try:
    from ...calibration.lm_calibration import *
    from ...calibration.lm_calibration import (
        load_calibration_master,
        calibrate_landmarks,
        apply_calibration,
    )
except ImportError:
    from calibration.lm_calibration import *
    from calibration.lm_calibration import (
        load_calibration_master,
        calibrate_landmarks,
        apply_calibration,
    )