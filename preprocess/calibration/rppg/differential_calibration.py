"""
differential_calibration.py (後方互換性エイリアス)

本モジュールは `preprocess.calibration.rppg_calibration` へ移行しました。
既存コードとの互換性維持のため、`rppg_calibration` の全シンボルを再エクスポートします。
今後は `from calibration.rppg_calibration import ...` または `from calibration import ...` を推奨します。
"""

try:
    from ..rppg_calibration import *
    from ..rppg_calibration import (
        DEFAULT_RPPG_CALIB_CONFIG,
        DifferentialRPPGCalibrator,
        calibrate_rppg_dataframe,
        calibrate_rppg_segment,
        calibrate_rppg_windows_array,
        compute_baseline_for_series,
        process_all_files_for_rppg_calibration,
        run_rppg_calibration,
    )
except ImportError:
    from calibration.rppg_calibration import *
    from calibration.rppg_calibration import (
        DEFAULT_RPPG_CALIB_CONFIG,
        DifferentialRPPGCalibrator,
        calibrate_rppg_dataframe,
        calibrate_rppg_segment,
        calibrate_rppg_windows_array,
        compute_baseline_for_series,
        process_all_files_for_rppg_calibration,
        run_rppg_calibration,
    )
