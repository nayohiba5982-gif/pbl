"""
calibration.rppg パッケージ (後方互換性エイリアス)

本パッケージは `preprocess.calibration.rppg_calibration` へ移行しました。
既存コードとの互換性維持のため、主要シンボルを再エクスポートします。
"""

try:
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

__all__ = [
    "DEFAULT_RPPG_CALIB_CONFIG",
    "DifferentialRPPGCalibrator",
    "calibrate_rppg_dataframe",
    "calibrate_rppg_segment",
    "calibrate_rppg_windows_array",
    "compute_baseline_for_series",
    "process_all_files_for_rppg_calibration",
    "run_rppg_calibration",
]
