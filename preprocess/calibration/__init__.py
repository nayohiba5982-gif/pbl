"""
calibration パッケージ

生体指標 (rPPG) および顔ランドマーク (LM) のキャリブレーション処理を提供します。
- lm_calibration.py: 顔ランドマーク用 外部マスター基準 (calibration_master.csv) キャリブレーション
- rppg_calibration.py: 生体指標 (rPPG / BPM) 用 10秒ウィンドウ内完結型 差分キャリブレーション
"""

try:
    from .lm_calibration import (
        load_calibration_master,
        calibrate_landmarks,
        apply_calibration,
    )
    from .feature_calculator import FeatureCalculator
    from . import cal_calibration
    from .rppg_calibration import (
        compute_baseline_for_series,
        calibrate_rppg_dataframe,
        calibrate_rppg_segment,
        calibrate_rppg_windows_array,
        DifferentialRPPGCalibrator,
        process_all_files_for_rppg_calibration,
        run_rppg_calibration,
    )
    from . import rppg
except ImportError:
    from lm_calibration import (
        load_calibration_master,
        calibrate_landmarks,
        apply_calibration,
    )
    from feature_calculator import FeatureCalculator
    try:
        import cal_calibration
    except ImportError:
        cal_calibration = None
    from rppg_calibration import (
        compute_baseline_for_series,
        calibrate_rppg_dataframe,
        calibrate_rppg_segment,
        calibrate_rppg_windows_array,
        DifferentialRPPGCalibrator,
        process_all_files_for_rppg_calibration,
        run_rppg_calibration,
    )
    try:
        import rppg
    except ImportError:
        rppg = None

__all__ = [
    # LM
    "load_calibration_master",
    "calibrate_landmarks",
    "apply_calibration",
    "FeatureCalculator",
    "cal_calibration",
    # rPPG
    "compute_baseline_for_series",
    "calibrate_rppg_dataframe",
    "calibrate_rppg_segment",
    "calibrate_rppg_windows_array",
    "DifferentialRPPGCalibrator",
    "process_all_files_for_rppg_calibration",
    "run_rppg_calibration",
    # Subpackage backward compatibility
    "rppg",
]
