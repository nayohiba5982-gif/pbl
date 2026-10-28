"""
feature_calculator.py (後方互換性エイリアス)

本モジュールは `preprocess.calibration.feature_calculator` へ移行しました。
既存コードとの互換性維持のため、`FeatureCalculator` を再エクスポートします。
今後は `from calibration.feature_calculator import FeatureCalculator` または
`from calibration import FeatureCalculator` を推奨します。
"""

try:
    from ...calibration.feature_calculator import FeatureCalculator
except ImportError:
    from calibration.feature_calculator import FeatureCalculator

__all__ = ["FeatureCalculator"]