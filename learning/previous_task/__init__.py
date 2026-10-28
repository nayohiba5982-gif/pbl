"""
Pretext Task (TS2Vec 埋め込み検証 ➔ PLS エルボー分析 ➔ PLS 特徴量抽出 ➔ GMM 直前の被験者別 Zスコア標準化) パッケージ

- TS2Vec 自己教師あり学習 (Cell 1〜5) は Google Colab (previous_task.ipynb) で実行されます。
- ローカル環境では、各機能に分割された通常の .py モジュール群を通じて以下の工程を実行します:
  - elbow_analysis.py : PLS 目的変数説明分散比 (累積 R²) エルボー分析
  - pls_extractor.py  : PLS 学習 & 潜在特徴量抽出・保存
  - standardization.py: GMM 入力直前の被験者別 (Subject-wise) Zスコア標準化
  - task.py           : パイプライン統括クラス (PreviousTask)
  - utils.py          : ウィンドウ正規化・標準化ユーティリティ
"""

from .task import PreviousTask, run_previous_task
from .utils import ALL_SUPPORTED_WINDOWS, normalize_window_list, standardize_subject_wise
from .elbow_analysis import run_pls_elbow_analysis
from .pls_extractor import run_pls_feature_extraction
from .standardization import run_subject_standardization

__all__ = [
    "PreviousTask",
    "run_previous_task",
    "ALL_SUPPORTED_WINDOWS",
    "normalize_window_list",
    "standardize_subject_wise",
    "run_pls_elbow_analysis",
    "run_pls_feature_extraction",
    "run_subject_standardization",
]
