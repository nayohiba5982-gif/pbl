"""
明示的物理特徴量（ドメイン統計量）で使用する定数・デフォルト定義モジュール
(learning/features/constants.py)
"""

# デフォルトの特徴量チャネル一覧（preprocess / TS2Vec の X_windows.npy 準拠: 8チャネル）
DEFAULT_FEATURE_COLS = [
    "ear_ratio",
    "head_pitch_diff",
    "head_yaw_diff",
    "head_roll_diff",
    "gaze_v_deg_diff",
    "gaze_h_deg_diff",
    "bvp_selected",
    "bpm_diff",
]

# デフォルトの移動量（頭部・視線）カラム設定
DEFAULT_HEAD_POSE_COLS = ["head_pitch_diff", "head_yaw_diff", "head_roll_diff"]
DEFAULT_GAZE_COLS = ["gaze_v_deg_diff", "gaze_h_deg_diff"]

# 潜在空間 (TS2Vec 埋め込み) の次元数設定:
# 視覚 (visual) および 生体 (vital) の各ストリームはそれぞれ 32 次元
DEFAULT_VISUAL_DIMS = 32
DEFAULT_VITAL_DIMS = 32
DEFAULT_TS2VEC_DIMS = 32                     # 各ストリーム単体のデフォルト次元数 (32次元)
DEFAULT_TOTAL_LATENT_DIMS = 64               # 結合潜在次元数 (32 + 32 = 64次元)

# 明示的物理統計量 (20次元: 平均8 + 標準偏差8 + 移動量分散4)
DEFAULT_EXPLICIT_STATS_DIMS = 20
