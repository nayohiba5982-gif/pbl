import numpy as np

class FeatureCalculator:
    """
    MediaPipeのランドマークデータから、EAR、Head Pose、
    Gazeを計算・管理するクラス。
    """
    def __init__(self, r_eye: float = 11.7):
        """r_eye (float): 眼球の平均半径 (mm)。デフォルト値は 11.7"""
        self.r_eye = r_eye
        self.ear_threshold = 0.2  # EAR閾値の初期値
        
        # キャリブレーション用の履歴保持
        self.ear_history_open = []
        self.ear_history_closed = []

    def _get_coord(self, landmarks, idx: int) -> np.ndarray:
        """
        様々な入力形式（NumPy二次元配列、辞書型、オブジェクト型など）から
        指定インデックスの 3D 座標 [x, y, z] を NumPy 配列として安全に抽出する内部メソッド。
        """
        # 1. NumPy配列、または通常のリストの場合
        if isinstance(landmarks, (np.ndarray, list)):
            return np.array(landmarks[idx])
            
        # 2. 辞書型（JSONやCSVの展開データを想定）の場合
        elif isinstance(landmarks, dict):
            key_str = f"lm[{idx}]"
            if key_str in landmarks:
                coord = landmarks[key_str]
                return np.array([coord['x'], coord['y'], coord['z']])
            
            # フラット化された辞書（CSVの1行など）の場合
            for prefix in ['lm_ear_', 'lm_headPose_', 'lm_gaze_']:
                x_key = f"{prefix}{idx}_x"
                y_key = f"{prefix}{idx}_y"
                z_key = f"{prefix}{idx}_z"
                if x_key in landmarks:
                    return np.array([landmarks[x_key], landmarks[y_key], landmarks[z_key]])
                    
            # 単純に数値キーが入っている場合
            if idx in landmarks:
                coord = landmarks[idx]
                if isinstance(coord, dict):
                    return np.array([coord.get('x', 0), coord.get('y', 0), coord.get('z', 0)])
                return np.array(coord)
                
        # 3. MediaPipeの生オブジェクト
        elif hasattr(landmarks, 'landmark'):
            lm = landmarks.landmark[idx]
            return np.array([lm.x, lm.y, lm.z])
            
        elif hasattr(landmarks, 'x') and hasattr(landmarks, 'y'):
            lm = landmarks[idx]
            return np.array([lm.x, lm.y, lm.z])

        raise ValueError(f"サポートされていないランドマークのデータ形式、またはインデックスが見つかりません: {idx}")

    def _calculate_distance_3d(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """2点間の3Dユークリッド距離を計算する内部メソッド"""
        return float(np.linalg.norm(p1 - p2))

    def _calculate_eye_ear_logic(self, landmarks, p0, p3, p4, p5, p8, p11, p12, p13) -> float:
        """指定されたインデックスを用いて片目分のEARを計算する"""
        coord_p0  = self._get_coord(landmarks, p0)
        coord_p3  = self._get_coord(landmarks, p3)
        coord_p4  = self._get_coord(landmarks, p4)
        coord_p5  = self._get_coord(landmarks, p5)
        coord_p8  = self._get_coord(landmarks, p8)
        coord_p11 = self._get_coord(landmarks, p11)
        coord_p12 = self._get_coord(landmarks, p12)
        coord_p13 = self._get_coord(landmarks, p13)

        # 分子: 垂直方向の距離の和
        bunshi = (self._calculate_distance_3d(coord_p3, coord_p13) + 
                  self._calculate_distance_3d(coord_p4, coord_p12) + 
                  self._calculate_distance_3d(coord_p5, coord_p11))
        
        # 分母: 水平方向の距離 * 3
        bunbo = self._calculate_distance_3d(coord_p0, coord_p8) * 3.0
        
        if bunbo == 0:
            return 0.0
            
        return bunshi / bunbo

    def calculate_instant_ear(self, landmarks) -> tuple[float, float, float]:
        """現在のフレームの瞬時EARを左右それぞれ計算し、その平均値とともに返す。"""
        # 1. 左目のEAR計算 (300番台のランドマーク)
        left_ear = self._calculate_eye_ear_logic(
            landmarks,
            p0=362, p3=384, p4=385, p5=386,
            p8=263, p11=373, p12=374, p13=380
        )
        # 2. 右目のEAR計算 (100番台・2桁のランドマーク)
        right_ear = self._calculate_eye_ear_logic(
            landmarks,
            p0=133, p3=157, p4=158, p5=159,
            p8=33,  p11=144, p12=145, p13=153
        )
        # 3. 両目の平均EARを計算
        average_ear = (left_ear + right_ear) / 2.0
        return left_ear, right_ear, average_ear

    def add_calibration_frame(self, landmarks, is_open: bool):
        """キャリブレーション用に、現在のフレームのランドマークを履歴に追加する。"""
        if is_open:
            self.ear_history_open.append(landmarks)
        else:
            self.ear_history_closed.append(landmarks)

    def update_personalized_ear_threshold(self) -> float:
        """蓄積されたキャリブレーション履歴データから判定閾値を更新する。"""
        if not self.ear_history_closed or not self.ear_history_open:
            raise ValueError("キャリブレーションデータが不足しています。")

        # --- EAR_Closed の計算 ---
        min_3_13 = min(self._calculate_distance_3d(self._get_coord(lm, 384), self._get_coord(lm, 380)) for lm in self.ear_history_closed)
        min_4_12 = min(self._calculate_distance_3d(self._get_coord(lm, 385), self._get_coord(lm, 374)) for lm in self.ear_history_closed)
        min_5_11 = min(self._calculate_distance_3d(self._get_coord(lm, 386), self._get_coord(lm, 373)) for lm in self.ear_history_closed)
        max_0_8  = max(self._calculate_distance_3d(self._get_coord(lm, 362), self._get_coord(lm, 263)) for lm in self.ear_history_closed)
        ear_closed = (min_3_13 + min_4_12 + min_5_11) / (3.0 * max_0_8)
        
        # --- EAR_Open の計算 ---
        mean_3_13 = np.mean([self._calculate_distance_3d(self._get_coord(lm, 384), self._get_coord(lm, 380)) for lm in self.ear_history_open])
        mean_4_12 = np.mean([self._calculate_distance_3d(self._get_coord(lm, 385), self._get_coord(lm, 374)) for lm in self.ear_history_open])
        mean_5_11 = np.mean([self._calculate_distance_3d(self._get_coord(lm, 386), self._get_coord(lm, 373)) for lm in self.ear_history_open])
        min_0_8   = min(self._calculate_distance_3d(self._get_coord(lm, 362), self._get_coord(lm, 263)) for lm in self.ear_history_open)
        ear_open = (mean_3_13 + mean_4_12 + mean_5_11) / (3.0 * min_0_8)
        
        self.ear_threshold = (ear_open - ear_closed) / 2.0
        return self.ear_threshold

    def is_blinking(self, landmarks) -> bool:
        """現在のフレームが瞬目（まばたき）状態であるかを判定する"""
        current_ear = self.calculate_instant_ear(landmarks)[2] # 平均EARを使用
        return current_ear < self.ear_threshold

    def calculate_head_pose(self, landmarks) -> tuple[float, float, float]:
        """頭部回転角 (Yaw, Roll, Pitch) を度数法(degree)で計算する。"""
        p4 = self._get_coord(landmarks, 4)    # 鼻の頂点
        p10 = self._get_coord(landmarks, 10)  # 額の頂点
        yaw = np.arctan2(p4[0], p4[2])
        roll = np.arctan2(p10[0], p10[1])
        pitch = np.arctan2(p4[1], p4[2])
        return float(np.degrees(yaw)), float(np.degrees(roll)), float(np.degrees(pitch))

    def calculate_eye_rotation(self, landmarks) -> tuple[float, float]:
        """眼球回転角 (Horizontal, Vertical) を度数法(degree)で計算する。"""
        p362 = self._get_coord(landmarks, 362)
        p263 = self._get_coord(landmarks, 263)
        eye_center = (p362 + p263) / 2.0
        iris_center = self._get_coord(landmarks, 473)
        x_diff = iris_center[0] - eye_center[0]
        y_diff = iris_center[1] - eye_center[1]
        
        sin_val_h = np.clip(x_diff / self.r_eye, -1.0, 1.0)
        eye_horizontal_degree = np.arcsin(sin_val_h)
        sin_val_v = np.clip(y_diff / self.r_eye, -1.0, 1.0)
        eye_vertical_degree = np.arcsin(sin_val_v)
        return float(np.degrees(eye_horizontal_degree)), float(np.degrees(eye_vertical_degree))
