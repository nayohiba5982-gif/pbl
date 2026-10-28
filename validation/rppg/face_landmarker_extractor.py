"""
face_landmarker_extractor.py
MediaPipe FaceLandmarker を用いて動画から25fpsで指定ランドマークのRGB値を取得するモジュール
"""

from pathlib import Path
from typing import Dict, List, Optional, Set
import av
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd

DEFAULT_LEFT_CHEEK: List[int] = [330, 347, 280, 266, 425]
DEFAULT_RIGHT_CHEEK: List[int] = [101, 118, 50, 36, 198]
DEFAULT_REGION_SIZE: int = 20  # px (data_collect/app/lib/rppg.ts と同一)


def find_model_path() -> Path:
  """FaceLandmarker の .task モデルファイルを探索"""
  candidates = [
      Path(__file__).resolve().parent / "face_landmarker.task",
      Path(__file__).resolve().parent.parent / "make_ai" / "face_landmarker.task",
      Path("/home/shigure/make_ai/make_ai/face_landmarker.task"),
  ]
  for p in candidates:
    if p.exists():
      return p
  raise FileNotFoundError(
      "face_landmarker.task が見つかりません。パスを確認してください。"
  )


class FaceLandmarkerRGBExtractor:
  """動画から25fpsで指定ランドマークのRGB輝度シグナルを抽出するクラス"""

  def __init__(
      self,
      model_path: Optional[Path | str] = None,
      region_size: int = DEFAULT_REGION_SIZE,
      target_fps: float = 25.0,
      left_cheek_idxs: Optional[List[int]] = None,
      right_cheek_idxs: Optional[List[int]] = None,
  ):
    self.model_path = Path(model_path) if model_path else find_model_path()
    self.region_size = region_size
    self.half_region = region_size // 2
    self.target_fps = target_fps

    self.left_cheek_idxs = left_cheek_idxs or DEFAULT_LEFT_CHEEK
    self.right_cheek_idxs = right_cheek_idxs or DEFAULT_RIGHT_CHEEK
    self.target_landmark_idxs = self.left_cheek_idxs + self.right_cheek_idxs

    base_options = python.BaseOptions(model_asset_path=str(self.model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    self.landmarker = vision.FaceLandmarker.create_from_options(options)

  def close(self):
    if self.landmarker:
      try:
        self.landmarker.close()
      except Exception:
        pass
      self.landmarker = None

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def extract_rgb_from_video(
      self,
      video_path: Path | str,
      gt_timestamps: Optional[np.ndarray] = None,
  ) -> pd.DataFrame:
    """動画から 30fps のタイミングでフレームをサンプリングし、

    指定ランドマーク (330, 347, 280, 266, 425, 101, 118, 50, 36, 198) の RGB値
    および左頬、右頬、両頬の平均RGB値を抽出して DataFrame を作成します。
    """
    video_path = Path(video_path)
    if not video_path.exists():
      raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    total_frames = stream.frames or 0
    fps = float(stream.average_rate or stream.base_rate or 30.0)

    # タイムスタンプグリッドの生成 (30fps: 40ms 刻み)
    dt = 1.0 / self.target_fps
    if gt_timestamps is not None and len(gt_timestamps) > 0:
      t_start, t_end = gt_timestamps[0], gt_timestamps[-1]
      target_times = np.arange(t_start, t_end, dt)
      # 各 target_time に最も近いフレームインデックスを決定
      raw_indices = np.searchsorted(gt_timestamps, target_times)
      raw_indices = np.clip(raw_indices, 0, len(gt_timestamps) - 1)
      frame_indices = raw_indices.copy()
      for i, tt in enumerate(target_times):
        idx = raw_indices[i]
        if idx > 0 and abs(gt_timestamps[idx - 1] - tt) < abs(
            gt_timestamps[idx] - tt
        ):
          frame_indices[i] = idx - 1
    else:
      duration = float(stream.duration * stream.time_base)
      target_times = np.arange(0.0, duration, dt)
      frame_indices = np.round(target_times * fps).astype(int)
      if total_frames > 0:
        frame_indices = np.clip(frame_indices, 0, total_frames - 1)

    selected_frame_set: Set[int] = set(frame_indices)

    # 選択されたフレームに対して FaceLandmarker を実行
    frame_rgb_data: Dict[int, Dict[int, np.ndarray]] = {}

    container.seek(0)
    for f_idx, frame in enumerate(container.decode(video=0)):
      if f_idx in selected_frame_set:
        arr = frame.to_ndarray(format="rgb24")
        h, w, _ = arr.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
        result = self.landmarker.detect(mp_image)

        if result.face_landmarks and len(result.face_landmarks) > 0:
          lm = result.face_landmarks[0]
          rgbs: Dict[int, np.ndarray] = {}
          for l_idx in self.target_landmark_idxs:
            cx = int(round(lm[l_idx].x * w))
            cy = int(round(lm[l_idx].y * h))
            x1 = max(0, cx - self.half_region)
            x2 = min(w, cx + self.half_region)
            y1 = max(0, cy - self.half_region)
            y2 = min(h, cy + self.half_region)
            crop = arr[y1:y2, x1:x2]
            if crop.size > 0:
              rgbs[l_idx] = crop.mean(axis=(0, 1))
            else:
              rgbs[l_idx] = np.array([np.nan, np.nan, np.nan])
          frame_rgb_data[f_idx] = rgbs
        else:
          # 顔未検出
          frame_rgb_data[f_idx] = {
              l_idx: np.array([np.nan, np.nan, np.nan])
              for l_idx in self.target_landmark_idxs
          }

    container.close()

    # DataFrame の構築 (25fps時系列)
    rows = []
    for t, f_idx in zip(target_times, frame_indices):
      rgbs = frame_rgb_data.get(f_idx, {})
      row = {"time": t}

      left_r, left_g, left_b = [], [], []
      right_r, right_g, right_b = [], [], []

      # 左頬
      for idx in self.left_cheek_idxs:
        rgb = rgbs.get(idx, [np.nan, np.nan, np.nan])
        row[f"rppg_lm{idx}_r"] = rgb[0]
        row[f"rppg_lm{idx}_g"] = rgb[1]
        row[f"rppg_lm{idx}_b"] = rgb[2]
        left_r.append(rgb[0])
        left_g.append(rgb[1])
        left_b.append(rgb[2])

      # 右頬
      for idx in self.right_cheek_idxs:
        rgb = rgbs.get(idx, [np.nan, np.nan, np.nan])
        row[f"rppg_lm{idx}_r"] = rgb[0]
        row[f"rppg_lm{idx}_g"] = rgb[1]
        row[f"rppg_lm{idx}_b"] = rgb[2]
        right_r.append(rgb[0])
        right_g.append(rgb[1])
        right_b.append(rgb[2])

      # 領域平均 (leftCheek, rightCheek, bothCheeks)
      row["rppg_leftCheek_r"] = np.nanmean(left_r) if not all(np.isnan(left_r)) else np.nan
      row["rppg_leftCheek_g"] = np.nanmean(left_g) if not all(np.isnan(left_g)) else np.nan
      row["rppg_leftCheek_b"] = np.nanmean(left_b) if not all(np.isnan(left_b)) else np.nan

      row["rppg_rightCheek_r"] = np.nanmean(right_r) if not all(np.isnan(right_r)) else np.nan
      row["rppg_rightCheek_g"] = np.nanmean(right_g) if not all(np.isnan(right_g)) else np.nan
      row["rppg_rightCheek_b"] = np.nanmean(right_b) if not all(np.isnan(right_b)) else np.nan

      both_r = left_r + right_r
      both_g = left_g + right_g
      both_b = left_b + right_b
      row["rppg_bothCheeks_r"] = np.nanmean(both_r) if not all(np.isnan(both_r)) else np.nan
      row["rppg_bothCheeks_g"] = np.nanmean(both_g) if not all(np.isnan(both_g)) else np.nan
      row["rppg_bothCheeks_b"] = np.nanmean(both_b) if not all(np.isnan(both_b)) else np.nan

      rows.append(row)

    return pd.DataFrame(rows)
