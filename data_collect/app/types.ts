// ============================================================
// types.ts
// アプリ全体で使用する型定義（blendshapes 非取得版）
// ============================================================

/** RGB値 */
export interface RGB {
  r: number;
  g: number;
  b: number;
}

/** rPPG用RGB計測データ */
export interface RppgMetrics {
  // ── 💡 各ランドマーク点個別のRGB（新インデックス計12点） ──
  // 左頬（LEFT_CHEEK_IDXS）
  lm330: RGB;
  lm347: RGB;
  lm348: RGB; // ★追加
  lm280: RGB;
  lm266: RGB;
  lm425: RGB;

  // 右頬（RIGHT_CHEEK_IDXS）
  lm101: RGB;
  lm118: RGB;
  lm119: RGB; // ★追加
  lm50: RGB;
  lm36: RGB;
  lm205: RGB;

  // ── 平均RGB ──
  leftCheek:  RGB;  // 左頬6点の平均 (★5点から6点に修正)
  rightCheek: RGB;  // 右頬6点の平均 (★5点から6点に修正)
  bothCheeks: RGB;  // 全12点の平均 (★10点から12点に修正)
}

/** MediaPipeランドマーク1点（生座標・小数点第5位まで） */
export interface RawLandmarkCoord {
  x: number;
  y: number;
  z: number;
}

/** MediaPipe顔メッシュのランドマーク数（虹彩除く） */
export const FACE_MESH_LANDMARK_COUNT = 468;

/** JSON送信用ランドマーク座標（抜粋20特徴点） */
export interface LandmarkPoints {
  ear: {
    "lm[362]": RawLandmarkCoord;
    "lm[133]": RawLandmarkCoord;
    "lm[384]": RawLandmarkCoord;
    "lm[157]": RawLandmarkCoord;
    "lm[385]": RawLandmarkCoord;
    "lm[158]": RawLandmarkCoord;
    "lm[386]": RawLandmarkCoord;
    "lm[159]": RawLandmarkCoord;
    "lm[263]": RawLandmarkCoord;
    "lm[33]":  RawLandmarkCoord;
    "lm[373]": RawLandmarkCoord;
    "lm[144]": RawLandmarkCoord;
    "lm[374]": RawLandmarkCoord;
    "lm[145]": RawLandmarkCoord;
    "lm[380]": RawLandmarkCoord;
    "lm[153]": RawLandmarkCoord;
  };
  headPose: {
    "lm[4]":  RawLandmarkCoord;
    "lm[10]": RawLandmarkCoord;
  };
  gaze: {
    "lm[473]": RawLandmarkCoord;
    "lm[468]": RawLandmarkCoord;
  };
}

/** キャリブレーション統計 */
export interface CalibrationStats {
  durationSec:   number;
  frameCount:    number;
  /** 抜粋20特徴点それぞれのキャリブレーション期間平均座標 */
  meanLandmarks: LandmarkPoints;
}

/** 計測データの1フレーム分 */
export interface CapturedFrame {
  time: number;
  clientEpochMs: number;
  clientPerfMs: number;
  /** 動画ラベル（1: 最初の講義動画 / 0: 終了の焚き火動画） */
  label: 0 | 1; 
  rppg: RppgMetrics | null; // ★ any から RppgMetrics に型を厳密化しておくと安全です
  landmarks: LandmarkPoints | null; 
  landmarkCentroid: RawLandmarkCoord | null; 
}

/** Firebaseに保存するデータ構造 */
export interface MeasurementRecord {
  data:             CapturedFrame[];
  clientCreatedAtMs: number; // Date.now()
  timestamp:         unknown; // serverTimestamp()
}