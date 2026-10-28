// ============================================================
// lib/landmarks.ts
// JSON送信用ランドマーク座標の抽出（blendshapes 非取得版）
// ============================================================

import { FACE_MESH_LANDMARK_COUNT, type LandmarkPoints, type RawLandmarkCoord } from "../types";

export interface Landmark {
  x: number;
  y: number;
  z: number;
}

/** * 小数点第6位で四捨五入して第5位まで保持する内部ヘルパー
 * 💡 blendshapes.ts 依存を排除するために直接定義
 */
function roundMediaPipeCoord(val: number): number {
  return Math.round(val * 100000) / 100000;
}

/** 生のランドマーク座標を丸め処理してRawLandmarkCoord型に変換する */
function toRawLandmark(lm: Landmark): RawLandmarkCoord {
  return {
    x: roundMediaPipeCoord(lm.x),
    y: roundMediaPipeCoord(lm.y),
    z: roundMediaPipeCoord(lm.z),
  };
}

/** 指定インデックスのランドマークを小数点第5位まで丸めて返す */
function pick(lm: Landmark[], index: number): RawLandmarkCoord {
  return toRawLandmark(lm[index]);
}

/** 複数フレームのランドマーク座標から各点の平均を算出する */
function avgCoord(frames: RawLandmarkCoord[]): RawLandmarkCoord {
  const n = frames.length;
  return {
    x: roundMediaPipeCoord(frames.reduce((s, c) => s + c.x, 0) / n),
    y: roundMediaPipeCoord(frames.reduce((s, c) => s + c.y, 0) / n),
    z: roundMediaPipeCoord(frames.reduce((s, c) => s + c.z, 0) / n),
  };
}

/** キャリブレーション期間の抜粋20特徴点それぞれの平均座標を算出する */
export function averageLandmarks(frames: LandmarkPoints[]): LandmarkPoints | null {
  if (frames.length === 0) return null;

  // 💡 外側のpick関数との競合（シャドウイング）を避けるため pickCoord に名称変更
  const pickCoord = (getter: (f: LandmarkPoints) => RawLandmarkCoord) =>
    avgCoord(frames.map(getter));

  return {
    ear: {
      "lm[362]": pickCoord((f) => f.ear["lm[362]"]),
      "lm[133]": pickCoord((f) => f.ear["lm[133]"]),
      "lm[384]": pickCoord((f) => f.ear["lm[384]"]),
      "lm[157]": pickCoord((f) => f.ear["lm[157]"]),
      "lm[385]": pickCoord((f) => f.ear["lm[385]"]),
      "lm[158]": pickCoord((f) => f.ear["lm[158]"]),
      "lm[386]": pickCoord((f) => f.ear["lm[386]"]),
      "lm[159]": pickCoord((f) => f.ear["lm[159]"]),
      "lm[263]": pickCoord((f) => f.ear["lm[263]"]),
      "lm[33]":  pickCoord((f) => f.ear["lm[33]"]),
      "lm[373]": pickCoord((f) => f.ear["lm[373]"]),
      "lm[144]": pickCoord((f) => f.ear["lm[144]"]),
      "lm[374]": pickCoord((f) => f.ear["lm[374]"]),
      "lm[145]": pickCoord((f) => f.ear["lm[145]"]),
      "lm[380]": pickCoord((f) => f.ear["lm[380]"]),
      "lm[153]": pickCoord((f) => f.ear["lm[153]"]),
    },
    headPose: {
      "lm[4]":  pickCoord((f) => f.headPose["lm[4]"]),
      "lm[10]": pickCoord((f) => f.headPose["lm[10]"]),
    },
    gaze: {
      "lm[473]": pickCoord((f) => f.gaze["lm[473]"]),
      "lm[468]": pickCoord((f) => f.gaze["lm[468]"]),
    },
  };
}

/** 468点顔メッシュの重心座標（各軸の平均）を算出する */
export function extractLandmarkCentroid(lm: Landmark[]): RawLandmarkCoord | null {
  if (!lm || lm.length < FACE_MESH_LANDMARK_COUNT) return null;

  let sx = 0;
  let sy = 0;
  let sz = 0;
  for (let i = 0; i < FACE_MESH_LANDMARK_COUNT; i++) {
    sx += lm[i].x;
    sy += lm[i].y;
    sz += lm[i].z;
  }
  const n = FACE_MESH_LANDMARK_COUNT;
  return {
    x: roundMediaPipeCoord(sx / n),
    y: roundMediaPipeCoord(sy / n),
    z: roundMediaPipeCoord(sz / n),
  };
}

/**
 * 計測JSON送信用のランドマーク座標を抽出する
 * 全座標は小数点第6位で四捨五入（第5位まで保持）
 */
export function extractLandmarks(lm: Landmark[]): LandmarkPoints | null {
  if (!lm || lm.length < 478) return null;

  return {
    ear: {
      "lm[362]": pick(lm, 362),
      "lm[133]": pick(lm, 133),
      "lm[384]": pick(lm, 384),
      "lm[157]": pick(lm, 157),
      "lm[385]": pick(lm, 385),
      "lm[158]": pick(lm, 158),
      "lm[386]": pick(lm, 386),
      "lm[159]": pick(lm, 159),
      "lm[263]": pick(lm, 263),
      "lm[33]":  pick(lm, 33),
      "lm[373]": pick(lm, 373),
      "lm[144]": pick(lm, 144),
      "lm[374]": pick(lm, 374),
      "lm[145]": pick(lm, 145),
      "lm[380]": pick(lm, 380),
      "lm[153]": pick(lm, 153),
    },
    headPose: {
      "lm[4]":  pick(lm, 4),
      "lm[10]": pick(lm, 10),
    },
    gaze: {
      "lm[473]": pick(lm, 473),
      "lm[468]": pick(lm, 468),
    },
  };
}