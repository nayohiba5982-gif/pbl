// ============================================================
// lib/rppg.ts
// rPPG用のRGB値取得ユーティリティ（6点ROI拡張版・1回getImageData最適化）
// ============================================================

import type { RGB, RppgMetrics } from "../types";

/** ランドマーク1点の型 */
interface Landmark {
  x: number;
  y: number;
}

// ── 💡 左右の頬のROIインデックス定義（ご指定の6点ずつに変更） ──
const LEFT_CHEEK_IDXS  = [330, 347, 348, 280, 266, 425]; // ★ 348 を追加
const RIGHT_CHEEK_IDXS = [101, 118, 119, 50, 36, 205];  // ★ 119 を追加

// 各ランドマーク周辺の取得領域サイズ（px）
const REGION_SIZE = 20;
const HALF        = REGION_SIZE / 2;

/**
 * 複数RGB値の平均を計算する
 * 💡 配列の長さ（rgbs.length）で割るため、5点から6点に増えても自動で正しく計算されます
 */
function avgRGB(rgbs: RGB[]): RGB {
  if (rgbs.length === 0) return { r: 0, g: 0, b: 0 };
  return {
    r: Math.round(rgbs.reduce((s, c) => s + c.r, 0) / rgbs.length),
    g: Math.round(rgbs.reduce((s, c) => s + c.g, 0) / rgbs.length),
    b: Math.round(rgbs.reduce((s, c) => s + c.b, 0) / rgbs.length),
  };
}

/**
 * 各ランドマーク点個別のRGBと平均RGBを取得する
 * 💡 最適化: 12回の個別getImageDataを避け、対象領域全体を1回のgetImageDataで取得
 */
export function extractRppgMetrics(
  ctx: CanvasRenderingContext2D,
  lm: Landmark[]
): RppgMetrics {
  const allIdxs = [...LEFT_CHEEK_IDXS, ...RIGHT_CHEEK_IDXS];

  // 1. 各ROIパッチの矩形領域を計算
  const rects = allIdxs.map((idx) => {
    if (!lm[idx]) return null;
    const px = lm[idx].x * ctx.canvas.width;
    const py = lm[idx].y * ctx.canvas.height;
    const cx = Math.max(0, Math.round(px - HALF));
    const cy = Math.max(0, Math.round(py - HALF));
    const cw = Math.min(REGION_SIZE, ctx.canvas.width - cx);
    const ch = Math.min(REGION_SIZE, ctx.canvas.height - cy);
    if (cw <= 0 || ch <= 0) return null;
    return { cx, cy, cw, ch };
  });

  const validRects = rects.filter((r): r is { cx: number; cy: number; cw: number; ch: number } => r !== null);
  const rgbMap = new Map<number, RGB>();

  if (validRects.length > 0) {
    // 2. 全パッチを包含するバウンディングボックスを算出
    let minX = ctx.canvas.width;
    let minY = ctx.canvas.height;
    let maxX = 0;
    let maxY = 0;

    for (const r of validRects) {
      if (r.cx < minX) minX = r.cx;
      if (r.cy < minY) minY = r.cy;
      if (r.cx + r.cw > maxX) maxX = r.cx + r.cw;
      if (r.cy + r.ch > maxY) maxY = r.cy + r.ch;
    }

    const bboxW = maxX - minX;
    const bboxH = maxY - minY;

    if (bboxW > 0 && bboxH > 0) {
      // 💡 1フレームあたり1回だけ getImageData を実行
      const imgData = ctx.getImageData(minX, minY, bboxW, bboxH);
      const data = imgData.data;
      const stride = bboxW * 4;

      allIdxs.forEach((idx, i) => {
        const rect = rects[i];
        if (!rect) {
          rgbMap.set(idx, { r: 0, g: 0, b: 0 });
          return;
        }

        const localX = rect.cx - minX;
        const localY = rect.cy - minY;
        let rSum = 0, gSum = 0, bSum = 0;
        const count = rect.cw * rect.ch;

        for (let dy = 0; dy < rect.ch; dy++) {
          const rowOffset = (localY + dy) * stride + localX * 4;
          for (let dx = 0; dx < rect.cw; dx++) {
            const pxOffset = rowOffset + dx * 4;
            rSum += data[pxOffset];
            gSum += data[pxOffset + 1];
            bSum += data[pxOffset + 2];
          }
        }

        rgbMap.set(idx, {
          r: Math.round(rSum / count),
          g: Math.round(gSum / count),
          b: Math.round(bSum / count),
        });
      });
    }
  }

  // 取得できなかったインデックスは 0 で補完
  allIdxs.forEach((idx) => {
    if (!rgbMap.has(idx)) {
      rgbMap.set(idx, { r: 0, g: 0, b: 0 });
    }
  });

  // 2. 頬全体の平均RGBを計算
  const leftRgbs  = LEFT_CHEEK_IDXS.map((idx) => rgbMap.get(idx)!);
  const rightRgbs = RIGHT_CHEEK_IDXS.map((idx) => rgbMap.get(idx)!);

  const leftCheek  = avgRGB(leftRgbs);
  const rightCheek = avgRGB(rightRgbs);
  const bothCheeks = avgRGB([...leftRgbs, ...rightRgbs]);

  // 3. 個別ランドマークのオブジェクトを生成
  const individualLandmarks: Record<string, RGB> = {};
  LEFT_CHEEK_IDXS.forEach((idx, i) => {
    individualLandmarks[`lm${idx}`] = leftRgbs[i];
  });
  RIGHT_CHEEK_IDXS.forEach((idx, i) => {
    individualLandmarks[`lm${idx}`] = rightRgbs[i];
  });

  return {
    ...individualLandmarks,
    leftCheek,
    rightCheek,
    bothCheeks,
  } as RppgMetrics;
}