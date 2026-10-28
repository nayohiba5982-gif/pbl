// ============================================================
// hooks/useWebcam.ts
// Webカメラのストリーム取得を管理するHook
// ============================================================

"use client";

import { useRef, useState, useEffect } from "react";

interface UseWebcamResult {
  videoRef: React.RefObject<HTMLVideoElement | null>; // null を許容
  isRunning: boolean;
  startCamera: (onReady: () => void) => Promise<void>;
  error: string | null;
}

/**
 * Webカメラを起動し、videoRef にストリームをセットする。
 * 映像データ読み込み完了後に onReady コールバックを呼ぶ。
 */
export function useWebcam(): UseWebcamResult {
  // ── 修正ポイント1: 返り値の型（UseWebcamResult）と完全に一致させる ──
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
  }, []);

  const startCamera = async (onReady: () => void) => {
    if (!videoRef.current) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      setIsRunning(true);

      // ── 修正ポイント2: より確実に発火する onloadedmetadata に変更し、明示的に再生する ──
      videoRef.current.onloadedmetadata = async () => {
        if (videoRef.current) {
          try {
            await videoRef.current.play();
            // 完全に映像が再生開始されてからMediaPipeのループ（onReady）を起動
            onReady();
          } catch (playErr) {
            console.error("ビデオ再生失敗:", playErr);
          }
        }
      };
    } catch (err) {
      console.error("カメラ起動失敗:", err);
      setError("カメラ起動失敗 (権限を確認してください)");
    }
  };

  return { videoRef, isRunning, startCamera, error };
}