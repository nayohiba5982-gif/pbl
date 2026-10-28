// ============================================================
// hooks/useFaceLandmarker.ts
// MediaPipe FaceLandmarker モデルの初期化を管理するHook（デバッグログ付き）
// ============================================================

"use client";

import { useEffect, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

interface UseFaceLandmarkerResult {
  faceLandmarkerRef: React.RefObject<FaceLandmarker | null>;
  isReady: boolean;
  error: string | null;
}

export function useFaceLandmarker(): UseFaceLandmarkerResult {
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const initializingRef = useRef(false);

  useEffect(() => {
    if (initializingRef.current) {
      console.log("[MediaPipe] すでに初期化が開始されているため、2回目の実行をスキップします。");
      return;
    }
    initializingRef.current = true;

    async function initialize() {
      try {
        console.log("[MediaPipe] 1. WASMリゾルバーの読み込みを開始します...");
        
        const filesetResolver = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm"
        );
        
        console.log("[MediaPipe] 2. WASMリゾルバーの読み込み完了。モデル（.task）の読み込みを開始します...");

        const landmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            // 💡 修正ポイント: 動作の確実性を上げるため "GPU" から "CPU" に変更
            delegate: "CPU", 
          },
          outputFaceBlendshapes:             false,
          outputFacialTransformationMatrixes: true,
          runningMode: "VIDEO",
          numFaces:    1,
        });

        console.log("[MediaPipe] 3. AIモデルの初期化がすべて正常に完了しました！");
        faceLandmarkerRef.current = landmarker;
        setIsReady(true);
      } catch (err) {
        console.error("[MediaPipe] ❌ 初期化中にエラーが発生しました:", err);
        setError("読み込み失敗");
        initializingRef.current = false;
      }
    }

    initialize();
  }, []);

  return { faceLandmarkerRef, isReady, error };
}