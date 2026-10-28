// ============================================================
// hooks/useMeasurement.ts
// 計測ループとデータの保存・送信を管理するHook（匿名認証対応版）
// ============================================================

"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { FaceLandmarker, type NormalizedLandmark } from "@mediapipe/tasks-vision";
import { signInAnonymously, type User } from "firebase/auth";
import { averageLandmarks, extractLandmarkCentroid, extractLandmarks } from "../lib/landmarks";
import { extractRppgMetrics } from "../lib/rppg";
import { getFirebaseStorage, getFirebaseAuth, uploadMeasurement, downloadMeasurementJson } from "../lib/firebase";
import type { CapturedFrame, CalibrationStats, LandmarkPoints } from "../types";
import type { FirebaseStorage } from "firebase/storage";

const CALIBRATION_DURATION = 5;
const TARGET_FPS           = 25;
const FRAME_INTERVAL       = 1000 / TARGET_FPS;

interface UseMeasurementOptions {
  videoRef:          React.RefObject<HTMLVideoElement | null>;
  canvasRef:         React.RefObject<HTMLCanvasElement | null>;
  measureVideoRef:   React.RefObject<HTMLVideoElement | null>;
  endVideoRef:       React.RefObject<HTMLVideoElement | null>;
  faceLandmarkerRef: React.RefObject<FaceLandmarker | null>;
}

type Phase = "idle" | "calibrating" | "watching_video" | "ending_video";

interface UseMeasurementResult {
  phase:             Phase;
  isMeasuring:       boolean;
  isFaceDetected:    boolean;
  remainingTime:     number | null;
  remainingTimeText: string;
  calibrationTime:   number;
  status:            string;
  startLoop:         () => void;
  startMeasurement:  () => void;
  handleVideoEnd:    () => void;
  handleEndVideoEnd: () => void;
  handleTimeUpdate:  (e: React.SyntheticEvent<HTMLVideoElement>) => void;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function useMeasurement({
  videoRef,
  canvasRef,
  measureVideoRef,
  endVideoRef,
  faceLandmarkerRef,
}: UseMeasurementOptions): UseMeasurementResult {

  const [phase,           setPhase]           = useState<Phase>("idle");
  const [isFaceDetected,  setIsFaceDetected]  = useState(true);
  const [remainingTime,   setRemainingTime]   = useState<number | null>(null);
  const [calibrationTime, setCalibrationTime] = useState(CALIBRATION_DURATION);
  const [status,          setStatus]          = useState<string>(() => {
    const auth = getFirebaseAuth();
    return auth ? "サーバー認証中..." : "認証システムの初期化に失敗しました";
  });
  
  const [currentUser,     setCurrentUser]     = useState<User | null>(null);

  const phaseRef            = useRef<Phase>("idle");
  const isFaceDetectedRef   = useRef(true);
  const capturedData        = useRef<CapturedFrame[]>([]);
  const lastVideoTime       = useRef(-1);
  const lastProcessedMs     = useRef(-1);
  const requestRef          = useRef<number | null>(null);
  const storageRef          = useRef<FirebaseStorage | null>(null);
  const calibrationStatsRef = useRef<CalibrationStats | null>(null);

  const calibrationLandmarkFrames = useRef<LandmarkPoints[]>([]);
  const calibrationStart          = useRef<number | null>(null);
  const videoStart                = useRef<number | null>(null);
  const endingVideoStart          = useRef<number | null>(null);

  const lastFrameTime  = useRef(0);

  useEffect(() => { phaseRef.current          = phase;          }, [phase]);
  useEffect(() => { isFaceDetectedRef.current = isFaceDetected; }, [isFaceDetected]);

  useEffect(() => {
    storageRef.current = getFirebaseStorage();
    const auth = getFirebaseAuth();

    if (!auth) {
      return;
    }

    signInAnonymously(auth)
      .then((cred) => {
        setCurrentUser(cred.user);
        console.log("🔒 匿名認証に成功しました。UID:", cred.user.uid);
        setStatus("待機中");
      })
      .catch((err) => {
        console.error("❌ 匿名認証に失敗しました:", err);
        setStatus("認証エラー: ページを再読み込みしてください");
      });
  }, []);

  useEffect(() => {
    return () => { if (requestRef.current) cancelAnimationFrame(requestRef.current); };
  }, []);

  /** フレームデータを収集する共通処理 */
  const collectFrame = (
    ctx: CanvasRenderingContext2D | null,
    lm: NormalizedLandmark[] | null,
    elapsedSec: number,
    timestamp: number,
    label: 0 | 1
  ) => {
    const rppg             = (ctx && lm) ? extractRppgMetrics(ctx, lm) : null;
    const landmarks        = lm ? extractLandmarks(lm) : null;
    const landmarkCentroid = lm ? extractLandmarkCentroid(lm) : null;

    capturedData.current.push({
      time:             parseFloat(elapsedSec.toFixed(3)),
      clientEpochMs:    Date.now(),
      clientPerfMs:     timestamp,
      label,
      rppg,
      landmarks,
      landmarkCentroid,
    });
  };

  const predictWebcamRef = useRef<((timestamp: number) => void) | null>(null);

  /** メインループ */
  const predictWebcam = useCallback((timestamp: number) => {
    if (predictWebcamRef.current) {
      requestRef.current = requestAnimationFrame(predictWebcamRef.current);
    }

    if (timestamp - lastFrameTime.current < FRAME_INTERVAL) return;
    lastFrameTime.current = timestamp;

    const video          = videoRef.current;
    const canvas         = canvasRef.current;
    const mVideo         = measureVideoRef.current;
    const faceLandmarker = faceLandmarkerRef.current;

    if (!video || !canvas || !faceLandmarker) return;
    if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0 || video.currentTime === lastVideoTime.current) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // 💡 Canvas サイズが変わった時だけ再設定（不要な再アロケーションを防止）
    if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
    if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;

    lastVideoTime.current = video.currentTime;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const currentMs = Math.floor(performance.now());
    if (currentMs <= lastProcessedMs.current) return;
    lastProcessedMs.current = currentMs;

    try {
      const results = faceLandmarker.detectForVideo(video, currentMs);
      const hasFace = !!(results.faceLandmarks?.length);

      if (hasFace) {
        if (!isFaceDetectedRef.current) setIsFaceDetected(true);
      } else {
        if (isFaceDetectedRef.current) setIsFaceDetected(false);
      }

      const lm = hasFace ? results.faceLandmarks[0] : null;

      if (phaseRef.current === "calibrating") {
        if (lm && calibrationStart.current !== null) {
          const landmarks = extractLandmarks(lm);
          if (landmarks) {
            calibrationLandmarkFrames.current.push(landmarks);
            const elapsed   = (timestamp - calibrationStart.current) / 1000;
            const remaining = Math.max(0, CALIBRATION_DURATION - elapsed);
            setCalibrationTime(remaining);

            if (elapsed >= CALIBRATION_DURATION) {
              const landmarkFrames = calibrationLandmarkFrames.current;
              const n              = landmarkFrames.length;
              const meanLandmarks  = averageLandmarks(landmarkFrames);
              if (n > 0 && meanLandmarks) {
                calibrationStatsRef.current = { durationSec: CALIBRATION_DURATION, frameCount: n, meanLandmarks };
              }
              capturedData.current              = [];
              calibrationLandmarkFrames.current = [];
              calibrationStart.current          = null;
              videoStart.current                = timestamp;

              phaseRef.current = "watching_video";
              setPhase("watching_video");
              setStatus("動画再生中...（音声が流れています。音量調節をしてください）");

              if (mVideo) {
                mVideo.muted = false;
                mVideo.currentTime = 0;
                mVideo.play().catch((playErr) => {
                  console.error("動画再生エラー:", playErr);
                  setPhase("idle");
                  setStatus("再生エラー");
                });
              }
            }
          }
        }
      }
      
      if (phaseRef.current === "watching_video") {
        if (mVideo) {
          const elapsedSec = mVideo.currentTime; 
          collectFrame(hasFace ? ctx : null, lm, elapsedSec, timestamp, 1);
        }
      }

      if (phaseRef.current === "ending_video") {
        const eVideo = endVideoRef.current;
        if (eVideo) {
          const elapsedSec = eVideo.currentTime; 
          collectFrame(hasFace ? ctx : null, lm, elapsedSec, timestamp, 0);
        }
      }
    } catch (err) {
      console.warn("[MediaPipe] Detection error:", err);
    }
  }, [videoRef, canvasRef, measureVideoRef, endVideoRef, faceLandmarkerRef]);

  useEffect(() => {
    predictWebcamRef.current = predictWebcam;
  }, [predictWebcam]);

  /** ループ開始 */
  const startLoop = useCallback(() => {
    setStatus("待機中");
    const checkVideoAndStart = () => {
      const video = videoRef.current;
      if (video && video.readyState >= 2 && video.videoWidth > 0) {
        if (requestRef.current) cancelAnimationFrame(requestRef.current);
        if (predictWebcamRef.current) {
          requestRef.current = requestAnimationFrame(predictWebcamRef.current);
        }
        setStatus("準備完了");
      } else {
        requestRef.current = requestAnimationFrame(checkVideoAndStart);
      }
    };
    if (requestRef.current) cancelAnimationFrame(requestRef.current);
    requestRef.current = requestAnimationFrame(checkVideoAndStart);
  }, [videoRef]);

  /** 計測開始 */
  const startMeasurement = useCallback(() => {
    capturedData.current              = [];
    calibrationLandmarkFrames.current = [];
    calibrationStatsRef.current       = null;
    calibrationStart.current          = performance.now();
    videoStart.current                = null;
    endingVideoStart.current          = null;
    setRemainingTime(null);

    const mVideo = measureVideoRef.current;
    const eVideo = endVideoRef.current;

    if (mVideo) {
      mVideo.muted = false;
      mVideo.play().then(() => { mVideo.pause(); mVideo.currentTime = 0; }).catch((e) => console.log(e));
    }
    if (eVideo) {
      eVideo.muted = false;
      eVideo.play().then(() => { eVideo.pause(); eVideo.currentTime = 0; }).catch((e) => console.log(e));
    }

    phaseRef.current = "calibrating";
    setPhase("calibrating");
    setIsFaceDetected(true);
    setCalibrationTime(CALIBRATION_DURATION);
    setStatus("キャリブレーション中...");
  }, [measureVideoRef, endVideoRef]);

  /** 最初の動画終了時 */
  const handleVideoEnd = useCallback(() => {
    if (phaseRef.current !== "watching_video") return;
    const mVideo = measureVideoRef.current;
    if (mVideo) { mVideo.pause(); mVideo.currentTime = 0; }
    const eVideo = endVideoRef.current;
    videoStart.current       = null;
    endingVideoStart.current = performance.now();
    phaseRef.current         = "ending_video";
    setPhase("ending_video");
    setRemainingTime(null);
    setStatus("リラックス動画再生中...");

    if (eVideo) {
      eVideo.muted = false;
      eVideo.currentTime = 0;
      eVideo.play().catch((err) => {
        console.error("リラックス動画再生エラー:", err);
        setPhase("idle");
        setStatus("再生エラー");
      });
    }
  }, [measureVideoRef, endVideoRef]);

  /** 終了動画終了時 → 自動保存・認証送信 */
  const handleEndVideoEnd = useCallback(async () => {
    if (phaseRef.current !== "ending_video") return;
    endingVideoStart.current = null;
    phaseRef.current         = "idle";
    setPhase("idle");
    setRemainingTime(null);

    if (capturedData.current.length === 0) {
      setStatus("計測データがありません");
      return;
    }

    if (!currentUser) {
      setStatus("エラー: 認証トークン未取得のため、安全のためにローカルへ保存します");
      downloadMeasurementJson(capturedData.current, calibrationStatsRef.current ?? undefined);
      return;
    }

    setStatus("データ保存・送信中...");
    try {
      if (storageRef.current) {
        await uploadMeasurement(
          storageRef.current,
          capturedData.current,
          calibrationStatsRef.current ?? undefined,
          currentUser.uid
        );
        const completeMessage = "保存＆送信 完了しました。ページを閉じてください。 ご協力ありがとうございました!!";
        setStatus(completeMessage);
        if (typeof window !== "undefined") {
          window.alert(completeMessage);
        }
      } else {
        throw new Error("Firebase Storageが初期化されていません");
      }
    } catch (err: unknown) {
      console.error("アップロードエラー:", err);
      setStatus("送信失敗: セキュリティエラーまたは通信障害。ローカルに保存します");
      downloadMeasurementJson(capturedData.current, calibrationStatsRef.current ?? undefined);
    }
  }, [currentUser]);

  /** 動画の残り時間を更新 */
  const handleTimeUpdate = useCallback((e: React.SyntheticEvent<HTMLVideoElement>) => {
    if (phaseRef.current !== "watching_video") return;
    const target = e.target as HTMLVideoElement;
    if (target.duration) setRemainingTime(target.duration - target.currentTime);
  }, []);

  const remainingTimeText = remainingTime != null ? formatTime(remainingTime) : "0:00";

  return {
    phase,
    isMeasuring: phase === "watching_video" || phase === "ending_video",
    isFaceDetected,
    remainingTime,
    remainingTimeText,
    calibrationTime,
    status,
    startLoop,
    startMeasurement,
    handleVideoEnd,
    handleEndVideoEnd,
    handleTimeUpdate,
  };
}