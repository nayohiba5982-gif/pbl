// ============================================================
// ConcentrationApp.tsx
// UIのみを担当するメインコンポーネント（インフォームドコンセント追加版）
// ============================================================

"use client";

import React, { useRef, useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import { useFaceLandmarker } from "./hooks/useFaceLandmarker";
import { useWebcam }         from "./hooks/useWebcam";
import { useMeasurement }    from "./hooks/useMeasurement";

const mdComponents: Components = {
  h1: ({ children }) => <h1 className="text-2xl font-bold mt-4 mb-2">{children}</h1>,
  h2: ({ children }) => <h2 className="text-xl font-bold mt-3 mb-1">{children}</h2>,
  h3: ({ children }) => <h3 className="text-base font-bold mt-2 mb-1">{children}</h3>,
  p:  ({ children }) => <p  className="text-sm leading-relaxed mb-2">{children}</p>,
  li: ({ children }) => <li className="text-sm ml-4 list-disc">{children}</li>,
};

interface ConcentrationAppProps {
  experimentPassword: string;
}

export default function ConcentrationApp({ experimentPassword }: ConcentrationAppProps) {
  const canvasRef       = useRef<HTMLCanvasElement>(null);
  const measureVideoRef = useRef<HTMLVideoElement>(null);
  const endVideoRef     = useRef<HTMLVideoElement>(null);
  const containerRef    = useRef<HTMLDivElement>(null);
  const consentScrollRef = useRef<HTMLDivElement>(null); // ★追加: 同意書スクロール用Ref
  const introScrollRef  = useRef<HTMLDivElement>(null);

  // ── 画面遷移の状態 ────────────────────────────────────────
  const [consentAgreed,      setConsentAgreed]      = useState(false); // ★追加: 説明と同意のフェーズ管理
  const [agreed,             setAgreed]             = useState(false);
  const [introRead,          setIntroRead]          = useState(false);
  const [termsChecked,       setTermsChecked]       = useState(false);
  const [privacyChecked,     setPrivacyChecked]     = useState(false);
  const [consentScrolledToEnd, setConsentScrolledToEnd] = useState(false); // ★追加: 同意書スクロール末尾検知
  const [introScrolledToEnd, setIntroScrolledToEnd] = useState(false);

  // ── コンテンツ取得 ────────────────────────────────────────
  const [consentContent, setConsentContent] = useState(""); // ★追加: 同意書テキスト用
  const [termsContent,   setTermsContent]   = useState("");
  const [privacyContent, setPrivacyContent] = useState("");
  const [introContent,   setIntroContent]   = useState("");

  // ── 各種Markdownのfetch ──────────────────────────────────
  useEffect(() => {
    if (!experimentPassword) return;

    const fetchHeaders = {
      "x-experiment-password": experimentPassword,
    };

    // ★追加: informed_consent のフェッチ処理
    fetch("/api/content?file=informed_consent", { headers: fetchHeaders })
      .then((r) => r.json())
      .then((d) => setConsentContent(d.content ?? "取得できませんでした"));

    fetch("/api/content?file=terms", { headers: fetchHeaders })
      .then((r) => r.json())
      .then((d) => setTermsContent(d.content ?? "取得できませんでした"));
      
    fetch("/api/content?file=privacy", { headers: fetchHeaders })
      .then((r) => r.json())
      .then((d) => setPrivacyContent(d.content ?? "取得できませんでした"));
      
    fetch("/api/content?file=introduction", { headers: fetchHeaders })
      .then((r) => r.json())
      .then((d) => setIntroContent(d.content ?? "取得できませんでした"));
      
  }, [experimentPassword]);

  // ── モデル初期化・カメラ起動・計測ループ ──────────────────
  const { faceLandmarkerRef, isReady, error: modelError } = useFaceLandmarker();
  const { videoRef, isRunning, startCamera, error: cameraError } = useWebcam();
  const {
    phase,
    isMeasuring,
    isFaceDetected,
    calibrationTime,
    status,
    startLoop,
    startMeasurement,
    handleVideoEnd,
    handleEndVideoEnd,
    handleTimeUpdate,
  } = useMeasurement({ videoRef, canvasRef, measureVideoRef, endVideoRef, faceLandmarkerRef });

  const isWatchingVideo = phase === "watching_video";
  const isEndingVideo   = phase === "ending_video";
  const isSessionActive = phase !== "idle";

  const displayStatus = modelError ?? cameraError ?? status;
  const handleStartCamera = () => startCamera(startLoop);

  const handleStartMeasurement = () => {
    const mVideo = measureVideoRef.current;
    const eVideo = endVideoRef.current;
    if (mVideo) {
      mVideo.muted = false; mVideo.volume = 1.0;
      mVideo.play().then(() => { mVideo.pause(); mVideo.currentTime = 0; }).catch((err) => console.log(err));
    }
    if (eVideo) {
      eVideo.muted = false; eVideo.volume = 1.0;
      eVideo.play().then(() => { eVideo.pause(); eVideo.currentTime = 0; }).catch((err) => console.log(err));
    }
    startMeasurement();
  };

  useEffect(() => {
    if (isEndingVideo && endVideoRef.current) {
      const video = endVideoRef.current;
      video.muted = false; video.currentTime = 0;
      video.play().catch((err) => console.error(err));
    }
  }, [isEndingVideo, phase]);

  // ── スクロール位置のリセットと末尾検知 ──────────────────────
  // 同意書テキスト読み込み時のスクロール位置初期化
  useEffect(() => {
    if (consentScrollRef.current) {
      consentScrollRef.current.scrollTop = 0;
    }
  }, [consentContent]);

  // 導入（はじめに）テキスト読み込み時のスクロール位置初期化
  useEffect(() => {
    if (introScrollRef.current) {
      introScrollRef.current.scrollTop = 0;
    }
  }, [introContent]);

  const handleConsentScroll = () => {
    const el = consentScrollRef.current;
    if (!el) return;
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 8;
    if (atBottom) setConsentScrolledToEnd(true);
  };

  const handleIntroScroll = () => {
    const el = introScrollRef.current;
    if (!el) return;
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 8;
    if (atBottom) setIntroScrolledToEnd(true);
  };

  // ── 1. 説明と同意書 (Informed Consent) 画面 ── ★新規追加
  if (!consentAgreed) {
    return (
      <div key="screen-consent" className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-6">
        <h1 className="text-2xl font-bold mb-6">インフォームドコンセント</h1>
        <div className="w-full max-w-2xl bg-gray-800 rounded-lg p-4 flex flex-col">
          <div
            key="scroll-consent"
            ref={consentScrollRef}
            onScroll={handleConsentScroll}
            className="h-96 overflow-y-auto bg-gray-900 rounded p-4 text-gray-300"
          >
            {consentContent ? (
              <ReactMarkdown components={mdComponents}>{consentContent}</ReactMarkdown>
            ) : (
              <p className="text-gray-400 text-center py-8">同意書を読み込み中...</p>
            )}
          </div>
          {!consentScrolledToEnd ? (
            <p className="text-center text-sm text-gray-400 mt-3 animate-pulse">
              ↓ 最後までスクロールしてください
            </p>
          ) : (
            <p className="text-center text-sm text-green-400 mt-3 font-medium">
              ✓ 最後まで確認しました
            </p>
          )}
          <button
            onClick={() => setConsentAgreed(true)}
            disabled={!consentScrolledToEnd}
            className="mt-4 w-full py-3 bg-blue-600 hover:bg-blue-500 rounded-lg font-bold text-lg disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            内容を理解して同意する
          </button>
        </div>
      </div>
    );
  }

  // ── 2. 利用規約とプライバシーポリシー同意画面 ──────────────────
  if (!agreed) {
    return (
      <div key="screen-terms" className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-6">
        <h1 className="text-2xl font-bold mb-6">Data Collect</h1>
        <div className="w-full max-w-2xl space-y-6">
          <div className="bg-gray-800 rounded-lg p-4">
            <h2 className="text-lg font-bold mb-3">利用規約</h2>
            <div className="h-48 overflow-y-auto bg-gray-900 rounded p-3 text-gray-300">
              <ReactMarkdown components={mdComponents}>{termsContent}</ReactMarkdown>
            </div>
            <label className="flex items-center gap-2 mt-3 cursor-pointer">
              <input type="checkbox" checked={termsChecked}
                onChange={(e) => setTermsChecked(e.target.checked)}
                className="w-4 h-4 accent-orange-500" />
              <span className="text-sm">利用規約に同意する</span>
            </label>
          </div>
          <div className="bg-gray-800 rounded-lg p-4">
            <h2 className="text-lg font-bold mb-3">プライバシーポリシー</h2>
            <div className="h-48 overflow-y-auto bg-gray-900 rounded p-3 text-gray-300">
              <ReactMarkdown components={mdComponents}>{privacyContent}</ReactMarkdown>
            </div>
            <label className="flex items-center gap-2 mt-3 cursor-pointer">
              <input type="checkbox" checked={privacyChecked}
                onChange={(e) => setPrivacyChecked(e.target.checked)}
                className="w-4 h-4 accent-orange-500" />
              <span className="text-sm">プライバシーポリシーに同意する</span>
            </label>
          </div>
          <button
            onClick={() => setAgreed(true)}
            disabled={!termsChecked || !privacyChecked}
            className="w-full py-3 bg-orange-600 hover:bg-orange-500 rounded-lg font-bold text-lg disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            ☑ 同意して開始する
          </button>
        </div>
      </div>
    );
  }

  // ── 3. 導入（はじめに）画面 ──────────────────────────────
  if (!introRead) {
    return (
      <div key="screen-intro" className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-6">
        <h1 className="text-2xl font-bold mb-6">はじめに</h1>
        <div className="w-full max-w-2xl bg-gray-800 rounded-lg p-4 flex flex-col">
          <div
            key="scroll-intro"
            ref={introScrollRef}
            onScroll={handleIntroScroll}
            className="h-96 overflow-y-auto bg-gray-900 rounded p-4 text-gray-300"
          >
            {introContent ? (
              <ReactMarkdown components={mdComponents}>{introContent}</ReactMarkdown>
            ) : (
              <p className="text-gray-400 text-center py-8">説明文を読み込み中...</p>
            )}
          </div>
          {!introScrolledToEnd ? (
            <p className="text-center text-sm text-gray-400 mt-3 animate-pulse">
              ↓ 最後までスクロールしてください
            </p>
          ) : (
            <p className="text-center text-sm text-green-400 mt-3 font-medium">
              ✓ 最後まで確認しました
            </p>
          )}
          <button
            onClick={() => setIntroRead(true)}
            disabled={!introScrolledToEnd}
            className="mt-4 w-full py-3 bg-orange-600 hover:bg-orange-500 rounded-lg font-bold text-lg disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            全て読みました
          </button>
        </div>
      </div>
    );
  }

  const isModelLoading = !isReady;

  // ── 4. メイン画面 ────────────────────────────────────────────
  return (
    <div className="flex flex-col items-center p-5 bg-gray-900 min-h-screen text-white">
      <h1 className="text-2xl font-bold mb-4">Data Collect</h1>
      <p className="mb-4">{displayStatus}</p>

      <div className="flex items-center gap-4 w-full max-w-5xl">
        <div ref={containerRef} className="relative flex-1 bg-black rounded-lg overflow-hidden border-2 border-gray-700 aspect-video">
          <video ref={videoRef} autoPlay playsInline muted className="absolute top-0 left-0 w-full h-full object-cover z-0" />
          <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full pointer-events-none z-10" />
          <video ref={measureVideoRef} src="/class_long.mp4" playsInline preload="auto" onEnded={handleVideoEnd} onTimeUpdate={handleTimeUpdate}  /*動画指定*/
            className={`absolute top-0 left-0 w-full h-full object-cover z-20 ${isWatchingVideo ? "opacity-100" : "opacity-0 pointer-events-none"}`}
          />
          <video ref={endVideoRef} src="/campfire.mp4" playsInline preload="auto" onEnded={handleEndVideoEnd}
            className={`absolute top-0 left-0 w-full h-full object-cover z-30 ${isEndingVideo ? "opacity-100" : "opacity-0 pointer-events-none"}`}
          />

          {phase === "calibrating" && (
            <div className="absolute top-0 left-0 w-full h-full z-30 pointer-events-none flex flex-col items-center justify-center gap-4 bg-black/40">
              <p className="text-2xl font-bold text-white">画面を見て目を開けてください</p>
              <div className="w-48 h-3 bg-gray-600 rounded-full overflow-hidden">
                <div className="h-full bg-green-400 rounded-full transition-all duration-100" style={{ width: `${((5 - calibrationTime) / 5) * 100}%` }} />
              </div>
              <p className="text-lg font-mono text-green-300">{Math.ceil(calibrationTime)} 秒</p>
            </div>
          )}

          {isMeasuring && !isFaceDetected && (
            <div className="absolute top-0 left-0 w-full h-full z-40 pointer-events-none flex items-center justify-center">
              <div className="bg-red-600/90 text-white px-8 py-5 rounded-xl text-2xl font-bold animate-pulse shadow-lg text-center">
                ⚠️ 警告<br /><span className="text-lg font-normal">顔が認識できません。カメラの正面を見てください。</span>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="mt-6 space-x-4">
        <button onClick={handleStartCamera} disabled={isRunning || isModelLoading} className="px-6 py-2 bg-blue-600 hover:bg-blue-500 rounded disabled:opacity-50 disabled:cursor-not-allowed transition">
          {isModelLoading ? "AIモデル読み込み中..." : "カメラ起動"}
        </button>
        <button onClick={handleStartMeasurement} disabled={!isRunning || isSessionActive} className="px-6 py-2 bg-orange-600 hover:bg-orange-500 rounded disabled:opacity-50 transition">
          計測開始
        </button>
      </div>
    </div>
  );
}