"use client";

import React, { useEffect, useRef, useState } from "react";
import { Html5Qrcode, Html5QrcodeScannerState } from "html5-qrcode";

interface QrScannerProps {
  onScanSuccess: (decodedText: string) => void;
  onError?: (error: string) => void;
}

export default function QrScanner({ onScanSuccess, onError }: QrScannerProps) {
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(true);
  const scannerRef = useRef<Html5Qrcode | null>(null);

  useEffect(() => {
    let isCancelled = false;
    let localScanner: Html5Qrcode | null = null;

    const stopAndReleaseCamera = async (scanner: Html5Qrcode | null) => {
      if (!scanner) return;
      try {
        const state = scanner.getState();
        if (
          state === Html5QrcodeScannerState.SCANNING ||
          state === Html5QrcodeScannerState.PAUSED
        ) {
          await scanner.stop();
        }
      } catch (e) {
        console.warn("Scanner stop warning:", e);
      }

      try {
        scanner.clear();
      } catch (e) {
        console.warn("Scanner clear warning:", e);
      }

      // 明示的にコンテナ内の全MediaStreamTrackを停止してカメラリソースを完全解放
      const container = document.getElementById("qr-reader");
      if (container) {
        const videoElements = container.getElementsByTagName("video");
        for (let i = 0; i < videoElements.length; i++) {
          const v = videoElements[i];
          if (v.srcObject && "getTracks" in (v.srcObject as MediaStream)) {
            (v.srcObject as MediaStream).getTracks().forEach((track) => track.stop());
            v.srcObject = null;
          }
        }
      }
    };

    const startScanner = async () => {
      setIsStarting(true);
      setCameraError(null);

      try {
        const scanner = new Html5Qrcode("qr-reader");
        localScanner = scanner;
        scannerRef.current = scanner;

        const config = {
          fps: 10,
          qrbox: { width: 220, height: 220 },
        };

        const handleSuccess = async (decodedText: string) => {
          if (isCancelled) return;
          isCancelled = true; // 重複スキャン防止

          // 次の画面（同意画面や計測画面）でカメラ重複（Device in use）が発生しないよう
          // 確実にストリームを停止・解放してからコールバックを呼ぶ
          await stopAndReleaseCamera(localScanner);
          onScanSuccess(decodedText);
        };

        const handleError = () => {
          // 各フレームごとの未検出エラーは無視
        };

        try {
          // まず環境カメラ（背面カメラ）を試す
          await scanner.start(
            { facingMode: "environment" },
            config,
            handleSuccess,
            handleError
          );
        } catch {
          if (isCancelled) {
            await stopAndReleaseCamera(scanner);
            return;
          }
          // 背面カメラがない（PC等）の場合はフロントカメラ/デフォルトカメラで試す
          await scanner.start(
            { facingMode: "user" },
            config,
            handleSuccess,
            handleError
          );
        }

        if (isCancelled) {
          await stopAndReleaseCamera(scanner);
        } else {
          setIsStarting(false);
        }
      } catch {
        if (!isCancelled) {
          setIsStarting(false);
          const msg =
            "カメラの起動に失敗しました。カメラの利用権限を許可するか、下のボタンからパスワード入力を行ってください。";
          setCameraError(msg);
          onError?.(msg);
        }
      }
    };

    startScanner();

    return () => {
      isCancelled = true;
      if (localScanner) {
        stopAndReleaseCamera(localScanner);
        scannerRef.current = null;
      }
    };
  }, [onScanSuccess, onError]);

  return (
    <div style={{ width: "100%", display: "flex", flexDirection: "column", alignItems: "center" }}>
      <div
        id="qr-reader"
        style={{
          width: "100%",
          maxWidth: "320px",
          minHeight: "260px",
          borderRadius: "8px",
          overflow: "hidden",
          backgroundColor: "#000",
          position: "relative",
        }}
      />

      {isStarting && !cameraError && (
        <p style={{ color: "#aaa", fontSize: "13px", marginTop: "12px" }}>
          カメラを起動しています...
        </p>
      )}

      {cameraError && (
        <p style={{ color: "#ff6b6b", fontSize: "13px", marginTop: "12px", textAlign: "center" }}>
          {cameraError}
        </p>
      )}
    </div>
  );
}
