"use client";

import { useState } from "react";
import ConcentrationApp from "./ConcentrationApp";
import QrScanner from "./components/QrScanner";

export default function Page() {
  const [authMode, setAuthMode] = useState<"qr" | "password">("qr");
  const [password, setPassword] = useState("");
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [error, setError] = useState("");
  const [isVerifying, setIsVerifying] = useState(false);
  const [scannerKey, setScannerKey] = useState(0);

  // 🔒 パスワードまたはQRコードの文字列を検証する共通関数
  const verifyCredential = async (credential: string) => {
    setIsVerifying(true);
    setError("");

    try {
      // api/content ルートに introduction を取得しにいく形でパスワード検証を行う
      const res = await fetch(`/api/content?file=introduction`, {
        method: "GET",
        headers: {
          "x-experiment-password": credential,
        },
      });

      if (res.ok) {
        setPassword(credential);
        setIsAuthenticated(true);
        setError("");
      } else if (res.status === 429) {
        setError("短時間の試行回数が上限を超えました。しばらく待ってから再度お試しください。");
      } else if (res.status === 400) {
        setError("入力形式が無効です（文字数が上限を超えています）。");
      } else {
        setError("認証に失敗しました。正しいQRコードまたはパスワードを入力してください。");
      }
    } catch {
      setError("通信エラーが発生しました。");
    } finally {
      setIsVerifying(false);
    }
  };

  // パスワード入力フォーム送信
  const handlePasswordSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!password.trim()) {
      setError("パスワードを入力してください。");
      return;
    }
    await verifyCredential(password.trim());
  };

  // QRコード読み取り成功時
  const handleQrScanSuccess = async (decodedText: string) => {
    await verifyCredential(decodedText.trim());
  };

  // 1. 認証済みの場合：同意画面（ConcentrationApp）へ遷移
  if (isAuthenticated) {
    return <ConcentrationApp experimentPassword={password} />;
  }

  // 2. 未認証の場合：QRコード読み取り または パスワード入力 画面を表示
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "100vh",
        backgroundColor: "#121212",
        color: "#fff",
        fontFamily: "sans-serif",
        padding: "20px",
      }}
    >
      <div
        style={{
          padding: "36px 28px",
          borderRadius: "14px",
          backgroundColor: "#1e1e1e",
          boxShadow: "0 8px 24px rgba(0,0,0,0.5)",
          textAlign: "center",
          maxWidth: "420px",
          width: "100%",
        }}
      >
        <h2 style={{ marginBottom: "8px", fontSize: "24px", fontWeight: "bold" }}>
          実験参加者ログイン
        </h2>

        {authMode === "qr" ? (
          <div>
            <p style={{ color: "#aaa", fontSize: "14px", marginBottom: "20px" }}>
              参加用QRコードをカメラにかざしてください。
            </p>

            {isVerifying ? (
              <div style={{ padding: "40px 0" }}>
                <p style={{ fontSize: "16px", color: "#60a5fa", fontWeight: "bold" }}>
                  認証中...
                </p>
                <p style={{ fontSize: "13px", color: "#888", marginTop: "8px" }}>
                  QRコードの情報を照合しています
                </p>
              </div>
            ) : (
              <div>
                <QrScanner
                  key={scannerKey}
                  onScanSuccess={handleQrScanSuccess}
                  onError={(err) => setError(err)}
                />

                {error && (
                  <button
                    onClick={() => {
                      setError("");
                      setScannerKey((k) => k + 1);
                    }}
                    style={{
                      marginTop: "16px",
                      padding: "8px 16px",
                      borderRadius: "6px",
                      border: "1px solid #555",
                      backgroundColor: "#2a2a2a",
                      color: "#eee",
                      fontSize: "13px",
                      cursor: "pointer",
                    }}
                  >
                    もう一度読み取る
                  </button>
                )}
              </div>
            )}

            <div style={{ marginTop: "24px", paddingTop: "20px", borderTop: "1px solid #333" }}>
              <button
                type="button"
                onClick={() => {
                  setError("");
                  setAuthMode("password");
                }}
                disabled={isVerifying}
                style={{
                  width: "100%",
                  padding: "12px",
                  borderRadius: "6px",
                  border: "1px solid #444",
                  backgroundColor: "#2a2a2a",
                  color: "#fff",
                  fontSize: "15px",
                  fontWeight: "600",
                  cursor: isVerifying ? "not-allowed" : "pointer",
                  transition: "background 0.2s",
                }}
                onMouseEnter={(e) => {
                  if (!isVerifying) e.currentTarget.style.backgroundColor = "#383838";
                }}
                onMouseLeave={(e) => {
                  if (!isVerifying) e.currentTarget.style.backgroundColor = "#2a2a2a";
                }}
              >
                パスワード入力でログイン
              </button>
            </div>
          </div>
        ) : (
          <div>
            <p style={{ color: "#aaa", fontSize: "14px", marginBottom: "20px" }}>
              許可された被験者用の合言葉を入力してください。
            </p>

            <form onSubmit={handlePasswordSubmit} style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              <input
                type="password"
                placeholder="パスワードを入力"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={isVerifying}
                style={{
                  padding: "12px",
                  borderRadius: "6px",
                  border: "1px solid #333",
                  backgroundColor: "#2a2a2a",
                  color: "#fff",
                  fontSize: "16px",
                  outline: "none",
                }}
              />
              <button
                type="submit"
                disabled={isVerifying}
                style={{
                  padding: "12px",
                  borderRadius: "6px",
                  border: "none",
                  backgroundColor: isVerifying ? "#555" : "#0070f3",
                  color: "#fff",
                  fontSize: "16px",
                  fontWeight: "bold",
                  cursor: isVerifying ? "not-allowed" : "pointer",
                  transition: "background 0.2s",
                }}
                onMouseEnter={(e) => {
                  if (!isVerifying) e.currentTarget.style.backgroundColor = "#0051a8";
                }}
                onMouseLeave={(e) => {
                  if (!isVerifying) e.currentTarget.style.backgroundColor = "#0070f3";
                }}
              >
                {isVerifying ? "認証中..." : "認証して入場"}
              </button>
            </form>

            <div style={{ marginTop: "24px", paddingTop: "20px", borderTop: "1px solid #333" }}>
              <button
                type="button"
                onClick={() => {
                  setError("");
                  setAuthMode("qr");
                }}
                disabled={isVerifying}
                style={{
                  width: "100%",
                  padding: "12px",
                  borderRadius: "6px",
                  border: "1px solid #444",
                  backgroundColor: "#2a2a2a",
                  color: "#fff",
                  fontSize: "15px",
                  fontWeight: "600",
                  cursor: isVerifying ? "not-allowed" : "pointer",
                  transition: "background 0.2s",
                }}
                onMouseEnter={(e) => {
                  if (!isVerifying) e.currentTarget.style.backgroundColor = "#383838";
                }}
                onMouseLeave={(e) => {
                  if (!isVerifying) e.currentTarget.style.backgroundColor = "#2a2a2a";
                }}
              >
                QRコード読み取りに戻る
              </button>
            </div>
          </div>
        )}

        {error && (
          <p
            style={{
              color: "#ff4d4d",
              marginTop: "16px",
              fontSize: "14px",
              fontWeight: "500",
              lineHeight: "1.4",
            }}
          >
            {error}
          </p>
        )}
      </div>
    </div>
  );
}