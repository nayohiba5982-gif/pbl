// ============================================================
// lib/firebase.ts
// Firebase初期化、認証、およびgzip圧縮データ送信ユーティリティ
// ============================================================

import { initializeApp, getApps } from "firebase/app";
import { getStorage, ref, uploadBytes, FirebaseStorage } from "firebase/storage";
import { getAuth } from "firebase/auth";
import type { CalibrationStats, CapturedFrame } from "../types";

/** Firebase StorageおよびAuthのインスタンスを返す */
const firebaseConfig = {
  apiKey:             process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain:         process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId:          process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  storageBucket:      process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
  messagingSenderId:  process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
  appId:              process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
};

const app = getApps().length === 0 ? initializeApp(firebaseConfig) : getApps()[0];

export function getFirebaseStorage(): FirebaseStorage | null {
  try {
    return getStorage(app);
  } catch (e) {
    console.error("Firebase Storage初期化エラー:", e);
    return null;
  }
}

export function getFirebaseAuth() {
  try {
    return getAuth(app);
  } catch (e) {
    console.error("Firebase Auth初期化エラー:", e);
    return null;
  }
}

/**
 * 💡 [新規追加] 指定された長さの完全ランダムな英数字文字列を生成する
 * 対象文字: アルファベット大文字、小文字、数字 (A-Z, a-z, 0-9)
 */
function generateRandomString(length: number = 16): string {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let result = "";
  
  // ブラウザのより安全な暗号ランダムAPIを使用（SSRエラー防止のためwindowチェック入り）
  if (typeof window !== "undefined" && window.crypto && window.crypto.getRandomValues) {
    const array = new Uint32Array(length);
    window.crypto.getRandomValues(array);
    for (let i = 0; i < length; i++) {
      result += chars[array[i] % chars.length];
    }
  } else {
    // フォールバック（予備のランダム処理）
    for (let i = 0; i < length; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
  }
  return result;
}

/**
 * JSON文字列をgzip圧縮してUint8Arrayに変換する
 */
async function gzipJson(data: object): Promise<Uint8Array> {
  const json   = JSON.stringify(data);
  const stream = new Blob([json]).stream().pipeThrough(new CompressionStream("gzip"));
  const buffer = await new Response(stream).arrayBuffer();
  return new Uint8Array(buffer);
}




/**
 * 計測データをgzip圧縮してFirebase Storageにアップロードする
 */
export async function uploadMeasurement(
  storage: FirebaseStorage,
  frames: CapturedFrame[],
  calibration: CalibrationStats | undefined,
  uid: string
): Promise<void> {





  const clientCreatedAtMs = Date.now();
  const fileData = {
    clientCreatedAtMs,
    calibration,
    data: frames,
  };

  const compressed = await gzipJson(fileData);
  
  // ── 💡 ファイル名を完全ランダムな英数字に変更 ──────────────────────────────
  // 大文字・小文字・数字が混ざった10文字のランダムファイル名を生成
  const randomFileName = generateRandomString(10);
  
  // 生成されるパスの例: `user_signals/【UID】/aB3dE6gH1jK4mN7p.json.gz`
  const storageRef = ref(storage, `user_signals/${uid}/pbl-b18-${randomFileName}.json.gz`);
  // ─────────────────────────────────────────────────────────────────────────

  await uploadBytes(storageRef, compressed, {
    contentType:     "application/json",
    contentEncoding: "gzip",
  });
}

/**
 * 計測データをJSONファイルとしてローカルにダウンロードする
 */
export function downloadMeasurementJson(
  frames: CapturedFrame[],
  calibration?: CalibrationStats
): void {

  const clientCreatedAtMs = Date.now();
  const fileData = {
    clientCreatedAtMs,
    timestamp:  new Date().toISOString(),
    calibration,
    data:       frames,
  };
  const blob = new Blob([JSON.stringify(fileData, null, 2)], { type: "application/json" });
  const url  = URL.createObjectURL(blob);

  const a    = document.createElement("a");
  a.href     = url;
  
  // 💡 ローカル保存用のバックアップファイル名も、完全ランダム英数字に変更して統一
  const randomFileName = generateRandomString(16);
  a.download = `pbl-b18-${randomFileName}.json`;

  a.click();
  URL.revokeObjectURL(url);
}