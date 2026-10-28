// ============================================================
// app/api/content/route.ts
// パスワード保護されたマークダウンファイルを返すAPIルート（informed_consent追加版）
// ============================================================

import { NextRequest, NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";
import crypto from "crypto";

// 簡易インメモリ・レートリミッター（ブルートフォース攻撃対策）
interface RateLimitRecord {
  count: number;
  resetTime: number;
}
const rateLimitMap = new Map<string, RateLimitRecord>();
const RATE_LIMIT_WINDOW_MS = 60 * 1000; // 1分間
const MAX_ATTEMPTS_PER_WINDOW = 15; // 1分間に最大15回

function isRateLimited(ip: string): boolean {
  const now = Date.now();

  // マップが肥大化した際の古いエントリ一括クリーンアップ
  if (rateLimitMap.size > 5000) {
    for (const [key, record] of rateLimitMap.entries()) {
      if (now > record.resetTime) {
        rateLimitMap.delete(key);
      }
    }
  }

  const record = rateLimitMap.get(ip);
  if (!record || now > record.resetTime) {
    rateLimitMap.set(ip, { count: 1, resetTime: now + RATE_LIMIT_WINDOW_MS });
    return false;
  }

  if (record.count >= MAX_ATTEMPTS_PER_WINDOW) {
    return true;
  }

  record.count += 1;
  return false;
}

export async function GET(req: NextRequest) {
  // 🔒 0. レート制限のチェック
  const ip =
    req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ||
    req.headers.get("x-real-ip") ||
    "unknown";

  if (isRateLimited(ip)) {
    return NextResponse.json(
      { error: "Too many requests. Please try again later." },
      { status: 429, headers: { "Retry-After": "60" } }
    );
  }

  // 🔒 1. パスワードの取得と入力文字数上限チェック
  const clientPassword = req.headers.get("x-experiment-password") || "";
  const correctPassword = process.env.EXPERIMENT_PASSWORD || "";

  // 64文字以上の入力は不正なリクエストとして即時400エラー
  if (clientPassword.length > 64) {
    return NextResponse.json(
      { error: "Bad Request: Password exceeds maximum allowed length" },
      { status: 400 }
    );
  }

  // 環境変数未設定の場合は401
  if (!correctPassword) {
    return NextResponse.json(
      { error: "Unauthorized: Invalid password" },
      { status: 401 }
    );
  }

  // タイミング攻撃を防ぐ定数時間比較（バイト長の一致を確認して例外を防止）
  const a = Buffer.from(clientPassword);
  const b = Buffer.from(correctPassword);
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) {
    return NextResponse.json(
      { error: "Unauthorized: Invalid password" },
      { status: 401 }
    );
  }

  // 2. ファイル名のバリデーション
  const { searchParams } = new URL(req.url);
  const file = searchParams.get("file");

  // ★修正: informed_consent も許可リストに追加しました
  if (
    file !== "terms" && 
    file !== "privacy" && 
    file !== "introduction" && 
    file !== "informed_consent"
  ) {
    return NextResponse.json({ error: "invalid file" }, { status: 400 });
  }

  // 3. ファイルの読み込みと返却
  // これにより、プロジェクト直下の `content/informed_consent.md` が読み込まれます
  const filePath = path.join(process.cwd(), "content", `${file}.md`);

  try {
    const content = await fs.readFile(filePath, "utf-8");
    return NextResponse.json({ content });
  } catch {
    return NextResponse.json({ error: "file not found" }, { status: 404 });
  }
}