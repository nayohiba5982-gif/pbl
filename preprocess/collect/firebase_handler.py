import os
from pathlib import Path
import firebase_admin
from firebase_admin import credentials, storage
from google.cloud.storage import Bucket
from config_loader import cfg, log

# config.yaml から設定を取得
KEY_PATH = cfg["firebase"]["key_path"]
STORAGE_BUCKET = cfg["firebase"]["storage_bucket"]
STORAGE_PREFIX = cfg["firebase"]["storage_prefix"]
WORK_DIR = cfg["firebase"]["work_dir"]

def init_firebase() -> Bucket:
    """Firebase Admin SDK を初期化して Storage バケットオブジェクトを返す。"""
    if not KEY_PATH.exists():
        raise FileNotFoundError(
            f"鍵ファイルが見つかりません: {KEY_PATH}\n"
            "ファイルを同じディレクトリに配置してください。"
        )

    if not firebase_admin._apps:
        cred = credentials.Certificate(str(KEY_PATH))
        firebase_admin.initialize_app(cred, {"storageBucket": STORAGE_BUCKET})
        log.info("Firebase 初期化完了")

    return storage.bucket()

def download_gz_files(bucket: Bucket) -> list[Path]:
    """
    user_signals/{user_id}/*.gz を探索し、ローカルの WORK_DIR に保存する。
    """
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    prefix = STORAGE_PREFIX.rstrip("/") + "/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    gz_blobs = [b for b in blobs if b.name.endswith(".gz")]

    if not gz_blobs:
        log.warning(f"Storage に .gz ファイルが見つかりませんでした (prefix={prefix})")
        return []

    log.info(f"Storage全体から {len(gz_blobs)} 件の .gz ファイルを検出しました")
    downloaded: list[Path] = []

    for blob in gz_blobs:
        filename = Path(blob.name).name

        if filename.startswith("pbl-b18-"):
            new_filename = filename.replace("pbl-b18-", "", 1)
        else:
            new_filename = filename

        local_path = WORK_DIR / new_filename

        if local_path.exists():
            log.info(f"  スキップ（既存）: {new_filename}")
            downloaded.append(local_path)
            continue

        log.info(f"  ダウンロード中: {filename} ➔ {new_filename} ...")
        blob.download_to_filename(str(local_path))
        log.info(f"  ✓ 保存: {local_path}")
        downloaded.append(local_path)

    return downloaded