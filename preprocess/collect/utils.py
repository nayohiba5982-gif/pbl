import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from config_loader import  log,  ZIP_OUTPUT, WORK_DIR

def pack_json_to_zip(json_files: list[Path]) -> Path | None:
    """json_files を単一の ZIP ファイルに圧縮して ZIP_OUTPUT に保存する。"""
    if not json_files:
        log.warning("ZIP に含める JSON ファイルがありません。")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = ZIP_OUTPUT / f"measurements_{timestamp}.zip"

    log.info(f"ZIP 作成中: {zip_path} ...")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for json_path in json_files:
            zf.write(json_path, arcname=json_path.name)
            log.info(f"  追加: {json_path.name}")

    log.info(f"✓ ZIP 完成: {zip_path}  ({zip_path.stat().st_size / 1024:.1f} KB)")
    return zip_path

def cleanup_work_dir():
    """一時フォルダ内のダウンロード済み .gz ファイルのみを安全に削除する。"""
    if WORK_DIR.exists():
        gz_files = list(WORK_DIR.glob("*.gz")) + list(WORK_DIR.glob("*.GZ"))
        for gz_file in gz_files:
            gz_file.unlink(missing_ok=True)
        log.info(f"一時ダウンロードファイル (.gz) を安全に削除しました ({len(gz_files)} 件): {WORK_DIR}")