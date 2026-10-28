import gzip
import json
import csv
import traceback
from pathlib import Path
import sys

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from config_loader import WORK_DIR, OUTPUT_DIR, CSV_OUTPUT_DIR, log

# =========================================================
# ★ 対象ファイルの指定
# TARGET_PATTERNS = ["04-12"]  # 特定キーワードで絞り込む場合
TARGET_PATTERNS = ['sTfnnnrGq4.json.gz', "EPu3DMEDU3.json.gz", "H6SaVKMe91.json.gz"]           # 空にすると preprocess/data 内の全 .gz を処理
# =========================================================

def flatten_frame_robust(frame: dict) -> dict:
    if not isinstance(frame, dict):
        return {}

    row = {}
    row['time'] = frame.get('time')
    row['clientEpochMs'] = frame.get('clientEpochMs')
    row['clientPerfMs'] = frame.get('clientPerfMs')
    row['label'] = frame.get('label')
    
    # 1. rPPG
    rppg = frame.get('rppg')
    if isinstance(rppg, dict):
        for key, rgb in rppg.items():
            if isinstance(rgb, dict):
                row[f'rppg_{key}_r'] = rgb.get('r')
                row[f'rppg_{key}_g'] = rgb.get('g')
                row[f'rppg_{key}_b'] = rgb.get('b')
            
    # 2. Landmarks
    landmarks = frame.get('landmarks')
    if isinstance(landmarks, dict):
        for category, lm_dict in landmarks.items():
            if isinstance(lm_dict, dict):
                for lm_key, coord in lm_dict.items():
                    clean_key = str(lm_key).replace("lm[", "").replace("]", "")
                    if isinstance(coord, dict):
                        row[f'lm_{category}_{clean_key}_x'] = coord.get('x')
                        row[f'lm_{category}_{clean_key}_y'] = coord.get('y')
                        row[f'lm_{category}_{clean_key}_z'] = coord.get('z')
            elif isinstance(lm_dict, list):
                for idx, coord in enumerate(lm_dict):
                    if isinstance(coord, dict):
                        row[f'lm_{category}_{idx}_x'] = coord.get('x')
                        row[f'lm_{category}_{idx}_y'] = coord.get('y')
                        row[f'lm_{category}_{idx}_z'] = coord.get('z')
    elif isinstance(landmarks, list):
        for idx, coord in enumerate(landmarks):
            if isinstance(coord, dict):
                row[f'lm_{idx}_x'] = coord.get('x')
                row[f'lm_{idx}_y'] = coord.get('y')
                row[f'lm_{idx}_z'] = coord.get('z')

    # 3. Centroid
    centroid = frame.get('landmarkCentroid')
    if isinstance(centroid, dict):
        row['centroid_x'] = centroid.get('x')
        row['centroid_y'] = centroid.get('y')
        row['centroid_z'] = centroid.get('z')
        
    # 4. Blendshapes
    blendshapes = frame.get('blendshapes')
    if isinstance(blendshapes, dict):
        for bs_name, value in blendshapes.items():
            row[f'bs_{bs_name}'] = value
    elif isinstance(blendshapes, list):
        for item in blendshapes:
            if isinstance(item, dict):
                name = item.get('categoryName') or item.get('name') or item.get('label')
                val = item.get('score') or item.get('value')
                if name:
                    row[f'bs_{name}'] = val

    return row

def process_single_gz(gz_path: Path):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stem = gz_path.stem
    json_name = stem if stem.endswith(".json") else stem + ".json"
    json_path = OUTPUT_DIR / json_name
    csv_path = CSV_OUTPUT_DIR / (json_path.stem + ".csv")

    log.info(f"⏳ 処理開始: {gz_path.name}")

    # 解凍
    try:
        try:
            with gzip.open(gz_path, "rb") as f_in:
                raw_bytes = f_in.read()
            raw_str = raw_bytes.decode("utf-8")
        except OSError:
            with open(gz_path, "r", encoding="utf-8") as f_in:
                raw_str = f_in.read()

        parsed_json = json.loads(raw_str)
        with open(json_path, "w", encoding="utf-8") as f_out:
            json.dump(parsed_json, f_out, ensure_ascii=False, indent=2)

    except Exception as e:
        log.error(f"❌ 解凍/JSONパースエラー [{gz_path.name}]: {e}")
        log.error(traceback.format_exc())
        return False

    # CSV変換
    try:
        if isinstance(parsed_json, dict):
            frames = parsed_json.get("data", [])
        elif isinstance(parsed_json, list):
            frames = parsed_json
        else:
            frames = []

        if not frames:
            log.warning(f"⚠️ スキップ: 'data' 配列が空です [{json_path.name}]")
            return False

        flattened_rows = [flatten_frame_robust(f) for f in frames if isinstance(f, dict)]
        flattened_rows = [r for r in flattened_rows if r]

        if not flattened_rows:
            log.warning(f"⚠️ スキップ: 有効なフレームがありません [{json_path.name}]")
            return False

        fieldnames = []
        for r in flattened_rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(csv_path, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened_rows)

        log.info(f"  ✓ CSV生成成功: {csv_path.name} ({len(flattened_rows)} 行)")
        return True

    except Exception as e:
        log.error(f"❌ CSV変換エラー [{gz_path.name}]: {e}")
        log.error(traceback.format_exc())
        return False

def main():
    print("=" * 50)
    print("  Ubuntu ローカル .gz 指定CSV変換")
    print("=" * 50)
    print(f"📂 参照ディレクトリ: {WORK_DIR.resolve()}")

    if not WORK_DIR.exists():
        log.error(f"❌ ディレクトリが存在しません: {WORK_DIR.resolve()}")
        return

    # Ubuntu環境向けに .gz と .GZ の両方を検索
    all_gz_files = sorted(
        list(WORK_DIR.glob("*.gz")) + list(WORK_DIR.glob("**/*.gz")) +
        list(WORK_DIR.glob("*.GZ")) + list(WORK_DIR.glob("**/*.GZ"))
    )
    # 重複除去
    all_gz_files = list(dict.fromkeys(all_gz_files))

    if not all_gz_files:
        log.warning(f"⚠️ '{WORK_DIR.resolve()}' 内に .gz ファイルが見つかりませんでした。")
        # フォルダ内にあるファイル・ディレクトリを一覧表示（確認用）
        items = list(WORK_DIR.iterdir())
        log.info(f"フォルダ直下の要素 ({len(items)} 個): {[i.name for i in items[:10]]}")
        return

    # キーワードフィルタリング
    if TARGET_PATTERNS:
        target_gz_files = [
            f for f in all_gz_files 
            if any(pattern in f.name for pattern in TARGET_PATTERNS)
        ]
    else:
        target_gz_files = all_gz_files

    log.info(f"全 {len(all_gz_files)} 件中、対象 {len(target_gz_files)} 件の .gz ファイルを検出しました:\n")

    success_count = 0
    for gz_path in target_gz_files:
        if process_single_gz(gz_path):
            success_count += 1

    print("\n" + "=" * 50)
    print("  処理完了サマリー")
    print("=" * 50)
    print(f"  成功: {success_count} / {len(target_gz_files)} 件")
    print(f"  CSV 保存先: {CSV_OUTPUT_DIR.resolve()}/")
    print("=" * 50)

if __name__ == "__main__":
    main()