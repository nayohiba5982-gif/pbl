import gzip
import json
import csv
import traceback
from pathlib import Path
from config_loader import cfg, log

# config.yaml から設定を取得
OUTPUT_DIR = cfg["firebase"]["output_dir"]
CSV_OUTPUT_DIR = cfg["firebase"]["csv_output_dir"]
DELETE_GZ_AFTER_EXTRACT = cfg["firebase"]["delete_gz_after_extract"]


def flatten_dict(d: dict, parent_key: str = '', sep: str = '_') -> dict:
    """辞書を再帰的にすべて平坦化する汎用関数"""
    items = []
    for k, v in d.items():
        clean_k = str(k).replace("lm[", "").replace("]", "")
        new_key = f"{parent_key}{sep}{clean_k}" if parent_key else clean_k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def extract_gz_to_json(gz_files: list[Path]) -> list[Path]:
    """各 .gz ファイルを解凍し、JSONとして保存する"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_files: list[Path] = []

    for gz_path in gz_files:
        stem = gz_path.stem
        json_name = stem if stem.endswith(".json") else stem + ".json"
        json_path = OUTPUT_DIR / json_name

        if json_path.exists():
            log.info(f"  スキップ（既存 JSON）: {json_path.name}")
            json_files.append(json_path)
            if DELETE_GZ_AFTER_EXTRACT:
                gz_path.unlink(missing_ok=True)
            continue

        try:
            log.info(f"  解凍中: {gz_path.name} ...")
            try:
                with gzip.open(gz_path, "rb") as f_in:
                    raw_bytes = f_in.read()
                raw_str = raw_bytes.decode("utf-8")
            except OSError as e:
                if "Not a gzipped file" in str(e):
                    log.info(f"中身はすでに解凍済みのテキスト（JSON）です。そのまま読み込みます。")
                    with open(gz_path, "r", encoding="utf-8") as f_in:
                        raw_str = f_in.read()
                else:
                    raise e

            try:
                parsed = json.loads(raw_str)
                with open(json_path, "w", encoding="utf-8") as f_out:
                    json.dump(parsed, f_out, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                log.warning(f"  JSON パース失敗 — テキストとして保存: {json_name}")
                json_path = OUTPUT_DIR / (stem + ".txt")
                json_path.write_text(raw_str, encoding="utf-8")

            log.info(f"  ✓ 取得完了: {json_path.name}")
            json_files.append(json_path)

            if DELETE_GZ_AFTER_EXTRACT:
                gz_path.unlink(missing_ok=True)
                log.info(f"  🗑  一時ファイルを削除しました")

        except Exception as e:
            log.error(f"  ✗ 処理エラー [{gz_path.name}]: {e}")

    return json_files


def flatten_frame(frame: dict) -> dict:
    """CapturedFrame 型のデータを堅牢に平坦化する（local_gz_csv.py の処理を統合）"""
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


def convert_json_to_csv(json_path: Path) -> Path | None:
    """JSON ファイルを解析し、時系列の特徴量テーブルとして CSV 保存する。"""
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CSV_OUTPUT_DIR / (json_path.stem + ".csv")
    
    if csv_path.exists():
        log.info(f"  スキップ（既存 CSV）: {csv_path.name}")
        return csv_path

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            record = json.load(f)
            
        # JSONの構造に合わせて柔軟にフレームを取得
        if isinstance(record, dict):
            frames = record.get("data", [])
        elif isinstance(record, list):
            frames = record
        else:
            frames = []

        if not frames:
            log.warning(f"  CSV 変換スキップ: 有効な 'data' 配列が見つかりません ({json_path.name})")
            return None
            
        # 安全な平坦化処理
        flattened_rows = [flatten_frame(f) for f in frames if isinstance(f, dict)]
        flattened_rows = [r for r in flattened_rows if r]

        if not flattened_rows:
            log.warning(f"⚠️ スキップ: 有効なフレームデータがありません [{json_path.name}]")
            return None

        # すべての行からヘッダーキーを漏れなく収集
        fieldnames = []
        for r in flattened_rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened_rows)
            
        log.info(f"  ✓ 時系列CSV生成完了: {csv_path.name} ({len(flattened_rows)} 行)")
        return csv_path
        
    except Exception as e:
        log.error(f"  ✗ 時系列CSV変換エラー [{json_path.name}]: {e}")
        log.error(traceback.format_exc())
        return None

# (extract_calibration_data と save_calibration_master は変更なしですが、設定エラーは解消されています)
def extract_calibration_data(record: dict, file_stem: str) -> dict | None:
    row = {}
    row['username'] = file_stem
    row['subject_id'] = file_stem
    tokens = file_stem.split('-')
    if 'session_id' in record:
        row['session_id'] = record['session_id']
    elif len(tokens) > 1:
        row['session_id'] = tokens[-1]

    if 'calibration' in record and isinstance(record['calibration'], dict):
        cal_data = record['calibration']
        flat_cal = flatten_dict(cal_data)
        for k, v in flat_cal.items():
            if k not in ['time', 'clientEpochMs', 'clientPerfMs', 'label', 'username', 'session_id']:
                row[k] = v

        # FeatureCalculator を用いた集約特徴量の自動算出
        mean_lm = cal_data.get('meanLandmarks')
        if isinstance(mean_lm, dict):
            try:
                try:
                    from calibration.feature_calculator import FeatureCalculator
                except ImportError:
                    from features.lm.feature_calculator import FeatureCalculator

                calc = FeatureCalculator()
                ear_dict = mean_lm.get('ear', {})
                head_dict = mean_lm.get('headPose', {})
                gaze_dict = mean_lm.get('gaze', {})

                if ear_dict:
                    l_ear, r_ear, avg_ear = calc.calculate_instant_ear(ear_dict)
                    row['ear_calib'] = avg_ear
                    row['ear_left_calib'] = l_ear
                    row['ear_right_calib'] = r_ear

                if head_dict:
                    yaw, roll, pitch = calc.calculate_head_pose(head_dict)
                    row['head_yaw_calib'] = yaw
                    row['head_roll_calib'] = roll
                    row['head_pitch_calib'] = pitch

                if gaze_dict:
                    merged = {**ear_dict, **gaze_dict}
                    gaze_h, gaze_v = calc.calculate_eye_rotation(merged)
                    row['gaze_h_deg_calib'] = gaze_h
                    row['gaze_v_deg_calib'] = gaze_v
                    row['gaze_yaw_calib'] = gaze_h
                    row['gaze_pitch_calib'] = gaze_v
            except Exception as e:
                log.warning(f"キャリブレーション特徴量計算スキップ: {e}")
    else:
        for k, v in record.items():
            if k in ['data', 'username', 'session_id']:
                continue
            if isinstance(v, dict):
                flat_sub = flatten_dict(v, parent_key=k)
                row.update(flat_sub)
            else:
                row[k] = v
    return row

def save_calibration_master(calibration_rows: list[dict]):
    if not calibration_rows:
        return
    cal_master_path = CSV_OUTPUT_DIR / "calibration_master.csv"
    all_keys = set()
    for r in calibration_rows:
        all_keys.update(r.keys())
        
    priority_cols = ['username', 'session_id']
    other_cols = sorted([k for k in all_keys if k not in priority_cols])
    fieldnames = [col for col in priority_cols if col in all_keys] + other_cols
    
    with open(cal_master_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(calibration_rows)
        
    log.info(f"🏆 キャリブレーションマスターCSVを保存完了 ➔ {cal_master_path}")