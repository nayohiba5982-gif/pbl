import logging
import sys
from pathlib import Path
import yaml

# プロジェクトルートディレクトリ（preprocess ディレクトリ自身）
BASE_DIR = Path(__file__).resolve().parent

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("preprocess")


def load_config(config_filename: str = "config.yaml") -> dict:
    """
    YAML設定ファイルを読み込み、パス定義を pathlib.Path に変換して返します。
    """
    config_path = BASE_DIR / config_filename
    
    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --- paths 項目の自動 Path オブジェクト化 ---
    if "paths" in cfg:
        for key, val in cfg["paths"].items():
            cfg["paths"][key] = BASE_DIR / val

    # --- firebase 項目の自動 Path オブジェクト化 ---
    if "firebase" in cfg:
        for path_key in ["work_dir", "output_dir", "csv_output_dir", "zip_output"]:
            if path_key in cfg["firebase"]:
                cfg["firebase"][path_key] = BASE_DIR / cfg["firebase"][path_key]

        # 鍵ファイルパスの解決（preprocess直下またはcollect直下）
        if "key_path" in cfg["firebase"]:
            kpath = BASE_DIR / cfg["firebase"]["key_path"]
            if not kpath.exists():
                kpath_collect = BASE_DIR / "collect" / cfg["firebase"]["key_path"]
                if kpath_collect.exists():
                    kpath = kpath_collect
            cfg["firebase"]["key_path"] = kpath

    return cfg


# モジュールロード時に共通設定と主要パスをエクスポート
cfg = load_config()

# 後方互換性および各モジュール直接インポート用エイリアス
RAW_DIR = cfg["paths"].get("raw_dir", BASE_DIR / "data" / "raw")
LM_DIR = cfg["paths"].get("lm_dir", BASE_DIR / "data" / "lm")
RGB_DIR = cfg["paths"].get("rgb_dir", BASE_DIR / "data" / "rgb")
WORK_DIR = cfg["firebase"].get("work_dir", BASE_DIR / "data")
OUTPUT_DIR = cfg["firebase"].get("output_dir", BASE_DIR / "collect" / "json_output")
CSV_OUTPUT_DIR = cfg["firebase"].get("csv_output_dir", BASE_DIR / "data" / "raw")
ZIP_OUTPUT = cfg["firebase"].get("zip_output", BASE_DIR)