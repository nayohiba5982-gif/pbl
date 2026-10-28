import logging
import sys
from pathlib import Path
import yaml

# プロジェクトルートディレクトリ（learning ディレクトリ自身）
BASE_DIR = Path(__file__).resolve().parent

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("learning")


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
            val_path = Path(val)
            if not val_path.is_absolute():
                val_path = (BASE_DIR / val_path).resolve()
            cfg["paths"][key] = val_path
    else:
        cfg["paths"] = {}

    # data_root (learning/data)
    if "data_root" not in cfg["paths"]:
        cfg["paths"]["data_root"] = (BASE_DIR / "data").resolve()

    # input_data_root (preprocess/data/windowed)
    if "input_data_root" not in cfg["paths"]:
        cfg["paths"]["input_data_root"] = (BASE_DIR / ".." / "preprocess" / "data" / "windowed").resolve()

    cfg["paths"]["data_root"].mkdir(parents=True, exist_ok=True)

    return cfg


# モジュールロード時にデフォルト設定をロード
try:
    cfg = load_config()
except Exception:
    cfg = {}
