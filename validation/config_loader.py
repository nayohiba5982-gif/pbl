"""
validation/config_loader.py
validation パッケージ（rppg / lm）共通の設定ファイル (YAML) ローダーおよびロガー設定
"""

import logging
from pathlib import Path
import sys
from typing import Optional, Union
import yaml

# validation のディレクトリ構造定義
VALIDATION_DIR = Path(__file__).resolve().parent
PROJECT_DIR = VALIDATION_DIR.parent
RPPG_DIR = VALIDATION_DIR / "rppg"
LM_DIR = VALIDATION_DIR / "lm"
PREPROCESS_DIR = PROJECT_DIR / "preprocess"

# 後方互換用エイリアス
BASE_DIR = VALIDATION_DIR

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("validation")


def load_config(
    config_path: Union[Path, str],
    cli_overrides: Optional[dict] = None,
) -> dict:
    """YAML設定ファイルを読み込み、CLIオーバーライドの反映およびパスの自動解決を行います。

    Parameters:
        config_path (Path | str): 設定ファイルへのパス (絶対パスまたは呼び出し元/BASE_DIRからの相対パス)
        cli_overrides (Optional[dict]): コマンドライン引数による上書き辞書

    Returns:
        dict: パスが Path オブジェクトに変換・解決された設定辞書
    """
    config_path = Path(config_path)
    
    # パス解決の試行
    if not config_path.is_absolute():
        candidates = [
            Path.cwd() / config_path,
            VALIDATION_DIR / config_path,
            LM_DIR / config_path,
            RPPG_DIR / "hr" / config_path,
            RPPG_DIR / "hrv" / config_path,
            RPPG_DIR / config_path,
        ]
        found = False
        for cand in candidates:
            if cand.exists():
                config_path = cand.resolve()
                found = True
                break
        if not found:
            config_path = (VALIDATION_DIR / config_path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg["_config_path"] = config_path
    config_dir = config_path.parent

    # CLI による上書きの反映
    if cli_overrides:
        paths_cfg = cfg.setdefault("paths", {})
        for k, v in cli_overrides.items():
            if v is not None:
                # 共通・パス関連
                if k in ["dataset_dir", "output_dir", "model_path", "data_dir", "raw_dir", "lm_normalized_dir"]:
                    paths_cfg[k] = v
                elif k == "subjects":
                    cfg["subjects"] = v
                # rPPG 関連
                elif k == "target_fps":
                    cfg.setdefault("extraction", {})["target_fps"] = v
                elif k == "ma_window_sec":
                    cfg.setdefault("moving_average", {})["window_sec"] = v
                elif k == "no_ma" and v:
                    cfg.setdefault("moving_average", {})["enabled"] = False
                elif k == "hrv_window_sec":
                    cfg.setdefault("hrv", {})["window_sec"] = v
                elif k == "hrv_stride_sec":
                    cfg.setdefault("hrv", {})["stride_sec"] = v
                elif k == "no_hrv" and v:
                    cfg.setdefault("hrv", {})["enabled"] = False
                # LM 関連
                elif k in ["save_plots", "dpi"]:
                    cfg.setdefault("visualization", {})[k] = v
                elif k in ["tolerance_mean", "tolerance_std", "max_nan_ratio"]:
                    cfg.setdefault("sanity_check", {})[k] = v

    # --- paths の自動解決 & Path オブジェクト化 ---
    paths_cfg = cfg.setdefault("paths", {})

    # 1. rPPG dataset_dir の解決
    if "dataset_dir" in paths_cfg or "rppg" in str(config_path):
        raw_ds = paths_cfg.get("dataset_dir", "datasets")
        ds_path = (config_dir / raw_ds).resolve()
        if not ds_path.exists():
            alt_candidates = [
                (RPPG_DIR / "datasets").resolve(),
                (RPPG_DIR / raw_ds).resolve(),
                (VALIDATION_DIR / "datasets").resolve(),
                (PROJECT_DIR / "validation" / "rppg" / "datasets").resolve(),
            ]
            for cand in alt_candidates:
                if cand.exists():
                    ds_path = cand
                    break
        paths_cfg["dataset_dir"] = ds_path

    # 2. LM 関連パス (data_dir, lm_normalized_dir, preprocess_dir等) の解決
    if "lm" in str(config_path) or "data_dir" in paths_cfg or "lm_normalized_dir" in paths_cfg:
        # preprocess の data ディレクトリを基準に自動解決
        preprocess_data = (PREPROCESS_DIR / "data").resolve()
        
        # data_dir
        if "data_dir" in paths_cfg:
            raw_d = paths_cfg["data_dir"]
            d_path = (config_dir / raw_d).resolve()
            if not d_path.exists():
                alt_d = (PREPROCESS_DIR / raw_d).resolve()
                if alt_d.exists():
                    d_path = alt_d
            paths_cfg["data_dir"] = d_path

        # lm_normalized_dir
        raw_norm = paths_cfg.get("lm_normalized_dir", "data/lm_normalized")
        norm_path = (config_dir / raw_norm).resolve()
        if not norm_path.exists():
            alt_norm1 = (PREPROCESS_DIR / raw_norm).resolve()
            alt_norm2 = (PREPROCESS_DIR / "data" / "lm_normalized").resolve()
            if alt_norm1.exists():
                norm_path = alt_norm1
            elif alt_norm2.exists():
                norm_path = alt_norm2
        paths_cfg["lm_normalized_dir"] = norm_path

    # 3. output_dir
    raw_out = paths_cfg.get("output_dir", "result")
    out_path = (config_dir / raw_out).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    paths_cfg["output_dir"] = out_path

    # 4. model_path
    raw_model = paths_cfg.get("model_path")
    if raw_model:
        m_path = (config_dir / raw_model).resolve()
        if not m_path.exists():
            alt_candidates = [
                (RPPG_DIR / raw_model).resolve(),
                (PROJECT_DIR / "make_ai" / "face_landmarker.task").resolve(),
                (PROJECT_DIR / "face_landmarker.task").resolve(),
            ]
            for cand in alt_candidates:
                if cand.exists():
                    m_path = cand
                    break
        paths_cfg["model_path"] = m_path
    else:
        paths_cfg["model_path"] = None

    return cfg
