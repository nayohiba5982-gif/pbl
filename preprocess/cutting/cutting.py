import os
import glob
from pathlib import Path
import pandas as pd
from config_loader import cfg as default_cfg, log, RAW_DIR, LM_DIR, RGB_DIR

def cut_data_only(cfg: dict = None) -> tuple[list[Path], list[Path]]:
    """
    data/raw/ 内のCSVファイルを読み込み、
    1. username 列をファイル名（拡張子なし）に修正
    2. Landmark 関連列（lm_*, centroid_*）を抽出して data/lm/ へ保存
    3. rPPG 関連列（rppg_*）を抽出して data/rgb/ へ保存
    
    Returns:
        tuple[list[Path], list[Path]]: (生成されたLMファイル一覧, 生成されたRGBファイル一覧)
    """
    c = cfg or default_cfg
    raw_dir = Path(c["paths"].get("raw_dir", RAW_DIR))
    lm_dir = Path(c["paths"].get("lm_dir", LM_DIR))
    rgb_dir = Path(c["paths"].get("rgb_dir", RGB_DIR))

    lm_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    
    # data/raw 内のすべてのCSVファイルを取得 (サブディレクトリも含め、calibration_master は除外)
    csv_candidates = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("**/*.csv"))
    # 重複除去 & calibration_master 除外
    seen = set()
    csv_files = []
    for p in csv_candidates:
        if p.is_file() and "calibration_master" not in p.name and p.resolve() not in seen:
            seen.add(p.resolve())
            csv_files.append(p)
            
    csv_files = sorted(csv_files)
    
    if not csv_files:
        log.warning(f"'{raw_dir}' 内に処理対象のCSVファイルが見つかりませんでした。")
        return [], []
        
    log.info(f"計 {len(csv_files)} 件のCSVファイルの切り出しを開始します。")

    lm_out_files = []
    rgb_out_files = []

    for file_path in csv_files:
        stem = file_path.stem
        # すでに lm_ や rgb_ が付いている場合は除外
        if stem.startswith("lm_") or stem.startswith("rgb_"):
            continue

        log.info(f"切り出し処理中: {file_path.name}")
        df = pd.read_csv(file_path)
        
        # ─── 1. username の修正 (科学表記バグ防止 & 一意化) ───
        df['username'] = stem

        # 基本メタデータ列
        base_cols = [c for c in ['time', 'clientEpochMs', 'clientPerfMs', 'label', 'username'] if c in df.columns]
        
        # ─── 2. Landmark 関連列の抽出 (lm_*, centroid_*) ───
        lm_cols = [c for c in df.columns if (c.startswith('lm_') or c.startswith('centroid_')) and c not in base_cols]
        if lm_cols:
            df_lm = df[base_cols + lm_cols]
            lm_path = lm_dir / f"lm_{stem}.csv"
            df_lm.to_csv(lm_path, index=False)
            lm_out_files.append(lm_path)
        else:
            log.warning(f"  ⚠️ {file_path.name} にLandmark列が見つかりません。")

        # ─── 3. rPPG 関連列の抽出 (rppg_*) ───
        rgb_cols = [c for c in df.columns if c.startswith('rppg_') and c not in base_cols]
        if rgb_cols:
            df_rgb = df[base_cols + rgb_cols]
            rgb_path = rgb_dir / f"rgb_{stem}.csv"
            df_rgb.to_csv(rgb_path, index=False)
            rgb_out_files.append(rgb_path)
        else:
            log.warning(f"  ⚠️ {file_path.name} にrPPG列が見つかりません。")

        log.info(f"  └─ 切り出し完了 ➔ lm/lm_{stem}.csv ({len(lm_cols)} 特徴量) / rgb/rgb_{stem}.csv ({len(rgb_cols)} 特徴量)")

    log.info(f"✓ すべてのファイルの切り出し処理が完了しました (LM: {len(lm_out_files)}件, RGB: {len(rgb_out_files)}件)")
    return lm_out_files, rgb_out_files