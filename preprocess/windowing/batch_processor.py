"""
複数被験者ファイルに対する一括スライディングウィンドウ生成・保存プロセッサ
各特徴量グループ（視覚的特徴量 visual、生体的特徴量 vital、統合 combined 等）別に
スライディングウィンドウを自動生成・保存します。
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# preprocess 直下のモジュールをインポート可能にする
_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

try:
    from config_loader import load_config, log
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("windowing")
    def load_config():
        return {}

try:
    from .trim_margins import normalize_trim_margins
    from .window_generator import (
        create_sliding_windows_by_label_from_df,
        create_sliding_windows_for_user_from_df,
    )
except ImportError:
    from trim_margins import normalize_trim_margins
    from window_generator import (
        create_sliding_windows_by_label_from_df,
        create_sliding_windows_for_user_from_df,
    )


def process_all_files_for_windowing(
    input_dir: str | Path,
    output_base_dir: str | Path,
    window_secs: list[float],
    stride_sec: float,
    feature_cols: list[str],
    feature_groups: dict[str, list[str]] | None = None,
    fs: float = 50.0,
    max_nan_ratio: float = 0.1,
    time_col: str = "time",
    user_col: str = "username",
    label_col: str = "label",
    trim_margins: dict | None = None,
):
    """
    指定ディレクトリ配下の全CSVを読み込み、各特徴量グループ別（視覚的 visual, 生体的 vital, 統合 combined）に
    スライディングウィンドウを作成して保存します。

    【保存構造】
      - 従来互換パス:
          output_base_dir/{win_sec}/X_windows.npy (combined: 統合8次元)
          output_base_dir/{win_sec}/X_visual.npy  (visual: 視覚6次元)
          output_base_dir/{win_sec}/X_vital.npy   (vital: 生体2次元)
          output_base_dir/{win_sec}/metadata.csv
      - モダリティ別サブディレクトリ (Colab / 学習スクリプトでの直接指定用):
          output_base_dir/{win_sec}/{modality}/X_windows.npy, metadata.csv
          output_base_dir/{modality}/{win_sec}/X_windows.npy, metadata.csv
      - 被験者別:
          output_base_dir/{win_sec}/by_user/{username}_X.npy (combined)
          output_base_dir/{win_sec}/by_user/{username}_X_visual.npy
          output_base_dir/{win_sec}/by_user/{username}_X_vital.npy
          output_base_dir/{win_sec}/by_user/{username}_meta.csv
    """
    input_path = Path(input_dir)
    output_base_path = Path(output_base_dir)

    csv_files = sorted(list(input_path.glob("*.csv")))
    if not csv_files:
        log.warning(f"⚠️ 対象データが見つかりません: {input_path}")
        return

    # 特徴量グループのデフォルト設定（指定がない場合の自動フォールバック）
    if not feature_groups:
        visual_cols = [c for c in feature_cols if any(k in c for k in ["ear", "head", "gaze"])]
        vital_cols = [c for c in feature_cols if any(k in c for k in ["bvp", "bpm"])]
        feature_groups = {
            "combined": feature_cols,
            "visual": visual_cols if visual_cols else feature_cols,
            "vital": vital_cols if vital_cols else feature_cols,
        }

    # 全グループで参照されるユニークな特徴量カラム一覧を基準にウィンドウ作成
    all_needed_cols = list(dict.fromkeys(
        feature_cols + [c for cols in feature_groups.values() for c in cols]
    ))
    col_to_idx = {c: idx for idx, c in enumerate(all_needed_cols)}

    log.info(f"計 {len(csv_files)} 件の被験者ファイルのスライディングウィンドウ作成を開始します。")
    log.info(f"  - 特徴量グループ構成: {', '.join([f'{k}({len(v)}ch)' for k, v in feature_groups.items()])}")

    for win_sec in window_secs:
        win_str = f"{int(win_sec)}s" if win_sec.is_integer() else f"{win_sec}s"
        save_dir = output_base_path / win_str
        by_user_dir = save_dir / "by_user"

        save_dir.mkdir(parents=True, exist_ok=True)
        by_user_dir.mkdir(parents=True, exist_ok=True)

        # モダリティ別サブディレクトリを作成
        for grp_name in feature_groups.keys():
            (save_dir / grp_name).mkdir(parents=True, exist_ok=True)
            (output_base_path / grp_name / win_str).mkdir(parents=True, exist_ok=True)

        all_meta_list = []
        all_X_groups_list = {grp: [] for grp in feature_groups.keys()}

        log.info("=" * 60)
        log.info(f"▶ ウィンドウ作成中: Window = {win_sec}秒 ({int(win_sec * fs)} steps), Stride = {stride_sec}秒")
        log.info("=" * 60)

        for file_path in csv_files:
            df = pd.read_csv(file_path)
            user_id = str(df[user_col].iloc[0]) if user_col in df.columns else file_path.stem.replace("features_", "")

            # 基準となる全特徴量時系列から被験者1本の連続ウィンドウを作成
            X_full, meta_user = create_sliding_windows_for_user_from_df(
                df=df,
                feature_cols=all_needed_cols,
                window_sec=win_sec,
                stride_sec=stride_sec,
                fs=fs,
                max_nan_ratio=max_nan_ratio,
                time_col=time_col,
                user_col=user_col,
                label_col=label_col,
            )

            if len(X_full) == 0:
                log.warning(f"  ⚠️ {user_id}: 有効なウィンドウが生成されませんでした (短小またはNaN超過)")
                continue

            # 各グループのカラムスライスを生成
            user_X_groups = {}
            for grp_name, grp_cols in feature_groups.items():
                grp_indices = [col_to_idx[c] for c in grp_cols if c in col_to_idx]
                X_grp = X_full[:, :, grp_indices]
                user_X_groups[grp_name] = X_grp
                all_X_groups_list[grp_name].append(X_grp)

            all_meta_list.append(meta_user)

            # --- 被験者別のファイル保存 ---
            # 1. 従来互換保存: {user_id}_X.npy (combined), {user_id}_meta.csv
            comb_data = user_X_groups.get("combined", X_full)
            np.save(by_user_dir / f"{user_id}_X.npy", comb_data)
            meta_user.to_csv(by_user_dir / f"{user_id}_meta.csv", index=False)

            # 2. グループ別被験者保存: {user_id}_X_{grp}.npy
            for grp_name, grp_arr in user_X_groups.items():
                np.save(by_user_dir / f"{user_id}_X_{grp_name}.npy", grp_arr)

            # 3. ディレクトリ形式 (by_user/{user_id}/)
            user_sub_dir = by_user_dir / user_id
            user_sub_dir.mkdir(parents=True, exist_ok=True)
            np.save(user_sub_dir / "X_windows.npy", comb_data)
            meta_user.to_csv(user_sub_dir / "metadata.csv", index=False)
            for grp_name, grp_arr in user_X_groups.items():
                np.save(user_sub_dir / f"X_{grp_name}.npy", grp_arr)

        # --- 全体集約データの保存 ---
        if all_meta_list:
            meta_all = pd.concat(all_meta_list, ignore_index=True)
            tot_l0 = (meta_all["label"] == 0).sum()
            tot_l1 = (meta_all["label"] == 1).sum()

            log.info(f"✨ [全体集約完了: {win_str}] 合計: {len(meta_all):,d} 窓 (Label 0: {tot_l0:,d}, Label 1: {tot_l1:,d})")

            # メタデータ保存
            meta_all.to_csv(save_dir / "metadata.csv", index=False)

            for grp_name, arr_list in all_X_groups_list.items():
                if not arr_list:
                    continue
                X_grp_all = np.concatenate(arr_list, axis=0)
                n_ch = X_grp_all.shape[2]

                # 1. win_str 直下への保存
                if grp_name == "combined":
                    np.save(save_dir / "X_windows.npy", X_grp_all)
                np.save(save_dir / f"X_{grp_name}.npy", X_grp_all)

                # 2. win_str/{grp_name}/ への保存
                grp_dir = save_dir / grp_name
                np.save(grp_dir / "X_windows.npy", X_grp_all)
                meta_all.to_csv(grp_dir / "metadata.csv", index=False)

                # 3. {grp_name}/{win_str}/ への保存
                modal_win_dir = output_base_path / grp_name / win_str
                np.save(modal_win_dir / "X_windows.npy", X_grp_all)
                meta_all.to_csv(modal_win_dir / "metadata.csv", index=False)

                log.info(
                    f"   - {grp_name:<10} ({n_ch}ch): shape={X_grp_all.shape} ➔ "
                    f"{win_str}/X_{grp_name}.npy & {win_str}/{grp_name}/X_windows.npy"
                )
        else:
            log.warning(f"⚠️ [全体集約] ウィンドウが1件も生成されませんでした: {win_str}")


def run_windowing(cfg: dict = None):
    """YAML設定に基づいて全スライディングウィンドウ処理を実行するヘルパー関数"""
    if cfg is None:
        cfg = load_config()

    win_cfg = cfg.get("windowing", {})
    paths_cfg = cfg.get("paths", {})

    # 入力元ディレクトリ (data/features を優先、なければ data/lm_normalized)
    input_dir = Path(paths_cfg.get("output_dir", _BASE_DIR / "data" / "features"))
    if not input_dir.exists() or not list(input_dir.glob("*.csv")):
        fallback_dir = Path(paths_cfg.get("lm_normalized_dir", _BASE_DIR / "data" / "lm_normalized"))
        if fallback_dir.exists() and list(fallback_dir.glob("*.csv")):
            input_dir = fallback_dir
            log.info(f"featuresディレクトリが空のため、lm_normalized ({input_dir}) を使用します。")

    out_val = win_cfg.get("output_dir", "data/windowed")
    output_dir = Path(out_val) if Path(out_val).is_absolute() else _BASE_DIR / out_val
    window_secs = win_cfg.get("window_secs", [5, 10, 15])
    stride_sec = float(win_cfg.get("stride_sec", 1.0))
    target_fs = float(win_cfg.get("target_fs", 50.0))
    max_nan_ratio = float(win_cfg.get("max_nan_ratio", 0.1))
    feature_cols = win_cfg.get("feature_cols", [])
    feature_groups = win_cfg.get("feature_groups", None)
    trim_margins = win_cfg.get("trim_margins", None)

    process_all_files_for_windowing(
        input_dir=input_dir,
        output_base_dir=output_dir,
        window_secs=window_secs,
        stride_sec=stride_sec,
        feature_cols=feature_cols,
        feature_groups=feature_groups,
        fs=target_fs,
        max_nan_ratio=max_nan_ratio,
        trim_margins=trim_margins,
    )
