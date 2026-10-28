"""
run_lm_validation.py
顔ランドマーク前処理 (Landmark Preprocessing) の検証実行スクリプト。
サニティチェックおよび Zスコア化後分布確認を一括または個別に実行します。

使用例:
    python run_lm_validation.py
    python run_lm_validation.py --mode sanity
    python run_lm_validation.py --mode distribution
    python run_lm_validation.py --config config.yaml --data-dir ../../preprocess/data/lm_normalized
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional
import pandas as pd

# validation ルートを sys.path に追加
_LM_DIR = Path(__file__).resolve().parent
_VAL_DIR = _LM_DIR.parent
if str(_VAL_DIR) not in sys.path:
    sys.path.insert(0, str(_VAL_DIR))

from config_loader import load_config, log, PREPROCESS_DIR
from sanity_check import LMSanityChecker
from distribution_checker import LMDistributionChecker


def find_target_csv_files(data_dir: Path, subjects: Optional[List[str]] = None) -> List[Path]:
    """対象データディレクトリから CSV ファイルを探索"""
    if not data_dir.exists():
        # 代替候補を探索
        alt_dirs = [
            PREPROCESS_DIR / "data" / "lm_normalized",
            PREPROCESS_DIR / "data" / "features",
            PREPROCESS_DIR / "data" / "lm_savgol",
        ]
        for alt in alt_dirs:
            if alt.exists() and list(alt.glob("*.csv")):
                log.info(f"データディレクトリを自動検出しました: {alt}")
                data_dir = alt
                break

    if not data_dir.exists():
        log.warning(f"データディレクトリが存在しません: {data_dir}")
        return []

    csv_files = sorted(list(data_dir.glob("*.csv")))
    if subjects:
        filtered = []
        for f in csv_files:
            stem = f.stem.replace("lm_", "").replace("_normalized", "")
            if stem in subjects:
                filtered.append(f)
        return filtered

    return csv_files


def run_lm_validation(config_path: Path | str, cli_overrides: Optional[dict] = None, mode: str = "all"):
    """ランドマーク前処理検証 (サニティチェック & 分布確認) のメイン実行関数"""
    # 1. 設定のロード
    cfg = load_config(config_path, cli_overrides=cli_overrides)
    paths_cfg = cfg.get("paths", {})
    data_dir = Path(paths_cfg.get("data_dir", "../../preprocess/data/lm_normalized"))
    output_dir = Path(paths_cfg.get("output_dir", "result"))
    output_dir.mkdir(parents=True, exist_ok=True)
    subjects = cfg.get("subjects")

    log.info("=" * 70)
    log.info("▶ 顔ランドマーク前処理 (LM Preprocess) 検証開始")
    log.info(f"  設定ファイル: {cfg.get('_config_path')}")
    log.info(f"  検証データディレクトリ: {data_dir}")
    log.info(f"  出力先ディレクトリ: {output_dir}")
    log.info(f"  実行モード: {mode}")
    log.info("=" * 70)

    # 2. 対象ファイルの収集
    files = find_target_csv_files(data_dir, subjects=subjects)
    if not files:
        log.error(f"❌ 検証対象の CSV ファイルが 1 件も見つかりませんでした。パスを確認してください: {data_dir}")
        log.info("💡 ヒント: preprocess パイプラインを実行して data/lm_normalized/*.csv を生成するか、--data-dir でパスを指定してください。")
        return False

    log.info(f"✓ 検証対象ファイル数: {len(files)} 件")
    for f in files:
        log.info(f"  - {f.name}")

    # 3. サニティチェック (Sanity Check) の実行
    if mode in ["all", "sanity"]:
        log.info("\n" + "=" * 50)
        log.info("【Phase 1】 サニティチェック (健全性・整合性検証)")
        log.info("=" * 50)
        checker = LMSanityChecker(cfg)
        summary_df, all_results = checker.run_all(files)

        # サマリー CSV の保存
        summary_csv_path = output_dir / "sanity_check_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        log.info(f"✓ サニティチェックサマリー保存: {summary_csv_path}")

        # 詳細結果 CSV の保存
        detailed_rows = []
        for sub_id, res_list in all_results.items():
            for r in res_list:
                detailed_rows.append({
                    "subject_id": sub_id,
                    "check_name": r.check_name,
                    "status": r.status,
                    "message": r.message,
                })
        detailed_df = pd.DataFrame(detailed_rows)
        detailed_csv_path = output_dir / "sanity_check_details.csv"
        detailed_df.to_csv(detailed_csv_path, index=False)
        log.info(f"✓ サニティチェック詳細ログ保存: {detailed_csv_path}")

        # 総合判定
        fail_files = summary_df[summary_df["overall_status"] == "FAIL"]
        warn_files = summary_df[summary_df["overall_status"] == "WARN"]
        pass_files = summary_df[summary_df["overall_status"] == "PASS"]

        log.info("-" * 50)
        log.info(f"サニティチェック総合結果: PASS={len(pass_files)}, WARN={len(warn_files)}, FAIL={len(fail_files)} / 全 {len(files)} 件")
        if len(fail_files) > 0:
            log.error(f"❌ FAIL 判定の被験者: {fail_files['subject_id'].tolist()}")
        if len(warn_files) > 0:
            log.warning(f"⚠️ WARN 判定の被験者: {warn_files['subject_id'].tolist()}")
        if len(fail_files) == 0 and len(warn_files) == 0:
            log.info("🎉 全てのサニティチェック項目に完全合格しました！")

    # 4. Zスコア化後 分布確認 (Distribution Analysis) の実行
    if mode in ["all", "dist", "distribution"]:
        log.info("\n" + "=" * 50)
        log.info("【Phase 2】 Zスコア化後 分布確認・統計分析・可視化")
        log.info("=" * 50)
        
        # データのロード
        dfs: Dict[str, pd.DataFrame] = {}
        for f in files:
            sub_id = f.stem.replace("lm_", "").replace("_normalized", "")
            try:
                dfs[sub_id] = pd.read_csv(f)
            except Exception as e:
                log.warning(f"ファイル読み込み失敗 ({f.name}): {e}")

        if not dfs:
            log.error("❌ 分布確認用のデータフレームを読み込めませんでした。")
            return False

        dist_checker = LMDistributionChecker(cfg)
        outputs = dist_checker.run_all(dfs)
        log.info("✓ 分布確認・可視化処理が完了しました。")

    log.info("\n" + "=" * 70)
    log.info(f"✅ 全ての検証工程が完了しました。出力ディレクトリ: {output_dir}")
    log.info("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="顔ランドマーク前処理 サニティチェック & Zスコア分布確認ツール"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="設定YAMLファイル (デフォルト: config.yaml)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "sanity", "dist", "distribution"],
        default="all",
        help="実行モード: all (全て), sanity (サニティチェックのみ), dist (分布確認のみ)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="検証対象の CSV ファイル格納ディレクトリ (config.yaml の設定を上書き)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="結果出力ディレクトリ (config.yaml の設定を上書き)",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="検証対象とする被験者ID (指定しない場合は全被験者)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="グラフ画像のプロット保存をスキップする",
    )

    args = parser.parse_args()

    cli_overrides = {}
    if args.data_dir:
        cli_overrides["data_dir"] = args.data_dir
        cli_overrides["lm_normalized_dir"] = args.data_dir
    if args.output_dir:
        cli_overrides["output_dir"] = args.output_dir
    if args.subjects:
        cli_overrides["subjects"] = args.subjects
    if args.no_plots:
        cli_overrides["save_plots"] = False

    run_lm_validation(
        config_path=args.config,
        cli_overrides=cli_overrides,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
