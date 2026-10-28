"""
rppg_calibration.py

生体指標 (rPPG / BPM / 脈波) に対するウィンドウ内完結型 差分キャリブレーションモジュール。
外部ベースライン（実験前安静時キャリブレーション等）への依存を撤廃し、
「10秒ウィンドウ内で完結する局所ベースライン（相対特徴）」を基準値（base）として、
時系列生体シグナルの差分キャリブレーション (x - base) を実施します。

【設計方針・実行ルートの統一】
- リアルタイム推論と因果律を一致させるため、3D配列側ではなく cutting 前の時系列 DataFrame に対して
  因果的移動窓（center=False）を適用するルートに一本化します。
- 出力カラムとして窓内変動系列（bpm_diff 等）と局所ベースライン水準（bpm_base_10s 等）の両方を保持します。
- 処理速度を最適化するため、rolling.apply は使用せず method="mean" を標準高速パスとします。
"""

from pathlib import Path
import sys
from typing import Literal, Sequence, Union
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
    log = logging.getLogger("rppg_calibration")
    def load_config():
        return {}


DEFAULT_RPPG_CALIB_CONFIG = {
    "window_sec": 10.0,
    "fs": 50.0,
    "target_cols": ["bpm", "bvp_selected"],
    "baseline_method": "mean",  # "mean", "median", "initial"
    "center": False,            # 移動窓の場合: False = 過去10秒 (因果的・リアルタイム), True = 前後10秒 (中心窓)
    "min_periods_ratio": 0.3,   # 窓内に最低限必要な有効データ割合
    "output_diff_suffix": "_diff",
    "output_base_suffix": "_base_10s",
}


def compute_baseline_for_series(
    series: pd.Series,
    window_steps: int,
    method: Literal["mean", "median", "initial"] = "mean",
    center: bool = False,
    min_periods: int = 1,
) -> pd.Series:
    """
    1次元時系列データに対して、指定ステップ数の移動窓内で局所ベースラインを算出します。
    外部データを用いず、そのウィンドウ内のデータのみから基準値を決定します。
    ※ 処理速度最適化のため、rolling.apply は一切使用せず Pandas/NumPy のC実装演算のみを使用します。
    """
    s_clean = series.interpolate(method="linear").bfill().ffill()

    if method == "mean":
        # 最速・標準パス: 純粋な C 実装移動平均
        baseline = s_clean.rolling(
            window=window_steps,
            center=center,
            min_periods=min_periods,
        ).mean()
    elif method == "median":
        # C 実装移動中央値
        baseline = s_clean.rolling(
            window=window_steps,
            center=center,
            min_periods=min_periods,
        ).median()
    elif method == "initial":
        # 窓先頭 20% の平均をベクトル演算で高速算出（rolling.apply 回避）
        init_steps = max(1, int(window_steps * 0.2))
        init_ma = s_clean.rolling(window=init_steps, min_periods=1).mean()
        # 過去窓(center=False)の場合: (window_steps - init_steps) 分シフトして窓先頭の値を取得
        shift_n = max(0, window_steps - init_steps)
        baseline = init_ma.shift(shift_n)
    else:
        raise ValueError(f"未知のベースライン算出手法です: {method} (選択肢: 'mean', 'median', 'initial')")

    # 窓の立ち上がり部分（NaN）は先頭有効値で埋める
    baseline = baseline.bfill().ffill()
    return baseline


def calibrate_rppg_dataframe(
    df: pd.DataFrame,
    target_cols: Sequence[str] = ("bpm", "bvp_selected"),
    window_sec: float = 10.0,
    fs: float = 50.0,
    method: Literal["mean", "median", "initial"] = "mean",
    center: bool = False,
    min_periods_ratio: float = 0.3,
    diff_suffix: str = "_diff",
    base_suffix: str = "_base_10s",
    replace_original: bool = False,
) -> pd.DataFrame:
    """
    時系列DataFrameに対し、「10秒ウィンドウ内で完結する相対特徴」をbaseとした
    差分キャリブレーションを実施します。

    Parameters:
        df: 入力時系列DataFrame
        target_cols: キャリブレーション対象の生体指標カラム名リスト (例: ['bpm', 'bvp_selected'])
        window_sec: 基準値を算出する局所ウィンドウ長 (秒, デフォルト: 10.0)
        fs: サンプリング周波数 (Hz, デフォルト: 50.0)
        method: ベースラインの算出手法 ('mean', 'median', 'initial')
        center: 移動窓の配置 (False: 過去10秒間の因果的窓, True: 前後10秒の中心窓)
        min_periods_ratio: ウィンドウ内に必要な最小データ率
        diff_suffix: 出力される差分カラムのサフィックス (例: '_diff' -> 'bpm_diff')
        base_suffix: 局所ベースラインを保存するカラムのサフィックス (例: '_base_10s' -> 'bpm_base_10s')
        replace_original: Trueの場合、元のカラム（'bpm'等）の値を差分値で上書き

    Returns:
        pd.DataFrame: 差分キャリブレーション列（*_diff）が追加または更新されたDataFrame
    """
    res_df = df.copy()
    window_steps = max(1, int(round(window_sec * fs)))
    min_periods = max(1, int(round(window_steps * min_periods_ratio)))

    actual_targets = [c for c in target_cols if c in res_df.columns]
    if not actual_targets:
        log.warning(f"差分キャリブレーション対象のカラムが見つかりません: {target_cols}")
        return res_df

    for col in actual_targets:
        series = res_df[col]
        # ウィンドウ内完結型ベースラインの算出
        base_series = compute_baseline_for_series(
            series=series,
            window_steps=window_steps,
            method=method,
            center=center,
            min_periods=min_periods,
        )

        # 差分キャリブレーション: diff = x - base
        diff_series = series - base_series

        # カラムの格納
        diff_col_name = f"{col}{diff_suffix}"
        base_col_name = f"{col}{base_suffix}"

        res_df[base_col_name] = base_series
        res_df[diff_col_name] = diff_series

        if replace_original:
            res_df[col] = diff_series

    return res_df


def calibrate_rppg_segment(
    segment_df: pd.DataFrame,
    target_cols: Sequence[str] = ("bpm", "bvp_selected"),
    method: Literal["mean", "median", "initial"] = "mean",
    diff_suffix: str = "_diff",
    base_suffix: str = "_base_10s",
    replace_original: bool = False,
) -> pd.DataFrame:
    """
    単一の10秒ウィンドウ（セグメント）内で完全に自己完結する差分キャリブレーションを実施します。
    セグメント全体の代表値（平均または中央値）をそのウィンドウのベースラインとし、
    全ステップから差し引いた差分（diff）を返します。
    """
    res_df = segment_df.copy()
    actual_targets = [c for c in target_cols if c in res_df.columns]

    for col in actual_targets:
        vals = res_df[col].dropna()
        if len(vals) == 0:
            base_val = 0.0
        elif method == "median":
            base_val = float(vals.median())
        elif method == "initial":
            init_n = max(1, int(len(vals) * 0.2))
            base_val = float(vals.iloc[:init_n].mean())
        else:  # "mean"
            base_val = float(vals.mean())

        diff_vals = res_df[col] - base_val

        diff_col = f"{col}{diff_suffix}"
        base_col = f"{col}{base_suffix}"

        res_df[base_col] = base_val
        res_df[diff_col] = diff_vals

        if replace_original:
            res_df[col] = diff_vals

    return res_df


def calibrate_rppg_windows_array(
    X: np.ndarray,
    feature_cols: list[str],
    target_cols: Sequence[str] = ("bpm", "bvp_selected"),
    method: Literal["mean", "median"] = "mean",
) -> np.ndarray:
    """
    機械学習・TS2Vec向け3次元ウィンドウ配列 X [N_windows, Window_steps, Channels] に対し、
    各ウィンドウ内部（時間軸: axis=1）で完結する差分キャリブレーション (x - base) を適用します。
    """
    if X.ndim != 3:
        raise ValueError(f"X は 3次元配列 [N, steps, channels] である必要があります (現在: {X.ndim}次元)")

    X_calibrated = X.copy()

    for col_name in target_cols:
        if col_name not in feature_cols:
            continue
        ch_idx = feature_cols.index(col_name)

        channel_data = X_calibrated[:, :, ch_idx]

        if method == "median":
            base = np.nanmedian(channel_data, axis=1, keepdims=True)
        else:
            base = np.nanmean(channel_data, axis=1, keepdims=True)

        base = np.nan_to_num(base, nan=0.0)
        X_calibrated[:, :, ch_idx] = channel_data - base

    return X_calibrated


class DifferentialRPPGCalibrator:
    """
    生体指標（rPPG / BPM）の10秒ウィンドウ内完結型 差分キャリブレータクラス。
    外部ベースラインを一切使用せず、時系列データ自身の局所時間窓から
    ベースラインを推定し、差分を抽出します。
    """

    def __init__(self, cfg: dict = None):
        if cfg is None:
            cfg = load_config()

        rppg_calib_cfg = cfg.get("rppg_calibration", {})
        rppg_cfg = cfg.get("rppg", {})

        self.window_sec = float(rppg_calib_cfg.get("window_sec", DEFAULT_RPPG_CALIB_CONFIG["window_sec"]))
        self.fs = float(rppg_calib_cfg.get("fs", rppg_cfg.get("fs", DEFAULT_RPPG_CALIB_CONFIG["fs"])))
        self.target_cols = rppg_calib_cfg.get("target_cols", DEFAULT_RPPG_CALIB_CONFIG["target_cols"])
        self.baseline_method = rppg_calib_cfg.get("baseline_method", DEFAULT_RPPG_CALIB_CONFIG["baseline_method"])
        self.center = bool(rppg_calib_cfg.get("center", DEFAULT_RPPG_CALIB_CONFIG["center"]))
        self.min_periods_ratio = float(rppg_calib_cfg.get("min_periods_ratio", DEFAULT_RPPG_CALIB_CONFIG["min_periods_ratio"]))

        log.info(
            f"DifferentialRPPGCalibrator 初期化: 窓幅={self.window_sec}s, fs={self.fs}Hz, "
            f"手法={self.baseline_method}, 対象列={self.target_cols}"
        )

    def calibrate_dataframe(self, df: pd.DataFrame, replace_original: bool = False) -> pd.DataFrame:
        """単一の時系列DataFrameに対する差分キャリブレーション"""
        return calibrate_rppg_dataframe(
            df=df,
            target_cols=self.target_cols,
            window_sec=self.window_sec,
            fs=self.fs,
            method=self.baseline_method,
            center=self.center,
            min_periods_ratio=self.min_periods_ratio,
            replace_original=replace_original,
        )

    def calibrate_segment(self, segment_df: pd.DataFrame, replace_original: bool = False) -> pd.DataFrame:
        """単一ウィンドウ（10秒）セグメントに対する差分キャリブレーション"""
        return calibrate_rppg_segment(
            segment_df=segment_df,
            target_cols=self.target_cols,
            method=self.baseline_method,
            replace_original=replace_original,
        )

    def calibrate_windows_array(self, X: np.ndarray, feature_cols: list[str]) -> np.ndarray:
        """スライディングウィンドウ分割済み3次元配列に対する差分キャリブレーション"""
        return calibrate_rppg_windows_array(
            X=X,
            feature_cols=feature_cols,
            target_cols=self.target_cols,
            method=self.baseline_method if self.baseline_method in ("mean", "median") else "mean",
        )


def process_all_files_for_rppg_calibration(
    input_dir: str | Path,
    output_dir: str | Path,
    target_cols: Sequence[str] = ("bpm", "bvp_selected"),
    window_sec: float = 10.0,
    fs: float = 50.0,
    method: Literal["mean", "median", "initial"] = "mean",
    replace_original: bool = False,
):
    """
    指定ディレクトリ内の全CSVファイルに対して、10秒ウィンドウ内完結の差分キャリブレーションを一括適用し、
    指定先に出力します。
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(list(in_path.glob("*.csv")))
    if not csv_files:
        log.warning(f"処理対象のCSVファイルが見つかりません: {in_path}")
        return

    log.info(f"計 {len(csv_files)} 件のCSVファイルに対して生体指標の10秒差分キャリブレーションを実施します。")

    for file_path in csv_files:
        df = pd.read_csv(file_path)
        df_calib = calibrate_rppg_dataframe(
            df=df,
            target_cols=target_cols,
            window_sec=window_sec,
            fs=fs,
            method=method,
            replace_original=replace_original,
        )
        out_file = out_path / file_path.name
        df_calib.to_csv(out_file, index=False)
        log.info(f"  ✓ 差分キャリブレーション完了: {file_path.name} -> {out_file}")


def run_rppg_calibration(cfg: dict = None):
    """設定ファイルに基づき生体指標差分キャリブレーションを実行するヘルパー関数"""
    if cfg is None:
        cfg = load_config()

    calibrator = DifferentialRPPGCalibrator(cfg)
    paths_cfg = cfg.get("paths", {})

    input_dir = Path(paths_cfg.get("rppg_hr_extracted_dir", _BASE_DIR / "data" / "rppg_hr_extracted"))
    output_dir = Path(paths_cfg.get("rppg_calibrated_dir", _BASE_DIR / "data" / "rppg_calibrated"))

    if not input_dir.exists() or not list(input_dir.glob("*.csv")):
        fallback_dir = Path(paths_cfg.get("output_dir", _BASE_DIR / "data" / "features"))
        if fallback_dir.exists() and list(fallback_dir.glob("*.csv")):
            input_dir = fallback_dir

    process_all_files_for_rppg_calibration(
        input_dir=input_dir,
        output_dir=output_dir,
        target_cols=calibrator.target_cols,
        window_sec=calibrator.window_sec,
        fs=calibrator.fs,
        method=calibrator.baseline_method,
    )


if __name__ == "__main__":
    cfg = load_config()
    run_rppg_calibration(cfg)
