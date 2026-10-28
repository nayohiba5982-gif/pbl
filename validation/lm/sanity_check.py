"""
sanity_check.py
顔ランドマーク前処理 (Landmark Preprocessing) の健全性・整合性を包括的に検証するサニティチェックモジュール。

【主な検証項目】
1. 完全性 & 構造検証: 必須カラムの存在、データ行数、時間長の妥当性
2. 欠損値 & 不正値検証: NaN, Inf, Null の混入比率チェック
3. 時間軸 & サンプリング整合性: タイムスタンプ単調増加性、サンプリング間隔 (dt) のジッター・異常検出
4. 物理レンジ & 特徴量妥当性: EAR比率、頭部姿勢角度差分、視線角差分の現実的範囲チェック
5. 微分・平滑化異常検出: 速度・加速度における不連続スパイクの検出
6. 個人内 Zスコア標準化の数学的整合性: 個人内平均 = 0.0, 標準偏差 = 1.0 の厳密チェック
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# validation ルートを sys.path に追加
_LM_DIR = Path(__file__).resolve().parent
_VAL_DIR = _LM_DIR.parent
if str(_VAL_DIR) not in sys.path:
    sys.path.insert(0, str(_VAL_DIR))

from config_loader import log


@dataclass
class CheckResult:
    subject_id: str
    check_name: str
    status: str  # "PASS", "WARN", "FAIL"
    message: str
    details: Dict = field(default_factory=dict)


class LMSanityChecker:
    """顔ランドマーク前処理結果のサニティチェックを実行するクラス"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sanity_cfg = cfg.get("sanity_check", {})
        self.max_nan_ratio = float(self.sanity_cfg.get("max_nan_ratio", 0.001))
        self.max_inf_ratio = float(self.sanity_cfg.get("max_inf_ratio", 0.0))
        self.expected_fps = float(self.sanity_cfg.get("expected_fps", 50.0))
        self.expected_dt = 1.0 / self.expected_fps
        self.dt_tolerance = float(self.sanity_cfg.get("dt_tolerance_sec", 0.005))
        
        z_cfg = self.sanity_cfg.get("zscore", {})
        self.z_mean_tol = float(z_cfg.get("mean_tolerance", 0.01))
        self.z_std_tol = float(z_cfg.get("std_tolerance", 0.01))
        
        self.ranges = self.sanity_cfg.get("physical_ranges", {})
        self.spike_z_thresh = float(self.sanity_cfg.get("spike_z_threshold", 8.0))

    def check_file(self, file_path: Union[Path, str]) -> Tuple[List[CheckResult], pd.DataFrame]:
        """1つの CSV ファイルに対して全サニティチェックを実行"""
        file_path = Path(file_path)
        subject_id = file_path.stem.replace("lm_", "").replace("_normalized", "")
        results: List[CheckResult] = []

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            results.append(
                CheckResult(
                    subject_id=subject_id,
                    check_name="file_read",
                    status="FAIL",
                    message=f"ファイル読み込みエラー: {e}",
                )
            )
            return results, pd.DataFrame()

        # 1. 基本構造チェック (行数)
        if len(df) < 50:
            results.append(
                CheckResult(
                    subject_id=subject_id,
                    check_name="data_length",
                    status="FAIL",
                    message=f"行数が極端に少なすぎます ({len(df)} 行 < 50)",
                    details={"rows": len(df)},
                )
            )
        else:
            results.append(
                CheckResult(
                    subject_id=subject_id,
                    check_name="data_length",
                    status="PASS",
                    message=f"正常 ({len(df)} 行)",
                    details={"rows": len(df)},
                )
            )

        # 2. 欠損値・不正値 (NaN / Inf) チェック
        nan_cols = {}
        inf_cols = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                n_nan = df[col].isna().sum()
                nan_ratio = n_nan / len(df)
                if nan_ratio > self.max_nan_ratio:
                    nan_cols[col] = (n_nan, nan_ratio)

                n_inf = np.isinf(df[col]).sum()
                inf_ratio = n_inf / len(df)
                if inf_ratio > self.max_inf_ratio:
                    inf_cols[col] = (n_inf, inf_ratio)

        if inf_cols:
            results.append(
                CheckResult(
                    subject_id=subject_id,
                    check_name="inf_check",
                    status="FAIL",
                    message=f"Inf (無限大) が検出されました: {list(inf_cols.keys())}",
                    details=inf_cols,
                )
            )
        else:
            results.append(
                CheckResult(
                    subject_id=subject_id,
                    check_name="inf_check",
                    status="PASS",
                    message="Inf値なし (0件)",
                )
            )

        if nan_cols:
            results.append(
                CheckResult(
                    subject_id=subject_id,
                    check_name="nan_check",
                    status="WARN" if all(r <= 0.05 for _, (_, r) in nan_cols.items()) else "FAIL",
                    message=f"許容値を超えるNaNが存在します ({len(nan_cols)} カラム)",
                    details=nan_cols,
                )
            )
        else:
            results.append(
                CheckResult(
                    subject_id=subject_id,
                    check_name="nan_check",
                    status="PASS",
                    message=f"NaN率は許容内 (<={self.max_nan_ratio*100:.2f}%)",
                )
            )

        # 3. 時間軸・サンプリング整合性チェック
        time_cols = [c for c in ["time", "timestamp", "clientPerfMs", "clientEpochMs"] if c in df.columns]
        if time_cols:
            t_col = time_cols[0]
            t_vals = df[t_col].dropna().values
            if len(t_vals) > 1:
                # 単調増加チェック
                dt_array = np.diff(t_vals)
                # ミリ秒の場合は秒に変換 (中央値が500以上ならmsと判定)
                if np.nanmedian(dt_array) > 5.0:
                    dt_array = dt_array / 1000.0

                non_positive = np.sum(dt_array <= 0)
                if non_positive > 0:
                    results.append(
                        CheckResult(
                            subject_id=subject_id,
                            check_name="temporal_monotonicity",
                            status="FAIL",
                            message=f"タイムスタンプに逆行・重複があります ({non_positive} 箇所)",
                            details={"non_positive_count": int(non_positive)},
                        )
                    )
                else:
                    results.append(
                        CheckResult(
                            subject_id=subject_id,
                            check_name="temporal_monotonicity",
                            status="PASS",
                            message="タイムスタンプは狭義単調増加しています",
                        )
                    )

                # dt の期待値との乖離チェック (リサンプリング検証)
                dt_median = float(np.nanmedian(dt_array))
                dt_std = float(np.nanstd(dt_array))
                dt_error = abs(dt_median - self.expected_dt)

                if dt_error > self.dt_tolerance:
                    results.append(
                        CheckResult(
                            subject_id=subject_id,
                            check_name="sampling_rate",
                            status="WARN",
                            message=f"サンプリング間隔の中央値 ({dt_median:.4f}s) が期待値 ({self.expected_dt:.4f}s, {self.expected_fps}Hz) から乖離しています",
                            details={"dt_median": dt_median, "dt_std": dt_std, "dt_error": dt_error},
                        )
                    )
                else:
                    results.append(
                        CheckResult(
                            subject_id=subject_id,
                            check_name="sampling_rate",
                            status="PASS",
                            message=f"サンプリング間隔正常 (中央値: {dt_median:.4f}s, ジッターstd: {dt_std:.6f}s)",
                            details={"dt_median": dt_median, "dt_std": dt_std},
                        )
                    )
        else:
            results.append(
                CheckResult(
                    subject_id=subject_id,
                    check_name="temporal_monotonicity",
                    status="WARN",
                    message="時間軸カラム (time等) が存在しません",
                )
            )

        # 4. 物理レンジ・差分レンジの妥当性チェック
        for target_col, (low, high) in self.ranges.items():
            # 速度・加速度 (_vel, _acc) は除き、該当の幾何特徴量および平滑化特徴量のみを対象
            matching_cols = [c for c in [target_col, f"{target_col}_smooth"] if c in df.columns]
            for col in matching_cols:
                series = df[col].dropna()
                if len(series) == 0:
                    continue
                min_v = float(series.min())
                max_v = float(series.max())
                out_of_bounds = ((series < low) | (series > high)).sum()
                if out_of_bounds > 0:
                    ratio = out_of_bounds / len(series)
                    results.append(
                        CheckResult(
                            subject_id=subject_id,
                            check_name=f"range_{col}",
                            status="WARN" if ratio < 0.05 else "FAIL",
                            message=f"{col} の値が想定範囲 [{low}, {high}] を逸脱 (範囲外: {ratio*100:.1f}%, min={min_v:.2f}, max={max_v:.2f})",
                            details={"min": min_v, "max": max_v, "ratio": ratio},
                        )
                    )
                else:
                    results.append(
                        CheckResult(
                            subject_id=subject_id,
                            check_name=f"range_{col}",
                            status="PASS",
                            message=f"正常範囲内 [min={min_v:.2f}, max={max_v:.2f}]",
                            details={"min": min_v, "max": max_v},
                        )
                    )

        # 5. 個人内 Zスコア標準化の数学的整合性チェック
        # 注意: lm_*_z や centroid_z (3D座標のz成分) を除外し、真のZスコア特徴量を抽出
        z_cols = [
            c for c in df.columns
            if (c.endswith("_smooth_z") or c.endswith("_z"))
            and not c.startswith("lm_")
            and not c.startswith("centroid_")
        ]
        if not z_cols:
            # _smooth 等から派生したzスコア列が存在しない場合
            results.append(
                CheckResult(
                    subject_id=subject_id,
                    check_name="zscore_existence",
                    status="FAIL",
                    message="Zスコア標準化カラム (*_smooth_z 等) が1つも見つかりません",
                )
            )
        else:
            z_failures = []
            for z_col in z_cols:
                z_series = df[z_col].dropna()
                if len(z_series) == 0:
                    continue
                mean_val = float(z_series.mean())
                std_val = float(z_series.std())

                is_mean_ok = abs(mean_val) <= self.z_mean_tol
                is_std_ok = abs(std_val - 1.0) <= self.z_std_tol

                if not (is_mean_ok and is_std_ok):
                    z_failures.append({
                        "col": z_col,
                        "mean": mean_val,
                        "std": std_val,
                        "mean_error": abs(mean_val),
                        "std_error": abs(std_val - 1.0),
                    })

            if z_failures:
                results.append(
                    CheckResult(
                        subject_id=subject_id,
                        check_name="zscore_math_integrity",
                        status="FAIL",
                        message=f"{len(z_failures)}/{len(z_cols)} の特徴量でZスコアの平均0・分散1が不成立です",
                        details={"failures": z_failures},
                    )
                )
            else:
                results.append(
                    CheckResult(
                        subject_id=subject_id,
                        check_name="zscore_math_integrity",
                        status="PASS",
                        message=f"全 {len(z_cols)} 特徴量で平均≒0.0 (|mean|<={self.z_mean_tol}), 分散≒1.0 (|std-1|<={self.z_std_tol}) が成立",
                    )
                )

        # 6. 極端なスパイク・外れ値チェック
        spike_cols = {}
        for z_col in z_cols:
            z_series = df[z_col].dropna()
            if len(z_series) == 0:
                continue
            extreme_count = (z_series.abs() > self.spike_z_thresh).sum()
            if extreme_count > 0:
                spike_cols[z_col] = int(extreme_count)

        if spike_cols:
            results.append(
                CheckResult(
                    subject_id=subject_id,
                    check_name="spike_anomaly",
                    status="WARN",
                    message=f"|Z| > {self.spike_z_thresh} を超える極端なスパイクが検出されました ({len(spike_cols)} カラム)",
                    details=spike_cols,
                )
            )
        else:
            results.append(
                CheckResult(
                    subject_id=subject_id,
                    check_name="spike_anomaly",
                    status="PASS",
                    message=f"極端なスパイク (|Z| > {self.spike_z_thresh}) は検出されませんでした",
                )
            )

        return results, df

    def run_all(self, files: List[Path]) -> Tuple[pd.DataFrame, Dict[str, List[CheckResult]]]:
        """指定された全ファイルに対してサニティチェックを実行し、サマリー表を返します"""
        all_results: Dict[str, List[CheckResult]] = {}
        summary_rows = []

        log.info(f"▶ サニティチェック開始: 対象ファイル数 = {len(files)}")

        for f in files:
            results, _ = self.check_file(f)
            subject_id = f.stem.replace("lm_", "").replace("_normalized", "")
            all_results[subject_id] = results

            # 合否集計
            n_pass = sum(1 for r in results if r.status == "PASS")
            n_warn = sum(1 for r in results if r.status == "WARN")
            n_fail = sum(1 for r in results if r.status == "FAIL")

            overall_status = "FAIL" if n_fail > 0 else ("WARN" if n_warn > 0 else "PASS")

            summary_rows.append({
                "subject_id": subject_id,
                "file_name": f.name,
                "overall_status": overall_status,
                "pass_count": n_pass,
                "warn_count": n_warn,
                "fail_count": n_fail,
                "total_checks": len(results),
            })

            log_msg = f"  [{overall_status}] {subject_id}: PASS={n_pass}, WARN={n_warn}, FAIL={n_fail}"
            if overall_status == "FAIL":
                log.error(log_msg)
                for r in results:
                    if r.status == "FAIL":
                        log.error(f"    ❌ {r.check_name}: {r.message}")
            elif overall_status == "WARN":
                log.warning(log_msg)
                for r in results:
                    if r.status == "WARN":
                        log.warning(f"    ⚠️ {r.check_name}: {r.message}")
            else:
                log.info(log_msg)

        summary_df = pd.DataFrame(summary_rows)
        return summary_df, all_results
