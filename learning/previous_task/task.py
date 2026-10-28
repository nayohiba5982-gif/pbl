"""
PreviousTask パイプライン統括クラス & エントリーポイント
(learning/previous_task/task.py)
"""

from __future__ import annotations

from pathlib import Path
import shutil
import pandas as pd

from config_loader import load_config, log

try:
    from .utils import normalize_window_list, ALL_SUPPORTED_WINDOWS
    from .elbow_analysis import run_pls_elbow_analysis
    from .pls_extractor import run_pls_feature_extraction
    from .standardization import run_subject_standardization
except ImportError:
    from utils import normalize_window_list, ALL_SUPPORTED_WINDOWS
    from elbow_analysis import run_pls_elbow_analysis
    from pls_extractor import run_pls_feature_extraction
    from standardization import run_subject_standardization


class PreviousTask:
    """
    Colab (previous_task.ipynb) で生成された TS2Vec 埋め込みファイルの検証、
    PLS 累積説明分散比 (R²) エルボー分析、PLS 潜在特徴量抽出、および
    GMM 入力直前の被験者別 (Subject-wise) Zスコア標準化を実行する統括クラス。
    """

    def __init__(
        self,
        cfg: dict,
        modality: str | None = None,
        window_size: str | list[str] | None = None,
        n_components_visual: int | None = None,
        n_components_vital: int | None = None,
    ):
        self.cfg = cfg
        self.paths = cfg.get("paths", {})
        self.dataset_cfg = cfg.get("dataset", {})
        self.prev_cfg = cfg.get("previous_task", {})
        self.loso_cfg = cfg.get("loso_cv", {})

        self.learning_root = Path(__file__).resolve().parent.parent

        # ウィンドウサイズの解決（引数 > previous_task.selected_window > dataset.window_size）
        default_ws = self.dataset_cfg.get("window_size", "15s")
        raw_ws = (
            window_size
            if window_size is not None
            else self.prev_cfg.get("selected_window", self.prev_cfg.get("selected_windows", default_ws))
        )
        self.target_windows = normalize_window_list(raw_ws, fallback=default_ws)
        self.window_size = self.target_windows[0]

        # PLS 成分数の解決（引数 > previous_task > loso_cv > デフォルト 3）
        self.n_components_visual = int(
            n_components_visual
            if n_components_visual is not None
            else self.prev_cfg.get("n_components_visual", self.loso_cfg.get("n_vis_pls", 3))
        )
        self.n_components_vital = int(
            n_components_vital
            if n_components_vital is not None
            else self.prev_cfg.get("n_components_vital", self.loso_cfg.get("n_bio_pls", 3))
        )
        self.candidate_components = list(self.prev_cfg.get("candidate_components", [1, 2, 3, 4, 5, 6]))
        self.apply_subject_norm = bool(self.prev_cfg.get("apply_subject_norm", True))
        self.save_plots = bool(self.prev_cfg.get("save_plots", True))
        self.group_col = self.dataset_cfg.get("group_col", "username")
        self.modality = modality or self.dataset_cfg.get("modality", "all")

        # ルートパス解決
        data_root = self.paths.get("data_root", "data")
        if not Path(data_root).is_absolute():
            data_root = (self.learning_root / data_root).resolve()
        self.data_root = Path(data_root)

        input_data_root = self.paths.get("input_data_root", "../preprocess/data/windowed")
        if not Path(input_data_root).is_absolute():
            input_data_root = (self.learning_root / input_data_root).resolve()
        self.input_data_root = Path(input_data_root)

        model_dir = self.paths.get("model_dir", "models")
        if not Path(model_dir).is_absolute():
            model_dir = (self.learning_root / model_dir).resolve()
        self.model_dir = Path(model_dir)
        self.pls_model_dir = self.model_dir / "pls"
        self.pls_model_dir.mkdir(parents=True, exist_ok=True)

        self.data_dir = self.data_root / self.window_size
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_dirs_for_window(self, ws: str) -> tuple[Path, Path]:
        """対象ウィンドウサイズの出力先 (data_dir) と入力検索先 (input_dir) を返します"""
        d_dir = self.data_root / ws
        d_dir.mkdir(parents=True, exist_ok=True)
        i_dir = self.input_data_root / ws
        return d_dir, i_dir

    def _resolve_split_files(self, ws: str, split: str = "train") -> tuple[Path | None, Path | None, Path | None]:
        """指定ウィンドウ・Split の (vis_emb_path, bio_emb_path, meta_path) を探索します"""
        d_dir, i_dir = self._get_dirs_for_window(ws)
        search_dirs = [d_dir, i_dir]

        vis_cands = []
        bio_cands = []
        meta_cands = []
        for s_dir in search_dirs:
            vis_cands.extend([
                s_dir / f"emb_{split}_visual.npy",
                s_dir / "visual" / f"emb_{split}.npy",
            ])
            bio_cands.extend([
                s_dir / f"emb_{split}_vital.npy",
                s_dir / "vital" / f"emb_{split}.npy",
            ])
            meta_cands.extend([
                s_dir / f"meta_{split}.csv",
                s_dir / "visual" / f"meta_{split}.csv",
                s_dir / "vital" / f"meta_{split}.csv",
            ])

        vis_path = next((p for p in vis_cands if p.exists()), None)
        bio_path = next((p for p in bio_cands if p.exists()), None)
        meta_path = next((p for p in meta_cands if p.exists()), None)

        # input_dir 側にのみメタデータがある場合は data_dir にもコピーして後続ステージで参照可能にする
        if meta_path is not None and meta_path.parent != d_dir and not (d_dir / f"meta_{split}.csv").exists():
            shutil.copy2(meta_path, d_dir / f"meta_{split}.csv")
            meta_path = d_dir / f"meta_{split}.csv"

        return vis_path, bio_path, meta_path

    # =========================================================================
    # 1. PLS 目的変数説明分散比 (累積 R²) エルボー分析
    # =========================================================================
    def run_pls_elbow_analysis(self, ws: str) -> pd.DataFrame | None:
        d_dir, _ = self._get_dirs_for_window(ws)
        vis_emb_path, bio_emb_path, meta_path = self._resolve_split_files(ws, "train")
        return run_pls_elbow_analysis(
            d_dir=d_dir,
            vis_emb_path=vis_emb_path,
            bio_emb_path=bio_emb_path,
            meta_path=meta_path,
            candidate_components=self.candidate_components,
            ws=ws,
            save_plots=self.save_plots,
            log=log,
        )

    # =========================================================================
    # 2. Train データのみによる PLS 学習 & 潜在特徴量抽出・保存
    # =========================================================================
    def run_pls_feature_extraction(self, ws: str) -> bool:
        d_dir, _ = self._get_dirs_for_window(ws)
        vis_train_path, bio_train_path, meta_train_path = self._resolve_split_files(ws, "train")
        return run_pls_feature_extraction(
            d_dir=d_dir,
            pls_model_dir=self.pls_model_dir,
            vis_train_path=vis_train_path,
            bio_train_path=bio_train_path,
            meta_train_path=meta_train_path,
            resolve_split_files_fn=self._resolve_split_files,
            n_components_visual=self.n_components_visual,
            n_components_vital=self.n_components_vital,
            ws=ws,
            log=log,
        )

    # =========================================================================
    # 3. GMM 入力直前の被験者別 (Subject-wise) Zスコア標準化
    # =========================================================================
    def run_subject_standardization(self, ws: str) -> bool:
        d_dir, _ = self._get_dirs_for_window(ws)
        return run_subject_standardization(
            d_dir=d_dir,
            resolve_split_files_fn=self._resolve_split_files,
            apply_subject_norm=self.apply_subject_norm,
            group_col=self.group_col,
            n_components_visual=self.n_components_visual,
            ws=ws,
            save_plots=self.save_plots,
            log=log,
        )

    # =========================================================================
    # 単一ウィンドウおよび全対象ウィンドウの一括実行
    # =========================================================================
    def run_for_window(self, ws: str, stages: set[str] | None = None) -> bool:
        """特定のウィンドウサイズに対して PreviousTask の各工程を実行"""
        d_dir, i_dir = self._get_dirs_for_window(ws)
        log.info(f"🔎 [{ws}] TS2Vec 埋め込みファイルを確認中: {d_dir}")

        emb_files = list(d_dir.glob("emb_*.npy")) + list(d_dir.glob("*/emb_*.npy"))
        if not emb_files and i_dir.exists():
            emb_files = list(i_dir.glob("emb_*.npy")) + list(i_dir.glob("*/emb_*.npy"))

        if not emb_files:
            log.warning(f"⚠️ [{ws}] TS2Vec 埋め込みファイルが見つかりません: {d_dir}")
            log.warning("  TS2Vec の学習・抽出は Google Colab (previous_task.ipynb) で実行し、生成された emb_*.npy 等を配置してください。")
            return False

        log.info(f"✓ [{ws}] 検出された埋め込みファイル ({len(emb_files)} 件): {[f.name for f in sorted(emb_files)[:6]]}")

        run_all = stages is None or "all" in stages

        # Step 1: PLS エルボー分析
        if run_all or "elbow" in stages:
            self.run_pls_elbow_analysis(ws)

        # Step 2: PLS 学習 & 潜在特徴量抽出・保存
        if run_all or "pls" in stages:
            ok_pls = self.run_pls_feature_extraction(ws)
            if not ok_pls:
                return False

        # Step 3: GMM 直前の被験者別 Zスコア標準化
        if run_all or "standardize" in stages or "norm" in stages:
            ok_std = self.run_subject_standardization(ws)
            if not ok_std:
                return False

        return True

    def run(self, stages: set[str] | None = None) -> bool:
        """設定された全対象ウィンドウサイズに対して順次実行"""
        any_success = False
        for idx, ws in enumerate(self.target_windows, 1):
            if len(self.target_windows) > 1:
                log.info("=" * 70)
                log.info(f" 🔄 [PreviousTask {idx}/{len(self.target_windows)}] ウィンドウサイズ: {ws} の処理を開始")
                log.info("=" * 70)
            ok = self.run_for_window(ws, stages=stages)
            if ok:
                any_success = True
        return any_success


def run_previous_task(
    cfg: dict | None = None,
    modality: str | None = None,
    window_size: str | list[str] | None = None,
    n_components_visual: int | None = None,
    n_components_vital: int | None = None,
) -> bool:
    """PreviousTask 実行ヘルパー関数"""
    if cfg is None:
        cfg = load_config()
    task = PreviousTask(
        cfg,
        modality=modality,
        window_size=window_size,
        n_components_visual=n_components_visual,
        n_components_vital=n_components_vital,
    )
    return task.run()
