"""
ベースモデル (Visual / Vital) ファクトリ & レジストリ
(learning/downstream_task/base_model/models/__init__.py)
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Type

from .base import BaseModalityRegressor
from .random_forest import RandomForestBaseRegressor
from .xgboost_model import XGBoostBaseRegressor
from .lightgbm_model import LightGBMBaseRegressor
from .catboost_model import CatBoostBaseRegressor
from .adaboost import AdaBoostBaseRegressor


DEFAULT_BASE_MODELS = [
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "adaboost",
]

BASE_MODEL_ALIASES: Dict[str, str] = {
    "random_forest": "random_forest",
    "rf": "random_forest",
    "randomforest": "random_forest",
    "xgboost": "xgboost",
    "xgb": "xgboost",
    "lightgbm": "lightgbm",
    "lgbm": "lightgbm",
    "lgb": "lightgbm",
    "catboost": "catboost",
    "cb": "catboost",
    "adaboost": "adaboost",
    "ada": "adaboost",
}

BASE_MODEL_REGISTRY: Dict[str, Type[BaseModalityRegressor]] = {
    "random_forest": RandomForestBaseRegressor,
    "xgboost": XGBoostBaseRegressor,
    "lightgbm": LightGBMBaseRegressor,
    "catboost": CatBoostBaseRegressor,
    "adaboost": AdaBoostBaseRegressor,
}

PARAM_KEY_MAP: Dict[str, str] = {
    "random_forest": "random_forest_params",
    "xgboost": "xgboost_params",
    "lightgbm": "lightgbm_params",
    "catboost": "catboost_params",
    "adaboost": "adaboost_params",
}


def normalize_base_model_name(name: str) -> str:
    """ベースモデル名のエイリアスを正規化します"""
    raw = str(name).lower().strip()
    for prefix in ["base_", "visual_", "vital_", "late_fusion_", "stacking_"]:
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    if raw not in BASE_MODEL_ALIASES:
        raise ValueError(
            f"未対応のベースモデルです: '{name}' (指定可能: {DEFAULT_BASE_MODELS})"
        )
    return BASE_MODEL_ALIASES[raw]


def create_base_model(
    model_name: str,
    modality: str = "visual",
    cfg: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None,
    early_stopping_rounds: Optional[int] = None,
) -> BaseModalityRegressor:
    """
    config.yaml の設定を参照してモダリティ別ベース回帰モデルを生成します。
    """
    canonical_name = normalize_base_model_name(model_name)
    model_cls = BASE_MODEL_REGISTRY[canonical_name]

    dt_cfg = cfg.get("downstream_task", {}) if cfg else {}
    bm_cfg = dt_cfg.get("base_model", {})
    es_cfg = bm_cfg.get("early_stopping", {})

    # Early Stopping ラウンド数の解決
    if early_stopping_rounds is not None:
        stop_rounds = int(early_stopping_rounds)
    elif es_cfg.get("enabled", True) is False:
        stop_rounds = 0
    else:
        stop_rounds = int(
            es_cfg.get(
                "stopping_rounds",
                bm_cfg.get("early_stopping_rounds", dt_cfg.get("early_stopping_rounds", 30)),
            )
        )

    # ハイパーパラメータの解決 (base_model.{model}_params > downstream_task.{model}_params)
    param_key = PARAM_KEY_MAP[canonical_name]
    raw_params = bm_cfg.get(param_key, dt_cfg.get(param_key, {}))
    params = dict(raw_params) if raw_params else {}

    # モダリティ個別の上書きパラメータがあれば反映 (例: base_model.visual.lightgbm_params)
    mod_specific = bm_cfg.get(modality, {}).get(param_key, {})
    if mod_specific:
        params.update(mod_specific)

    if random_state is not None:
        if canonical_name == "catboost":
            params.setdefault("random_seed", int(random_state))
        else:
            params.setdefault("random_state", int(random_state))

    return model_cls(
        modality=modality,
        params=params,
        early_stopping_rounds=stop_rounds,
    )


__all__ = [
    "BaseModalityRegressor",
    "RandomForestBaseRegressor",
    "XGBoostBaseRegressor",
    "LightGBMBaseRegressor",
    "CatBoostBaseRegressor",
    "AdaBoostBaseRegressor",
    "DEFAULT_BASE_MODELS",
    "BASE_MODEL_REGISTRY",
    "normalize_base_model_name",
    "create_base_model",
]
