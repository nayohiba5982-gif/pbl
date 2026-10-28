"""
メタモデル (Visual + Vital 予測値統合) ファクトリ & レジストリ
(learning/downstream_task/meta_model/models/__init__.py)
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Type

from .base import BaseMetaRegressor
from .logistic_regression import LogisticMetaRegressorModel
from .ridge import RidgeMetaRegressorModel
from .lasso import LassoMetaRegressorModel
from .elasticnet import ElasticNetMetaRegressorModel


DEFAULT_META_MODELS = [
    "logistic_regression",
    "ridge",
    "lasso",
    "elastic_net",
]

META_MODEL_ALIASES: Dict[str, str] = {
    "logistic_regression": "logistic_regression",
    "logistic": "logistic_regression",
    "lr": "logistic_regression",
    "logreg": "logistic_regression",
    "ridge": "ridge",
    "rigde": "ridge",
    "lasso": "lasso",
    "elastic_net": "elastic_net",
    "elasticnet": "elastic_net",
    "en": "elastic_net",
}

META_MODEL_REGISTRY: Dict[str, Type[BaseMetaRegressor]] = {
    "logistic_regression": LogisticMetaRegressorModel,
    "ridge": RidgeMetaRegressorModel,
    "lasso": LassoMetaRegressorModel,
    "elastic_net": ElasticNetMetaRegressorModel,
}

META_PARAM_KEY_MAP: Dict[str, str] = {
    "logistic_regression": "logistic_regression_params",
    "ridge": "ridge_params",
    "lasso": "lasso_params",
    "elastic_net": "elastic_net_params",
}


def normalize_meta_model_name(name: str) -> str:
    """メタモデル名のエイリアスを正規化します"""
    raw = str(name).lower().strip()
    for prefix in ["meta_", "late_fusion_", "stacking_"]:
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    if raw not in META_MODEL_ALIASES:
        raise ValueError(
            f"未対応のメタモデルです: '{name}' (指定可能: {DEFAULT_META_MODELS})"
        )
    return META_MODEL_ALIASES[raw]


def create_meta_model(
    model_name: str,
    cfg: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None,
) -> BaseMetaRegressor:
    """config.yaml の設定を参照してメタモデルインスタンスを生成します"""
    canonical_name = normalize_meta_model_name(model_name)
    model_cls = META_MODEL_REGISTRY[canonical_name]

    dt_cfg = cfg.get("downstream_task", {}) if cfg else {}
    mm_cfg = dt_cfg.get("meta_model", {})

    param_key = META_PARAM_KEY_MAP[canonical_name]
    raw_params = mm_cfg.get(param_key, dt_cfg.get(param_key, {}))
    params = dict(raw_params) if raw_params else {}

    if random_state is not None:
        params.setdefault("random_state", int(random_state))

    return model_cls(params=params)


__all__ = [
    "BaseMetaRegressor",
    "LogisticMetaRegressorModel",
    "RidgeMetaRegressorModel",
    "LassoMetaRegressorModel",
    "ElasticNetMetaRegressorModel",
    "DEFAULT_META_MODELS",
    "META_MODEL_REGISTRY",
    "normalize_meta_model_name",
    "create_meta_model",
]
