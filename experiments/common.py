"""Shared helpers for experiment entrypoints."""

from __future__ import annotations

import logging
from pathlib import Path

from skilleval.core.config import Config
from skilleval.core.registry import dataset_registry, model_registry
from skilleval.utils.compat import check_transformers_trl_compat

# Trigger registry side effects once for all scripts importing this module.
import skilleval.datasets  # noqa: F401
import skilleval.models  # noqa: F401

logger = logging.getLogger(__name__)

DATASET_CONFIG_ALIASES = {
    "math": "math_bench",
}

MODEL_CONFIG_MAP = {
    "gpt-4o": ("openai", "configs/models/gpt4o.yaml"),
    "gpt-4o-mini": ("openai", "configs/models/gpt4o_mini.yaml"),
    "claude-sonnet": ("anthropic", "configs/models/claude.yaml"),
    "gemini-pro": ("google", "configs/models/gemini.yaml"),
    "local-llm": ("local", "configs/models/local.yaml"),
    "hf-transformers": ("hf_transformers", "configs/models/hf_transformers.yaml"),
}


def warn_compatibility_issues() -> None:
    for warning in check_transformers_trl_compat(strict=False):
        logger.warning("Compatibility warning: %s", warning)


def resolve_dataset_config_path(dataset: str, override: str | None = None) -> str:
    if override:
        return override
    direct = Path(f"configs/datasets/{dataset}.yaml")
    if direct.exists():
        return str(direct)
    alias = DATASET_CONFIG_ALIASES.get(dataset, dataset)
    aliased = Path(f"configs/datasets/{alias}.yaml")
    if aliased.exists():
        return str(aliased)
    raise FileNotFoundError(
        f"No dataset config found for '{dataset}'. Tried: {direct}, {aliased}"
    )


def build_dataset(dataset_name: str, dataset_config_override: str | None = None):
    ds_config = Config.from_yaml(
        resolve_dataset_config_path(dataset_name, dataset_config_override)
    )
    return dataset_registry.build(dataset_name, ds_config)


def build_model(provider: str, model_config_path: str):
    model_config = Config.from_yaml(model_config_path)
    return model_registry.build(provider, model_config)
