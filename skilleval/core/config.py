"""Hierarchical YAML-based configuration management."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Thin wrapper around a nested dict loaded from YAML files.

    Supports merging (child overrides parent), dot-notation access,
    and conversion back to dict.
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = data or {}

    # -- I/O -----------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        cfg = cls(raw)
        if "_base_" in raw:
            base_path = path.parent / raw.pop("_base_")
            base = cls.from_yaml(base_path)
            base.merge(cfg)
            return base
        return cfg

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False, sort_keys=False)

    # -- access --------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Dot-notation access: ``cfg.get('model.name')``."""
        keys = key.split(".")
        node: Any = self._data
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    def set(self, key: str, value: Any) -> None:
        keys = key.split(".")
        node = self._data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value

    def __getitem__(self, key: str) -> Any:
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    # -- merge ---------------------------------------------------------------

    def merge(self, other: Config) -> Config:
        """Deep-merge *other* into self (other wins on conflict)."""
        self._data = _deep_merge(self._data, other._data)
        return self

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result
