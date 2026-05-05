"""Plugin registry for extensible component registration.

Datasets, models, skill creators, runtime policies, and evaluators all
register themselves via ``Registry`` so that new implementations can be
added without touching existing code.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Registry:
    """Name -> class / factory mapping with optional metadata."""

    _stores: dict[str, dict[str, Any]] = {}

    def __init__(self, namespace: str) -> None:
        self._ns = namespace
        if namespace not in Registry._stores:
            Registry._stores[namespace] = {}

    # -- public API ----------------------------------------------------------

    def register(self, name: str | None = None) -> Callable:
        """Decorator that registers a class or factory function.

        Usage::

            dataset_registry = Registry("datasets")

            @dataset_registry.register("gsm8k")
            class GSM8KDataset(BaseDataset):
                ...
        """
        def decorator(cls_or_fn: T) -> T:
            key = name or getattr(cls_or_fn, "__name__", str(cls_or_fn))
            Registry._stores[self._ns][key] = cls_or_fn
            return cls_or_fn
        return decorator

    def get(self, name: str) -> Any:
        """Retrieve a registered component by name."""
        store = Registry._stores.get(self._ns, {})
        if name not in store:
            available = ", ".join(sorted(store.keys())) or "(none)"
            raise KeyError(
                f"[{self._ns}] '{name}' not registered. Available: {available}"
            )
        return store[name]

    def list(self) -> list[str]:
        """Return sorted names of all registered components."""
        return sorted(Registry._stores.get(self._ns, {}).keys())

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Get and instantiate a registered component."""
        cls_or_fn = self.get(name)
        return cls_or_fn(*args, **kwargs)

    @classmethod
    def reset(cls) -> None:
        """Clear all registries (useful in tests)."""
        cls._stores.clear()


# Pre-defined registries
dataset_registry = Registry("datasets")
model_registry = Registry("models")
skill_creator_registry = Registry("skill_creators")
runtime_policy_registry = Registry("runtime_policies")
evaluator_registry = Registry("evaluators")
skill_protocol_registry = Registry("skill_protocols")
