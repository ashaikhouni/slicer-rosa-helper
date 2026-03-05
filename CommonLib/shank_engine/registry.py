"""Registries for detection pipelines and pluggable components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .contracts import DetectionContext, DetectionError, DetectionResult
from .interfaces import DetectionPipeline


PipelineFactory = Callable[[], DetectionPipeline]


@dataclass
class RegisteredPipeline:
    key: str
    factory: PipelineFactory


class PipelineRegistry:
    """Factory-based registry for named detection pipelines."""

    def __init__(self):
        self._pipelines: dict[str, RegisteredPipeline] = {}

    @staticmethod
    def _normalize_key(key: str) -> str:
        return str(key).strip().lower()

    def register_pipeline(self, key: str, factory: PipelineFactory, overwrite: bool = False) -> None:
        k = self._normalize_key(key)
        if not k:
            raise DetectionError("pipeline key must be non-empty")
        if k in self._pipelines and not overwrite:
            raise DetectionError(f"pipeline '{k}' already registered")
        self._pipelines[k] = RegisteredPipeline(key=k, factory=factory)

    # Backward-compatible alias.
    register = register_pipeline

    def unregister_pipeline(self, key: str) -> None:
        self._pipelines.pop(self._normalize_key(key), None)

    def create_pipeline(self, key: str) -> DetectionPipeline:
        k = self._normalize_key(key)
        item = self._pipelines.get(k)
        if item is None:
            raise DetectionError(f"pipeline '{k}' is not registered")
        return item.factory()

    # Backward-compatible alias.
    create = create_pipeline

    def run(self, key: str, ctx: DetectionContext) -> DetectionResult:
        return self.create_pipeline(key).run(ctx)

    def keys(self) -> list[str]:
        return sorted(self._pipelines.keys())


class ComponentRegistry:
    """Registry for named components used inside pipelines."""

    def __init__(self):
        self._components: dict[str, Any] = {}

    @staticmethod
    def _normalize_key(key: str) -> str:
        return str(key).strip().lower()

    def register_component(self, key: str, component: Any, overwrite: bool = False) -> None:
        k = self._normalize_key(key)
        if not k:
            raise DetectionError("component key must be non-empty")
        if k in self._components and not overwrite:
            raise DetectionError(f"component '{k}' already registered")
        self._components[k] = component

    # Backward-compatible alias.
    register = register_component

    def get_component(self, key: str, default: Any = None) -> Any:
        return self._components.get(self._normalize_key(key), default)

    # Backward-compatible alias.
    get = get_component

    def keys(self) -> list[str]:
        return sorted(self._components.keys())


_DEFAULT_REGISTRY: PipelineRegistry | None = None


def get_default_registry() -> PipelineRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = PipelineRegistry()
    return _DEFAULT_REGISTRY
