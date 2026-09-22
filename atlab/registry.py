from __future__ import annotations

from .models import StrategyVersion


class StrategyRegistry:
    """Registry that rejects mutation of an existing strategy version."""
    def __init__(self):
        self._versions: dict[tuple[str, str], StrategyVersion] = {}

    def register(self, strategy: StrategyVersion) -> StrategyVersion:
        key = (strategy.strategy_id, strategy.version)
        existing = self._versions.get(key)
        if existing is not None and existing != strategy:
            raise ValueError("STRATEGY_VERSION_IMMUTABLE")
        self._versions[key] = strategy
        return strategy

    def get(self, strategy_id: str, version: str) -> StrategyVersion:
        try:
            return self._versions[(strategy_id, version)]
        except KeyError as exc:
            raise KeyError("STRATEGY_VERSION_NOT_FOUND") from exc
