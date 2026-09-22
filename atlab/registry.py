from __future__ import annotations

from .models import StrategyVersion


class StrategyRegistry:
    """In-memory registry of immutable strategy-version definitions."""

    def __init__(self):
        self._versions: dict[tuple[str, str], StrategyVersion] = {}

    def register(self, strategy: StrategyVersion) -> StrategyVersion:
        key = (strategy.strategy_id, strategy.version)
        existing = self._versions.get(key)
        if existing is not None:
            if existing != strategy:
                raise ValueError("STRATEGY_VERSION_IMMUTABLE")
            return existing
        self._versions[key] = strategy
        return strategy

    def get(self, strategy_id: str, version: str) -> StrategyVersion:
        try:
            return self._versions[(strategy_id, version)]
        except KeyError as exc:
            raise KeyError("STRATEGY_VERSION_NOT_FOUND") from exc
