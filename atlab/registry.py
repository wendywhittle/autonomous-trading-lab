from __future__ import annotations

from .models import StrategyVersion


class StrategyRegistry:
    """Registry of immutable strategy-version definitions."""

    def __init__(self):
        self._versions: dict[tuple[str, str], StrategyVersion] = {}

    def register(self, strategy: StrategyVersion) -> StrategyVersion:
        key = (strategy.strategy_id, strategy.version)
        candidate = strategy.model_copy(deep=True)
        existing = self._versions.get(key)
        if existing is not None:
            if existing != candidate:
                raise ValueError("STRATEGY_VERSION_IMMUTABLE")
            return existing.model_copy(deep=True)
        self._versions[key] = candidate
        return candidate.model_copy(deep=True)

    def get(self, strategy_id: str, version: str) -> StrategyVersion:
        try:
            return self._versions[(strategy_id, version)].model_copy(deep=True)
        except KeyError as exc:
            raise KeyError("STRATEGY_VERSION_NOT_FOUND") from exc
