from __future__ import annotations

from typing import Protocol

from .models import JEVDecision, MarketState, StrategyVersion


class JEVAdapter(Protocol):
    """Contract for the real JEV evaluation capability.

    JEV is intentionally an external evaluation boundary. Implementations must
    return a typed JEVDecision; this module does not emulate JEV reasoning.
    """

    def evaluate(
        self,
        strategy: StrategyVersion,
        state: MarketState,
        proposal: JEVDecision,
    ) -> JEVDecision:
        """Evaluate a strategy proposal using the connected JEV capability."""
        ...


class UnconfiguredJEV:
    """Explicit fail-closed adapter until a real JEV implementation is connected."""

    def evaluate(
        self,
        strategy: StrategyVersion,
        state: MarketState,
        proposal: JEVDecision,
    ) -> JEVDecision:
        raise RuntimeError("JEV_NOT_CONFIGURED")
