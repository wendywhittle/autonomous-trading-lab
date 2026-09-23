from __future__ import annotations

from typing import Protocol

from .models import JEVDecision, MarketState, StrategyVersion


class JEVAdapter(Protocol):
    """Contract for the real JEV evaluation capability.

    This module does not emulate JEV. A connected implementation must supply
    the actual JEV evaluation capability and return a typed JEVDecision.
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
    """Fail-closed adapter until a real JEV implementation is connected."""

    def evaluate(
        self,
        strategy: StrategyVersion,
        state: MarketState,
        proposal: JEVDecision,
    ) -> JEVDecision:
        raise RuntimeError("JEV_NOT_CONFIGURED")
