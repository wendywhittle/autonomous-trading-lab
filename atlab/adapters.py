from __future__ import annotations

from typing import Protocol, Iterable

from .models import MarketObservation


class MarketDataAdapter(Protocol):
    """Provider-independent market data contract."""
    def observations(self, symbol: str) -> Iterable[MarketObservation]: ...


class InMemoryMarketData:
    def __init__(self, observations: Iterable[MarketObservation]):
        self._observations = tuple(observations)

    def observations(self, symbol: str) -> Iterable[MarketObservation]:
        return tuple(item for item in self._observations if item.symbol == symbol)
