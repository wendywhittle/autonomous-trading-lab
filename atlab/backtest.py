from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .models import MarketObservation, StrategyVersion
from .state import build_state
from .strategy import make_decision


@dataclass(frozen=True)
class BacktestResult:
    decisions: tuple
    start: datetime | None
    end: datetime | None


def run_backtest(observations: list[MarketObservation], strategy: StrategyVersion) -> BacktestResult:
    ordered = sorted(observations, key=lambda item: item.timestamp)
    if not ordered:
        return BacktestResult((), None, None)
    decisions = []
    for index in range(1, len(ordered)):
        state = build_state(ordered[: index + 1], as_of=ordered[index].timestamp)
        decisions.append(make_decision(strategy, state))
    return BacktestResult(tuple(decisions), ordered[0].timestamp, ordered[-1].timestamp)
