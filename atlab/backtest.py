from __future__ import annotations

import math
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .adapters import InMemoryMarketData
from .ledger import ImmutableLedger
from .models import MarketObservation, Side, StrategyVersion
from .portfolio import PaperPortfolio
from .risk import DeterministicRiskEngine
from .runtime import PaperTradingEngine, TradingMode
from .state import build_state
from .strategy import make_decision


@dataclass(frozen=True)
class BacktestMetrics:
    """Deterministic paper-accounting summary of a backtest run."""

    num_observations: int
    num_decisions: int
    num_trades: int
    num_round_trips: int
    starting_equity: float
    ending_equity: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    equity_curve: tuple  # tuple of (iso_timestamp, equity)


@dataclass(frozen=True)
class BacktestResult:
    decisions: tuple
    start: datetime | None
    end: datetime | None
    metrics: BacktestMetrics | None = None


def run_backtest(observations: list[MarketObservation], strategy: StrategyVersion) -> BacktestResult:
    ordered = sorted(observations, key=lambda item: item.timestamp)
    if not ordered:
        return BacktestResult((), None, None)
    decisions = []
    for index in range(1, len(ordered)):
        state = build_state(ordered[: index + 1], as_of=ordered[index].timestamp)
        decisions.append(make_decision(strategy, state))
    return BacktestResult(tuple(decisions), ordered[0].timestamp, ordered[-1].timestamp)


def _zero_metrics() -> BacktestMetrics:
    return BacktestMetrics(
        num_observations=0,
        num_decisions=0,
        num_trades=0,
        num_round_trips=0,
        starting_equity=0.0,
        ending_equity=0.0,
        total_return=0.0,
        max_drawdown=0.0,
        sharpe_ratio=0.0,
        win_rate=0.0,
        equity_curve=(),
    )


def run_paper_backtest(
    observations: list[MarketObservation],
    strategy: StrategyVersion,
    *,
    starting_cash: float = 10_000.0,
    quantity: float = 1.0,
    quantity_policy: Callable[[float], float] | None = None,
    stop_loss: float | None = None,
    risk: DeterministicRiskEngine | None = None,
    workdir: str | Path | None = None,
    periods_per_year: float = 252.0,
) -> BacktestResult:
    """Run the full paper trading loop over historical observations.

    Drives :class:`PaperTradingEngine` (strategy -> deterministic risk ->
    simulated fills -> portfolio accounting) with no JEV API and no broker.
    Returns decisions plus deterministic metrics: equity curve, max drawdown,
    Sharpe ratio (risk-free rate 0), total return, win rate over completed
    round trips, and trade counts.
    """
    ordered = sorted(observations, key=lambda item: item.timestamp)
    if not ordered:
        return BacktestResult((), None, None, _zero_metrics())
    symbols = {item.symbol for item in ordered}
    if len(symbols) != 1:
        raise ValueError("BACKTEST_MULTIPLE_SYMBOLS")
    symbol = next(iter(symbols))

    if workdir is None:
        workdir_context = tempfile.TemporaryDirectory(prefix="atlab-backtest-")
        workdir_path = Path(workdir_context.name)
    else:
        workdir_path = Path(workdir)
        workdir_path.mkdir(parents=True, exist_ok=True)
        workdir_context = None
    try:
        engine = PaperTradingEngine(
            InMemoryMarketData(ordered),
            strategy,
            risk or DeterministicRiskEngine(),
            PaperPortfolio(starting_cash),
            ImmutableLedger(workdir_path / "ledger.sqlite3"),
            quantity=quantity,
            quantity_policy=quantity_policy,
            stop_loss=stop_loss,
            mode=TradingMode.PAPER,
            portfolio_state_path=workdir_path / "portfolio.json",
            risk_state_path=workdir_path / "risk_session.json",
        )
        cycles = engine.run(symbol)
    finally:
        if workdir_context is not None:
            workdir_context.cleanup()

    decisions = tuple(cycle.decision for cycle in cycles)
    metrics = _compute_metrics(
        ordered, cycles, starting_cash=starting_cash, periods_per_year=periods_per_year
    )
    return BacktestResult(decisions, ordered[0].timestamp, ordered[-1].timestamp, metrics)


def _compute_metrics(ordered, cycles, *, starting_cash: float, periods_per_year: float) -> BacktestMetrics:
    # Cycle i corresponds to observations[i + 1] on a fresh run: the engine
    # iterates range(1, len(observations)) in order and appends one result
    # per bar. Zip defensively so a resumed ledger can never misalign.
    points = []
    for cycle, observation in zip(cycles, ordered[1:]):
        points.append((observation.timestamp.isoformat(), cycle.equity))
    equities = [equity for _, equity in points]

    if equities:
        peak = equities[0]
        max_drawdown = 0.0
        for equity in equities:
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - equity) / peak)
        ending_equity = equities[-1]
    else:
        max_drawdown = 0.0
        ending_equity = starting_cash

    total_return = (ending_equity - starting_cash) / starting_cash if starting_cash else 0.0

    returns = [
        equities[i] / equities[i - 1] - 1
        for i in range(1, len(equities))
        if equities[i - 1] != 0
    ]
    sharpe_ratio = 0.0
    if len(returns) >= 2:
        mean = sum(returns) / len(returns)
        variance = sum((item - mean) ** 2 for item in returns) / (len(returns) - 1)
        std = math.sqrt(variance)
        if std > 0 and periods_per_year > 0:
            sharpe_ratio = mean / std * math.sqrt(periods_per_year)

    # FIFO round-trip accounting: a round trip completes when the position
    # returns to flat; win rate is the fraction of round trips with PnL > 0.
    open_lots: list[list[float]] = []  # [quantity, fill_price]
    trip_pnl = 0.0
    round_trip_pnls: list[float] = []
    num_trades = 0
    for cycle in cycles:
        order = cycle.order
        if order is None:
            continue
        num_trades += 1
        if order.side is Side.BUY:
            open_lots.append([order.quantity, order.fill_price])
        else:
            remaining = order.quantity
            while remaining > 0 and open_lots:
                lot_quantity, lot_price = open_lots[0]
                take = min(remaining, lot_quantity)
                trip_pnl += (order.fill_price - lot_price) * take
                remaining -= take
                lot_quantity -= take
                if lot_quantity <= 0:
                    open_lots.pop(0)
                else:
                    open_lots[0] = [lot_quantity, lot_price]
            if not open_lots:
                round_trip_pnls.append(trip_pnl)
                trip_pnl = 0.0

    win_rate = (
        sum(1 for pnl in round_trip_pnls if pnl > 0) / len(round_trip_pnls)
        if round_trip_pnls
        else 0.0
    )

    return BacktestMetrics(
        num_observations=len(ordered),
        num_decisions=len(cycles),
        num_trades=num_trades,
        num_round_trips=len(round_trip_pnls),
        starting_equity=starting_cash,
        ending_equity=ending_equity,
        total_return=total_return,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        win_rate=win_rate,
        equity_curve=tuple(points),
    )

