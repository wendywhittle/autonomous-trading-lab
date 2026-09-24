import statistics
from datetime import UTC, datetime

import pytest

from atlab.backtest import _compute_metrics, run_backtest, run_paper_backtest
from atlab.models import (
    DecisionAction,
    JEVDecision,
    MarketObservation,
    OrderStatus,
    PaperOrder,
    Side,
    StrategyVersion,
)
from atlab.runtime import CycleResult


def obs(price, day, symbol="T"):
    return MarketObservation(
        symbol=symbol,
        timestamp=datetime(2024, 1, day, tzinfo=UTC),
        price=price,
        source="test",
    )


def strategy():
    return StrategyVersion(
        strategy_id="momentum",
        version="1.0.0",
        hypothesis="test",
        parameters={"entry_return": 0.001},
    )


def test_run_backtest_still_returns_decisions_without_metrics():
    result = run_backtest([obs(100, 1), obs(110, 2), obs(99, 3)], strategy())
    assert len(result.decisions) == 2
    assert result.metrics is None


def test_paper_backtest_losing_round_trip_metrics():
    # Bar2: +10% -> ENTER BUY 1 @110 (equity 10000).
    # Bar3: -10% -> EXIT SELL 1 @99 (equity 9989, realized -11).
    result = run_paper_backtest([obs(100, 1), obs(110, 2), obs(99, 3)], strategy())
    metrics = result.metrics
    assert metrics.num_observations == 3
    assert metrics.num_decisions == 2
    assert metrics.num_trades == 2
    assert metrics.num_round_trips == 1
    assert metrics.starting_equity == pytest.approx(10_000.0)
    assert metrics.ending_equity == pytest.approx(9_989.0)
    assert metrics.total_return == pytest.approx(-0.0011)
    assert metrics.max_drawdown == pytest.approx(0.0011)
    assert metrics.sharpe_ratio == 0.0  # a single per-cycle return: no std dev
    assert metrics.win_rate == 0.0
    assert metrics.equity_curve == (
        ("2024-01-02T00:00:00+00:00", pytest.approx(10_000.0)),
        ("2024-01-03T00:00:00+00:00", pytest.approx(9_989.0)),
    )


def test_paper_backtest_no_trades():
    result = run_paper_backtest([obs(100, 1), obs(100, 2), obs(100, 3)], strategy())
    metrics = result.metrics
    assert metrics.num_trades == 0
    assert metrics.num_round_trips == 0
    assert metrics.win_rate == 0.0
    assert metrics.total_return == 0.0
    assert metrics.max_drawdown == 0.0
    assert metrics.ending_equity == pytest.approx(10_000.0)


def test_paper_backtest_empty_observations():
    result = run_paper_backtest([], strategy())
    assert result.decisions == ()
    assert result.start is None and result.end is None
    metrics = result.metrics
    assert metrics.num_observations == 0
    assert metrics.equity_curve == ()


def test_paper_backtest_rejects_multiple_symbols():
    with pytest.raises(ValueError, match="BACKTEST_MULTIPLE_SYMBOLS"):
        run_paper_backtest([obs(100, 1, "A"), obs(100, 2, "B")], strategy())


def test_paper_backtest_is_deterministic(tmp_path):
    observations = [obs(100 + (day % 7), day) for day in range(1, 30)]
    first = run_paper_backtest(observations, strategy(), workdir=tmp_path / "a")
    second = run_paper_backtest(observations, strategy(), workdir=tmp_path / "b")
    assert first.metrics == second.metrics
    assert [d.decision_id for d in first.decisions] == [d.decision_id for d in second.decisions]


def make_cycle(day, side, quantity, fill_price, equity):
    decision = JEVDecision(
        decision_id=f"d{day}",
        strategy_id="m",
        strategy_version="1",
        symbol="T",
        action=DecisionAction.ENTER if side is Side.BUY else DecisionAction.EXIT,
        side=side,
        confidence=1.0,
        rationale="test",
        state_fingerprint="s",
    )
    order = PaperOrder(
        order_id=f"o{day}",
        decision_id=f"d{day}",
        symbol="T",
        side=side,
        quantity=quantity,
        fill_price=fill_price,
        notional=quantity * fill_price,
        status=OrderStatus.FILLED,
    )
    return CycleResult(decision=decision, order=order, risk_reason="APPROVED", equity=equity)


def test_compute_metrics_winning_round_trip():
    ordered = [obs(100, 1), obs(110, 2), obs(110, 3)]
    cycles = [
        make_cycle(1, Side.BUY, 2, 100.0, 10_000.0),
        make_cycle(2, Side.SELL, 2, 110.0, 10_020.0),
    ]
    metrics = _compute_metrics(ordered, cycles, starting_cash=10_000.0, periods_per_year=252.0)
    assert metrics.num_trades == 2
    assert metrics.num_round_trips == 1
    assert metrics.win_rate == 1.0
    assert metrics.total_return == pytest.approx(0.002)
    assert metrics.max_drawdown == 0.0


def test_compute_metrics_max_drawdown():
    ordered = [obs(100, day) for day in range(1, 6)]
    cycles = [
        make_cycle(1, Side.BUY, 1, 100.0, 10_000.0),
        make_cycle(2, Side.BUY, 1, 100.0, 11_000.0),
        make_cycle(3, Side.BUY, 1, 100.0, 10_500.0),
        make_cycle(4, Side.BUY, 1, 100.0, 9_000.0),
    ]
    metrics = _compute_metrics(ordered, cycles, starting_cash=10_000.0, periods_per_year=252.0)
    assert metrics.max_drawdown == pytest.approx((11_000 - 9_000) / 11_000)
    assert metrics.total_return == pytest.approx(-0.1)


def test_compute_metrics_sharpe_ratio():
    ordered = [obs(100, day) for day in range(1, 6)]
    equities = [10_000.0, 10_100.0, 10_000.0, 10_100.0]
    cycles = [make_cycle(day, Side.BUY, 1, 100.0, equity) for day, equity in zip(range(1, 5), equities)]
    metrics = _compute_metrics(ordered, cycles, starting_cash=10_000.0, periods_per_year=252.0)
    returns = [equities[i] / equities[i - 1] - 1 for i in range(1, len(equities))]
    expected = statistics.mean(returns) / statistics.stdev(returns) * (252.0 ** 0.5)
    assert metrics.sharpe_ratio == pytest.approx(expected)


def test_compute_metrics_partial_pyramid_completes_one_round_trip():
    # Two buys then two sells: the round trip completes only when flat.
    ordered = [obs(100, day) for day in range(1, 6)]
    cycles = [
        make_cycle(1, Side.BUY, 1, 100.0, 10_000.0),
        make_cycle(2, Side.BUY, 1, 110.0, 10_010.0),
        make_cycle(3, Side.SELL, 1, 120.0, 10_020.0),  # closes the 100 lot: +20
        make_cycle(4, Side.SELL, 1, 105.0, 10_015.0),  # closes the 110 lot: -5
    ]
    metrics = _compute_metrics(ordered, cycles, starting_cash=10_000.0, periods_per_year=252.0)
    assert metrics.num_round_trips == 1
    # Net round-trip PnL is +15, so it counts as a win.
    assert metrics.win_rate == 1.0
