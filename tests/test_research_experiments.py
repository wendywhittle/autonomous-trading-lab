"""Tests for experiment records, the registry, and the walk-forward runner."""

import itertools
from datetime import UTC, datetime

import pytest

from atlab.models import MarketObservation, StrategyVersion
from atlab.research.costs import CostModel
from atlab.research.experiments import (
    ExperimentRegistry,
    ExperimentResult,
    ResearchStage,
    run_experiment,
    run_walk_forward,
)


def obs(price, day, symbol="T"):
    return MarketObservation(
        symbol=symbol,
        timestamp=datetime(2024, 1, day, tzinfo=UTC),
        price=price,
        source="test",
    )


def momentum_strategy(**parameters):
    params = {"entry_return": 0.001}
    params.update(parameters)
    return StrategyVersion(
        strategy_id="momentum",
        version="1.0.0",
        hypothesis="test",
        parameters=params,
    )


def losing_series():
    # Bar2: +10% -> ENTER BUY 1 @110. Bar3: -10% -> EXIT SELL 1 @99 (-$11).
    return [obs(100, 1), obs(110, 2), obs(99, 3)]


def test_run_experiment_records_net_of_costs_accounting():
    cost_model = CostModel(name="c", fee_flat=1.0)
    result = run_experiment(losing_series(), momentum_strategy(), cost_model=cost_model)
    assert isinstance(result, ExperimentResult)
    assert result.num_trades == 2
    assert result.gross_pnl == pytest.approx(-11.0)
    assert result.total_costs == pytest.approx(2.0)
    assert result.net_pnl == pytest.approx(-13.0)
    assert result.expectancy_per_trade == pytest.approx(-6.5)
    assert result.stage is ResearchStage.RESEARCH
    assert result.notional_per_trade == pytest.approx(10.0)
    assert result.eval_start < result.eval_end


def test_run_experiment_is_reproducible():
    kwargs = {"cost_model": CostModel(name="c"), "random_seed": 7}
    first = run_experiment(losing_series(), momentum_strategy(), **kwargs)
    second = run_experiment(losing_series(), momentum_strategy(), **kwargs)
    assert first.experiment_id == second.experiment_id
    assert first == second


def test_run_experiment_id_changes_with_seed_costs_or_stage():
    base = {"cost_model": CostModel(name="c")}
    plain = run_experiment(losing_series(), momentum_strategy(), **base)
    assert (
        plain.experiment_id
        != run_experiment(
            losing_series(), momentum_strategy(), cost_model=CostModel(name="c"), random_seed=1
        ).experiment_id
    )
    assert (
        plain.experiment_id
        != run_experiment(
            losing_series(), momentum_strategy(), cost_model=CostModel(name="d")
        ).experiment_id
    )
    assert (
        plain.experiment_id
        != run_experiment(
            losing_series(), momentum_strategy(), stage=ResearchStage.PAPER, **base
        ).experiment_id
    )


def test_run_experiment_rejects_bad_input():
    with pytest.raises(ValueError, match="OBSERVATIONS_NOT_CHRONOLOGICAL"):
        run_experiment(
            [obs(100, 2), obs(110, 1)], momentum_strategy(), cost_model=CostModel(name="c")
        )
    with pytest.raises(ValueError, match="EMPTY_DATA_SNAPSHOT"):
        run_experiment([], momentum_strategy(), cost_model=CostModel(name="c"))
    with pytest.raises(ValueError, match="MULTIPLE_SYMBOLS"):
        run_experiment(
            [obs(100, 1), obs(110, 2, symbol="OTHER")],
            momentum_strategy(),
            cost_model=CostModel(name="c"),
        )
    with pytest.raises(ValueError, match="EXPERIMENT_NON_POSITIVE_NOTIONAL"):
        run_experiment(
            losing_series(),
            momentum_strategy(),
            cost_model=CostModel(name="c"),
            notional_per_trade=0.0,
        )


def test_run_walk_forward_yields_one_result_per_test_window_in_order():
    observations = [obs(100 + day, day) for day in range(1, 13)]
    results = run_walk_forward(
        observations,
        momentum_strategy(),
        cost_model=CostModel(name="c"),
        train_size=3,
        validate_size=2,
        test_size=2,
        step=2,
    )
    assert len(results) == 3
    assert [r.eval_start for r in results] == sorted(r.eval_start for r in results)
    for first, second in itertools.pairwise(results):
        assert first.eval_start < first.eval_end <= second.eval_start < second.eval_end
    assert all(r.strategy_id == "momentum" for r in results)
    assert all(r.parameters == results[0].parameters for r in results)


def test_run_walk_forward_empty_when_no_window_fits():
    results = run_walk_forward(
        [obs(100, 1), obs(101, 2)],
        momentum_strategy(),
        cost_model=CostModel(name="c"),
        train_size=3,
        validate_size=2,
        test_size=2,
        step=1,
    )
    assert results == ()


def test_registry_register_get_and_list():
    registry = ExperimentRegistry()
    result = run_experiment(losing_series(), momentum_strategy(), cost_model=CostModel(name="c"))
    assert registry.register(result) == result
    assert len(registry) == 1
    assert registry.get(result.experiment_id) == result
    assert registry.list_by_strategy("momentum") == (result,)
    assert registry.list_by_strategy("momentum", version="9.9") == ()
    assert registry.list_by_stage(ResearchStage.RESEARCH) == (result,)
    assert registry.list_by_stage(ResearchStage.PAPER) == ()
    # Re-registering identical content is idempotent.
    assert registry.register(result) == result
    assert len(registry) == 1


def test_registry_rejects_conflicting_reregistration():
    registry = ExperimentRegistry()
    result = run_experiment(losing_series(), momentum_strategy(), cost_model=CostModel(name="c"))
    registry.register(result)
    tampered = ExperimentResult(**{**result.__dict__, "net_pnl": result.net_pnl + 1000.0})
    with pytest.raises(ValueError, match="EXPERIMENT_RESULT_IMMUTABLE"):
        registry.register(tampered)


def test_registry_get_missing_raises():
    with pytest.raises(KeyError, match="EXPERIMENT_NOT_FOUND"):
        ExperimentRegistry().get("nope")
