"""Reproducible, results-carrying experiment records.

The Sniper v1 spec requires every experiment to carry a unique identity over
``experiment_id, dataset_version, feature_version, model_version,
hyperparameters, random_seed, cost_model, evaluation_period, results``.
:data:`ExperimentRecord` (in :mod:`atlab.experiment`) already covers identity;
this module adds the results layer:

- :class:`ExperimentResult` — one immutable record binding an experiment
  identity to its net-of-costs outcome over a declared evaluation period.
- :class:`ExperimentRegistry` — in-memory store with the same immutability
  guarantee as :class:`atlab.registry.StrategyRegistry`: re-registering an id
  with different content is refused.
- :func:`run_experiment` — run one experiment end to end.
- :func:`run_walk_forward` — run one experiment per walk-forward test window.

Cost accounting note: the paper backtest reports gross P&L from simulated
fills. Net P&L is derived analytically as
``gross_pnl - cost_model.total_cost(num_trades, notional_per_trade)`` where
``notional_per_trade`` is a *declared* assumption (default $10, matching the
risk charter's per-trade cap), recorded on the result so the assumption is
auditable rather than hidden.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from ..backtest import run_paper_backtest
from ..experiment import DataSnapshot, ExperimentRecord
from ..models import MarketObservation, StrategyVersion
from .costs import CostModel
from .splits import assert_chronological, walk_forward_windows


class ResearchStage(str, Enum):
    """Research pipeline stages, from the Sniper v1 spec's promotion ladder."""

    RESEARCH = "RESEARCH"
    SHADOW = "SHADOW"
    PAPER = "PAPER"
    MICRO_LIVE = "MICRO_LIVE"
    LIMITED_LIVE = "LIMITED_LIVE"


@dataclass(frozen=True)
class ExperimentResult:
    """Immutable record of one experiment's net-of-costs outcome."""

    experiment_id: str
    strategy_id: str
    strategy_version: str
    data_snapshot_id: str
    parameters: tuple[tuple[str, float], ...]
    random_seed: int
    cost_model: CostModel
    stage: ResearchStage
    eval_start: str
    eval_end: str
    num_observations: int
    num_trades: int
    num_round_trips: int
    gross_pnl: float
    total_costs: float
    net_pnl: float
    expectancy_per_trade: float
    max_drawdown: float
    win_rate: float
    sharpe_ratio: float
    notional_per_trade: float


class ExperimentRegistry:
    """Registry of immutable experiment results, keyed by experiment id."""

    def __init__(self) -> None:
        self._results: dict[str, ExperimentResult] = {}

    def register(self, result: ExperimentResult) -> ExperimentResult:
        existing = self._results.get(result.experiment_id)
        if existing is not None:
            if existing != result:
                raise ValueError("EXPERIMENT_RESULT_IMMUTABLE")
            return existing
        self._results[result.experiment_id] = result
        return result

    def get(self, experiment_id: str) -> ExperimentResult:
        try:
            return self._results[experiment_id]
        except KeyError as exc:
            raise KeyError("EXPERIMENT_NOT_FOUND") from exc

    def list_by_strategy(
        self, strategy_id: str, version: str | None = None
    ) -> tuple[ExperimentResult, ...]:
        return tuple(
            result
            for result in self._results.values()
            if result.strategy_id == strategy_id
            and (version is None or result.strategy_version == version)
        )

    def list_by_stage(self, stage: ResearchStage) -> tuple[ExperimentResult, ...]:
        return tuple(result for result in self._results.values() if result.stage == stage)

    def __len__(self) -> int:
        return len(self._results)


def _experiment_id(
    identity: ExperimentRecord,
    *,
    random_seed: int,
    cost_model: CostModel,
    eval_start: str,
    eval_end: str,
    stage: ResearchStage,
) -> str:
    payload = "|".join(
        (
            identity.experiment_id,
            str(random_seed),
            cost_model.fingerprint(),
            eval_start,
            eval_end,
            stage.value,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def run_experiment(
    observations: Sequence[MarketObservation],
    strategy: StrategyVersion,
    *,
    cost_model: CostModel,
    random_seed: int = 0,
    stage: ResearchStage = ResearchStage.RESEARCH,
    notional_per_trade: float = 10.0,
    starting_cash: float = 10_000.0,
    **backtest_kwargs,
) -> ExperimentResult:
    """Run the paper engine over ``observations`` and record net-of-costs results.

    ``notional_per_trade`` is the declared per-trade notional assumption used
    for cost math (default $10, the charter cap). It is stored on the result.
    Extra keyword arguments are forwarded to
    :func:`atlab.backtest.run_paper_backtest`.
    """
    ordered = list(observations)
    assert_chronological(ordered)
    if notional_per_trade <= 0:
        raise ValueError("EXPERIMENT_NON_POSITIVE_NOTIONAL")
    snapshot = DataSnapshot.create(ordered)
    identity = ExperimentRecord.create(strategy, snapshot)
    eval_start = snapshot.start.isoformat()
    eval_end = snapshot.end.isoformat()

    backtest_result = run_paper_backtest(
        ordered, strategy, starting_cash=starting_cash, **backtest_kwargs
    )
    metrics = backtest_result.metrics
    if metrics is None:  # pragma: no cover - unreachable for non-empty input
        raise ValueError("EXPERIMENT_NO_METRICS")

    gross_pnl = metrics.ending_equity - metrics.starting_equity
    total_costs = cost_model.total_cost(metrics.num_trades, notional_per_trade)
    net_pnl = gross_pnl - total_costs
    expectancy = net_pnl / metrics.num_trades if metrics.num_trades else 0.0

    return ExperimentResult(
        experiment_id=_experiment_id(
            identity,
            random_seed=random_seed,
            cost_model=cost_model,
            eval_start=eval_start,
            eval_end=eval_end,
            stage=stage,
        ),
        strategy_id=strategy.strategy_id,
        strategy_version=strategy.version,
        data_snapshot_id=snapshot.snapshot_id,
        parameters=identity.parameters,
        random_seed=random_seed,
        cost_model=cost_model,
        stage=stage,
        eval_start=eval_start,
        eval_end=eval_end,
        num_observations=metrics.num_observations,
        num_trades=metrics.num_trades,
        num_round_trips=metrics.num_round_trips,
        gross_pnl=gross_pnl,
        total_costs=total_costs,
        net_pnl=net_pnl,
        expectancy_per_trade=expectancy,
        max_drawdown=metrics.max_drawdown,
        win_rate=metrics.win_rate,
        sharpe_ratio=metrics.sharpe_ratio,
        notional_per_trade=notional_per_trade,
    )


def run_walk_forward(
    observations: Sequence[MarketObservation],
    strategy: StrategyVersion,
    *,
    cost_model: CostModel,
    train_size: int,
    validate_size: int,
    test_size: int,
    step: int,
    expanding: bool = True,
    embargo: int = 0,
    random_seed: int = 0,
    stage: ResearchStage = ResearchStage.RESEARCH,
    notional_per_trade: float = 10.0,
    starting_cash: float = 10_000.0,
    **backtest_kwargs,
) -> tuple[ExperimentResult, ...]:
    """Evaluate a fixed strategy version on every walk-forward test window.

    Each window's *test* slice becomes one experiment; the strategy version is
    held fixed across windows (correct for the lab's stateless strategies —
    there is no fitted model to retrain). Returns one result per window, in
    chronological order.
    """
    ordered = list(observations)
    assert_chronological(ordered)
    results = []
    for window in walk_forward_windows(
        len(ordered),
        train_size=train_size,
        validate_size=validate_size,
        test_size=test_size,
        step=step,
        expanding=expanding,
        embargo=embargo,
    ):
        test_slice = ordered[window.test[0] : window.test[1]]
        results.append(
            run_experiment(
                test_slice,
                strategy,
                cost_model=cost_model,
                random_seed=random_seed,
                stage=stage,
                notional_per_trade=notional_per_trade,
                starting_cash=starting_cash,
                **backtest_kwargs,
            )
        )
    return tuple(results)
