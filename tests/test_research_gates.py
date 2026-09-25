"""Tests for the research promotion gate."""

import itertools
from datetime import UTC, datetime, timedelta

import pytest

from atlab.research.costs import CostModel
from atlab.research.experiments import ExperimentResult, ResearchStage
from atlab.research.gates import ResearchEvidence, ResearchGate

_counter = itertools.count()


def make_result(
    *,
    net_pnl,
    num_trades=10,
    expectancy=None,
    day=1,
    span_days=30,
    strategy_id="s",
    version="1",
    parameters=(("p", 1.0),),
    gross_pnl=None,
    cost_model=None,
    stage=ResearchStage.RESEARCH,
):
    start = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=day)
    end = start + timedelta(days=span_days)
    gross = net_pnl if gross_pnl is None else gross_pnl
    return ExperimentResult(
        experiment_id=f"exp-{next(_counter)}",
        strategy_id=strategy_id,
        strategy_version=version,
        data_snapshot_id="snap",
        parameters=parameters,
        random_seed=0,
        cost_model=cost_model or CostModel(name="base"),
        stage=stage,
        eval_start=start.isoformat(),
        eval_end=end.isoformat(),
        num_observations=100,
        num_trades=num_trades,
        num_round_trips=5,
        gross_pnl=gross,
        total_costs=gross - net_pnl,
        net_pnl=net_pnl,
        expectancy_per_trade=net_pnl / num_trades if expectancy is None else expectancy,
        max_drawdown=0.01,
        win_rate=0.6,
        sharpe_ratio=1.2,
        notional_per_trade=10.0,
    )


def passing_evidence(**overrides):
    windows = (
        make_result(net_pnl=5.0, day=1),
        make_result(net_pnl=8.0, day=40),
    )
    neighbors = tuple(make_result(net_pnl=4.0, day=100 + i * 40) for i in range(4)) + (
        make_result(net_pnl=-1.0, day=300),
    )
    evidence = ResearchEvidence(
        strategy_id="s",
        strategy_version="1",
        windows=windows,
        stress_cost_model=CostModel(name="stress", fee_flat=0.1),
        param_neighbors=neighbors,
        regimes_tested=("TREND", "MEAN_REVERT"),
        leakage_reviewed=True,
        risk_limits_tested=True,
    )
    return evidence if not overrides else _replace(evidence, **overrides)


def _replace(evidence, **overrides):
    return ResearchEvidence(**{**evidence.__dict__, **overrides})


def test_gate_promotes_to_shadow_when_everything_passes():
    result = ResearchGate().evaluate(passing_evidence(), ResearchStage.SHADOW)
    assert result.eligible
    assert result.failed == ()
    assert result.target is ResearchStage.SHADOW
    details = dict(result.details)
    assert details["mean_expectancy_per_trade"] == pytest.approx(0.65)
    assert details["window_positive_fraction"] == pytest.approx(1.0)
    assert details["distinct_regimes"] == pytest.approx(2.0)


def test_gate_rejects_negative_expectancy():
    evidence = passing_evidence(
        windows=(make_result(net_pnl=-5.0, day=1), make_result(net_pnl=-8.0, day=40))
    )
    result = ResearchGate().evaluate(evidence, ResearchStage.SHADOW)
    assert not result.eligible
    assert "NET_EXPECTANCY_NOT_POSITIVE" in result.failed


def test_gate_rejects_inconsistent_walk_forward():
    evidence = passing_evidence(
        windows=(make_result(net_pnl=5.0, day=1), make_result(net_pnl=-2.0, day=40))
    )
    result = ResearchGate().evaluate(evidence, ResearchStage.SHADOW)
    assert "WALK_FORWARD_INCONSISTENT" in result.failed


def test_gate_rejects_when_stress_costs_kill_the_edge():
    # Stress costs of $1/trade on 10 trades erase a $5 gross edge.
    evidence = passing_evidence(
        stress_cost_model=CostModel(name="brutal", fee_flat=1.0),
    )
    result = ResearchGate().evaluate(evidence, ResearchStage.SHADOW)
    assert "STRESS_COSTS_NOT_SURVIVED" in result.failed


def test_gate_rejects_overlapping_windows():
    evidence = passing_evidence(
        windows=(make_result(net_pnl=5.0, day=1), make_result(net_pnl=8.0, day=10))
    )
    result = ResearchGate().evaluate(evidence, ResearchStage.SHADOW)
    assert "WINDOWS_NOT_CHRONOLOGICAL" in result.failed


def test_gate_rejects_mixed_window_identity():
    evidence = passing_evidence(
        windows=(
            make_result(net_pnl=5.0, day=1),
            make_result(net_pnl=8.0, day=40, parameters=(("p", 2.0),)),
        )
    )
    result = ResearchGate().evaluate(evidence, ResearchStage.SHADOW)
    assert "WINDOWS_MIXED_IDENTITY" in result.failed


def test_gate_rejects_missing_windows():
    evidence = passing_evidence(windows=())
    result = ResearchGate().evaluate(evidence, ResearchStage.SHADOW)
    assert "WINDOWS_MISSING" in result.failed


def test_gate_requires_parameter_sensitivity():
    evidence = passing_evidence(param_neighbors=())
    result = ResearchGate().evaluate(evidence, ResearchStage.SHADOW)
    assert "PARAMETER_SENSITIVITY_NOT_TESTED" in result.failed

    unstable = tuple(make_result(net_pnl=-4.0, day=100 + i * 40) for i in range(5))
    evidence = passing_evidence(param_neighbors=unstable)
    result = ResearchGate().evaluate(evidence, ResearchStage.SHADOW)
    assert "PARAMETER_STABILITY_FAILED" in result.failed


def test_gate_requires_regime_coverage_and_attestations():
    evidence = passing_evidence(regimes_tested=("TREND",))
    assert (
        "REGIME_COVERAGE_INSUFFICIENT"
        in ResearchGate().evaluate(evidence, ResearchStage.SHADOW).failed
    )

    evidence = passing_evidence(leakage_reviewed=False)
    assert (
        "LEAKAGE_REVIEW_MISSING" in ResearchGate().evaluate(evidence, ResearchStage.SHADOW).failed
    )

    evidence = passing_evidence(risk_limits_tested=False)
    assert (
        "RISK_LIMITS_NOT_TESTED" in ResearchGate().evaluate(evidence, ResearchStage.SHADOW).failed
    )


def test_gate_paper_target_requires_agreeing_paper_result():
    gate = ResearchGate()
    evidence = passing_evidence()
    result = gate.evaluate(evidence, ResearchStage.PAPER)
    assert "PAPER_RESULT_MISSING" in result.failed

    paper = make_result(net_pnl=7.0, expectancy=0.7, day=500, stage=ResearchStage.PAPER)
    evidence = passing_evidence(paper_result=paper)
    result = gate.evaluate(evidence, ResearchStage.PAPER)
    assert result.eligible, result.failed

    disagreeing = make_result(net_pnl=50.0, expectancy=5.0, day=500, stage=ResearchStage.PAPER)
    evidence = passing_evidence(paper_result=disagreeing)
    assert "PAPER_DISAGREES_WITH_SIMULATION" in gate.evaluate(evidence, ResearchStage.PAPER).failed


def test_gate_micro_live_requires_positive_paper():
    gate = ResearchGate()
    losing_paper = make_result(net_pnl=-3.0, expectancy=-0.3, day=500, stage=ResearchStage.PAPER)
    evidence = passing_evidence(paper_result=losing_paper)
    result = gate.evaluate(evidence, ResearchStage.MICRO_LIVE)
    assert "PAPER_NOT_NET_POSITIVE" in result.failed

    winning_paper = make_result(net_pnl=7.0, expectancy=0.7, day=500, stage=ResearchStage.PAPER)
    evidence = passing_evidence(paper_result=winning_paper)
    result = gate.evaluate(evidence, ResearchStage.MICRO_LIVE)
    assert result.eligible, result.failed
    result = gate.evaluate(evidence, ResearchStage.LIMITED_LIVE)
    assert result.eligible, result.failed


def test_gate_rejects_promotion_to_research():
    with pytest.raises(ValueError, match="RESEARCH_GATE_INVALID_TARGET"):
        ResearchGate().evaluate(passing_evidence(), ResearchStage.RESEARCH)


def test_gate_validates_threshold_arguments():
    with pytest.raises(ValueError, match="RESEARCH_GATE_INVALID_WINDOW_FRACTION"):
        ResearchGate(min_window_fraction=0.0)
    with pytest.raises(ValueError, match="RESEARCH_GATE_INVALID_NEIGHBOR_FRACTION"):
        ResearchGate(min_neighbor_fraction=1.5)
    with pytest.raises(ValueError, match="RESEARCH_GATE_INVALID_MIN_REGIMES"):
        ResearchGate(min_regimes=0)
    with pytest.raises(ValueError, match="RESEARCH_GATE_INVALID_PAPER_TOLERANCE"):
        ResearchGate(paper_abs_tol=-0.1)


def test_gate_thresholds_are_configurable():
    # A gate tolerating one bad window in four passes where the strict one fails.
    windows = (
        make_result(net_pnl=5.0, day=1),
        make_result(net_pnl=6.0, day=40),
        make_result(net_pnl=7.0, day=80),
        make_result(net_pnl=-1.0, day=120),
    )
    evidence = passing_evidence(windows=windows)
    assert (
        "WALK_FORWARD_INCONSISTENT"
        in ResearchGate().evaluate(evidence, ResearchStage.SHADOW).failed
    )
    lenient_failed = (
        ResearchGate(min_window_fraction=0.75).evaluate(evidence, ResearchStage.SHADOW).failed
    )
    assert "WALK_FORWARD_INCONSISTENT" not in lenient_failed
