"""The research promotion gate.

The Sniper v1 spec refuses to promote a candidate merely because it made money
in one backtest. This gate encodes that refusal as code. It is deliberately
separate from :class:`atlab.promotion.PromotionGate`, which checks operational
readiness (tests green, kill switch, credentials); this gate checks *research
validity*: does the edge survive costs, walk-forward testing, multiple
regimes, parameter perturbation, and paper trading?

Stage requirements:

- ``RESEARCH -> SHADOW``: net expectancy, walk-forward consistency, stress
  costs, chronological windows, parameter stability, regime coverage, leakage
  review, risk-limit testing.
- ``SHADOW -> PAPER``: everything above, plus paper-trading behavior agreeing
  with simulation.
- ``PAPER -> MICRO_LIVE`` and ``MICRO_LIVE -> LIMITED_LIVE``: everything
  above, plus the paper result itself being net positive.

Two checks cannot be fully automated and are enforced as explicit researcher
attestations on the evidence: ``leakage_reviewed`` (no feature uses
future information) and ``risk_limits_tested``. Everything else is computed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime

from .costs import CostModel
from .experiments import ExperimentResult, ResearchStage


@dataclass(frozen=True)
class ResearchEvidence:
    """Everything the research gate needs to judge a promotion."""

    strategy_id: str
    strategy_version: str
    windows: tuple[ExperimentResult, ...]  #: walk-forward test-window results
    stress_cost_model: CostModel  #: harsher costs the edge must also survive
    param_neighbors: tuple[ExperimentResult, ...] = ()  #: nearby-parameter runs
    paper_result: ExperimentResult | None = None  #: paper-stage result, if any
    regimes_tested: tuple[str, ...] = ()  #: market regimes covered by testing
    leakage_reviewed: bool = False  #: researcher attests: no future-data leakage
    risk_limits_tested: bool = False  #: researcher attests: risk limits exercised


@dataclass(frozen=True)
class ResearchGateResult:
    eligible: bool
    target: ResearchStage
    failed: tuple[str, ...]
    details: tuple[tuple[str, float], ...]  #: key numbers behind the verdict


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values) if values else 0.0


def _eval_bounds(result: ExperimentResult) -> tuple[datetime, datetime]:
    return (
        datetime.fromisoformat(result.eval_start),
        datetime.fromisoformat(result.eval_end),
    )


class ResearchGate:
    """Cost-aware, walk-forward research promotion gate.

    All thresholds are constructor parameters with strict defaults — they are
    initial research parameters, not sacred numbers, and every evaluation
    records the values used in its details.
    """

    def __init__(
        self,
        *,
        min_expectancy_per_trade: float = 0.0,
        min_window_fraction: float = 1.0,
        min_neighbor_fraction: float = 0.8,
        min_regimes: int = 2,
        paper_rel_tol: float = 0.5,
        paper_abs_tol: float = 0.05,
    ) -> None:
        if not 0.0 < min_window_fraction <= 1.0:
            raise ValueError("RESEARCH_GATE_INVALID_WINDOW_FRACTION")
        if not 0.0 < min_neighbor_fraction <= 1.0:
            raise ValueError("RESEARCH_GATE_INVALID_NEIGHBOR_FRACTION")
        if min_regimes < 1:
            raise ValueError("RESEARCH_GATE_INVALID_MIN_REGIMES")
        if paper_rel_tol < 0 or paper_abs_tol < 0:
            raise ValueError("RESEARCH_GATE_INVALID_PAPER_TOLERANCE")
        self.min_expectancy_per_trade = min_expectancy_per_trade
        self.min_window_fraction = min_window_fraction
        self.min_neighbor_fraction = min_neighbor_fraction
        self.min_regimes = min_regimes
        self.paper_rel_tol = paper_rel_tol
        self.paper_abs_tol = paper_abs_tol

    # -- internal checks ----------------------------------------------------

    def _check_windows_present(self, evidence: ResearchEvidence) -> str | None:
        if not evidence.windows:
            return "WINDOWS_MISSING"
        return None

    def _check_windows_consistent(self, evidence: ResearchEvidence) -> str | None:
        if not evidence.windows:
            return None  # covered by _check_windows_present
        first = evidence.windows[0]
        expected_identity = (
            evidence.strategy_id,
            evidence.strategy_version,
            first.parameters,
            first.cost_model.fingerprint(),
        )
        for window in evidence.windows:
            actual = (
                window.strategy_id,
                window.strategy_version,
                window.parameters,
                window.cost_model.fingerprint(),
            )
            if actual != expected_identity:
                return "WINDOWS_MIXED_IDENTITY"
        return None

    def _check_windows_chronological(self, evidence: ResearchEvidence) -> str | None:
        if not evidence.windows:
            return None  # covered by _check_windows_present
        bounds = [_eval_bounds(window) for window in evidence.windows]
        for (prev_start, prev_end), (cur_start, cur_end) in itertools.pairwise(bounds):
            if cur_start < prev_end or cur_end <= cur_start or prev_end <= prev_start:
                return "WINDOWS_NOT_CHRONOLOGICAL"
        return None

    def _mean_expectancy(self, evidence: ResearchEvidence) -> float:
        return _mean(tuple(w.expectancy_per_trade for w in evidence.windows))

    def _window_positive_fraction(self, evidence: ResearchEvidence) -> float:
        wins = sum(1 for w in evidence.windows if w.net_pnl > 0)
        return wins / len(evidence.windows)

    def _stress_net(self, window: ExperimentResult, stress: CostModel) -> float:
        return window.gross_pnl - stress.total_cost(window.num_trades, window.notional_per_trade)

    def _check_expectancy(self, evidence: ResearchEvidence) -> str | None:
        if not evidence.windows:
            return None  # covered by _check_windows_present
        if self._mean_expectancy(evidence) <= self.min_expectancy_per_trade:
            return "NET_EXPECTANCY_NOT_POSITIVE"
        return None

    def _check_walk_forward(self, evidence: ResearchEvidence) -> str | None:
        if not evidence.windows:
            return None  # covered by _check_windows_present
        if self._window_positive_fraction(evidence) < self.min_window_fraction:
            return "WALK_FORWARD_INCONSISTENT"
        return None

    def _check_stress_costs(self, evidence: ResearchEvidence) -> str | None:
        for window in evidence.windows:
            if self._stress_net(window, evidence.stress_cost_model) <= 0:
                return "STRESS_COSTS_NOT_SURVIVED"
        return None

    def _check_parameter_stability(self, evidence: ResearchEvidence) -> str | None:
        neighbors = evidence.param_neighbors
        if not neighbors:
            return "PARAMETER_SENSITIVITY_NOT_TESTED"
        for neighbor in neighbors:
            if (neighbor.strategy_id, neighbor.strategy_version) != (
                evidence.strategy_id,
                evidence.strategy_version,
            ):
                return "PARAMETER_NEIGHBORS_MIXED_IDENTITY"
        positive = sum(1 for n in neighbors if n.net_pnl > 0) / len(neighbors)
        if positive < self.min_neighbor_fraction:
            return "PARAMETER_STABILITY_FAILED"
        return None

    def _check_regimes(self, evidence: ResearchEvidence) -> str | None:
        if len(set(evidence.regimes_tested)) < self.min_regimes:
            return "REGIME_COVERAGE_INSUFFICIENT"
        return None

    def _check_paper_agreement(self, evidence: ResearchEvidence) -> str | None:
        paper = evidence.paper_result
        if paper is None:
            return "PAPER_RESULT_MISSING"
        if (paper.strategy_id, paper.strategy_version) != (
            evidence.strategy_id,
            evidence.strategy_version,
        ):
            return "PAPER_STRATEGY_MISMATCH"
        mean_exp = self._mean_expectancy(evidence)
        tolerance = max(self.paper_abs_tol, self.paper_rel_tol * abs(mean_exp))
        if abs(paper.expectancy_per_trade - mean_exp) > tolerance:
            return "PAPER_DISAGREES_WITH_SIMULATION"
        return None

    def _check_paper_positive(self, evidence: ResearchEvidence) -> str | None:
        paper = evidence.paper_result
        if paper is None:  # covered by _check_paper_agreement; stay silent here
            return None
        if paper.net_pnl <= 0 or paper.expectancy_per_trade <= self.min_expectancy_per_trade:
            return "PAPER_NOT_NET_POSITIVE"
        return None

    # -- public API ----------------------------------------------------------

    def evaluate(self, evidence: ResearchEvidence, target: ResearchStage) -> ResearchGateResult:
        """Evaluate a promotion to ``target``; never promotes *to* RESEARCH."""
        if target is ResearchStage.RESEARCH:
            raise ValueError("RESEARCH_GATE_INVALID_TARGET")

        failed: list[str] = []
        for check in (
            self._check_windows_present,
            self._check_windows_consistent,
            self._check_windows_chronological,
            self._check_expectancy,
            self._check_walk_forward,
            self._check_stress_costs,
            self._check_parameter_stability,
            self._check_regimes,
        ):
            reason = check(evidence)
            if reason is not None:
                failed.append(reason)

        if evidence.leakage_reviewed is not True:
            failed.append("LEAKAGE_REVIEW_MISSING")
        if evidence.risk_limits_tested is not True:
            failed.append("RISK_LIMITS_NOT_TESTED")

        if target in (
            ResearchStage.PAPER,
            ResearchStage.MICRO_LIVE,
            ResearchStage.LIMITED_LIVE,
        ):
            reason = self._check_paper_agreement(evidence)
            if reason is not None:
                failed.append(reason)

        if target in (ResearchStage.MICRO_LIVE, ResearchStage.LIMITED_LIVE):
            reason = self._check_paper_positive(evidence)
            if reason is not None:
                failed.append(reason)

        mean_exp = self._mean_expectancy(evidence)
        paper_diff = (
            abs(evidence.paper_result.expectancy_per_trade - mean_exp)
            if evidence.paper_result is not None
            else float("nan")
        )
        details = (
            ("mean_expectancy_per_trade", mean_exp),
            (
                "window_positive_fraction",
                self._window_positive_fraction(evidence) if evidence.windows else 0.0,
            ),
            (
                "stress_pass_fraction",
                sum(
                    1
                    for w in evidence.windows
                    if self._stress_net(w, evidence.stress_cost_model) > 0
                )
                / len(evidence.windows)
                if evidence.windows
                else 0.0,
            ),
            (
                "neighbor_positive_fraction",
                sum(1 for n in evidence.param_neighbors if n.net_pnl > 0)
                / len(evidence.param_neighbors)
                if evidence.param_neighbors
                else 0.0,
            ),
            ("distinct_regimes", float(len(set(evidence.regimes_tested)))),
            ("paper_expectancy_diff", paper_diff),
        )
        return ResearchGateResult(
            eligible=not failed,
            target=target,
            failed=tuple(failed),
            details=details,
        )
