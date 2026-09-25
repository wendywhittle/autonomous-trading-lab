"""Default research configuration.

:func:`ResearchConfig.defaults` is the single source of truth used by the
research package. ``config/research.yaml`` mirrors these values for operators;
the YAML is documentation until a loader lands, so if they ever diverge, this
module wins and the YAML must be updated to match.

Every number here is an initial research parameter, not a sacred constant —
the Sniper v1 spec is explicit that thresholds like these are to be optimized
through out-of-sample testing, never hard-coded as truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .costs import CostModel


def _base_costs() -> CostModel:
    return CostModel(
        name="retail-base", fee_flat=0.0, fee_bps=0.0, slippage_bps=10.0, spread_bps=5.0
    )


def _stress_costs() -> CostModel:
    return CostModel(
        name="retail-stress",
        fee_flat=0.0,
        fee_bps=0.0,
        slippage_bps=20.0,
        spread_bps=10.0,
    )


@dataclass(frozen=True)
class ResearchConfig:
    """Walk-forward geometry, cost models, and gate thresholds for research."""

    # Walk-forward geometry, in bars (daily bars: ~21 per month).
    train_size: int = 63  # ~3 months, per the spec's example
    validate_size: int = 21  # ~1 month
    test_size: int = 21  # ~1 month
    step: int = 21  # roll forward one month per window
    expanding: bool = True
    embargo: int = 0

    # All-in cost assumptions. Base = realistic retail limit-order costs;
    # stress = harsher assumptions the edge must also survive.
    base_costs: CostModel = field(default_factory=_base_costs)
    stress_costs: CostModel = field(default_factory=_stress_costs)

    # Research gate thresholds.
    min_expectancy_per_trade: float = 0.0
    min_window_fraction: float = 1.0  # every walk-forward window net positive
    min_neighbor_fraction: float = 0.8  # 80% of nearby params net positive
    min_regimes: int = 2
    paper_rel_tol: float = 0.5
    paper_abs_tol: float = 0.05  # dollars per trade

    # Declared per-trade notional assumption for cost math (charter: $10/trade).
    notional_per_trade: float = 10.0

    @classmethod
    def defaults(cls) -> ResearchConfig:
        return cls()


DEFAULTS = ResearchConfig.defaults()
