"""Research protocol: walk-forward evaluation, cost-aware experiments, and promotion gates.

Ports the Sniper v1 research discipline (see ``sniper-v1-spec.md``) into the
lab at daily-bar scale:

- :mod:`atlab.research.splits` — chronological train/validate/test splits and
  walk-forward windows. Time series are never shuffled.
- :mod:`atlab.research.costs` — explicit all-in cost models (fees, slippage,
  spread) so every result is net of execution costs.
- :mod:`atlab.research.experiments` — reproducible experiment records carrying
  results, plus a registry and a walk-forward runner.
- :mod:`atlab.research.gates` — the research promotion gate: net expectancy,
  stress costs, walk-forward consistency, regime coverage, leakage review,
  parameter stability, and paper agreement.
- :mod:`atlab.research.report` — the confidence-bucket table (does higher
  model confidence actually correspond to higher realized net expectancy?).
- :mod:`atlab.research.config` — default research configuration.
"""

from .config import ResearchConfig
from .costs import CostModel
from .experiments import (
    ExperimentRegistry,
    ExperimentResult,
    ResearchStage,
    run_experiment,
    run_walk_forward,
)
from .gates import ResearchEvidence, ResearchGate, ResearchGateResult
from .report import ConfidenceBucketRow, ScoredTrade, confidence_bucket_table
from .splits import ChronoSplit, WalkForwardWindow, chronological_split, walk_forward_windows

__all__ = [
    "ChronoSplit",
    "ConfidenceBucketRow",
    "CostModel",
    "ExperimentRegistry",
    "ExperimentResult",
    "ResearchConfig",
    "ResearchEvidence",
    "ResearchGate",
    "ResearchGateResult",
    "ResearchStage",
    "ScoredTrade",
    "WalkForwardWindow",
    "chronological_split",
    "confidence_bucket_table",
    "run_experiment",
    "run_walk_forward",
    "walk_forward_windows",
]
