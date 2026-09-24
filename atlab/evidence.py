"""Locally-verified promotion evidence collection (H9 audit fix).

Every evidence field records its provenance:

- ``"verified"``: the CLI checked it locally just now and it held.
- ``"attested"``: a human asserted it via an explicit CLI flag; it was NOT
  independently verified.

Nothing is inferred, assumed, or carried over from a previous run. Fields
that cannot be verified locally (there is no live broker adapter, no live
credentials, and no live control plane in this repo) are reported as False
with an explicit reason instead of being attested silently.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from . import broker as broker_module
from .adapters import CsvMarketData
from .broker import BrokerMode
from .ledger import ImmutableLedger
from .models import DecisionAction, JEVDecision, Side, StrategyVersion
from .paper import PaperExecution
from .portfolio import PaperPortfolio
from .promotion import PromotionEvidence, PromotionMode
from .replay import validate_ledger_sequence
from .risk import DeterministicRiskEngine, RiskLimits
from .runtime import PaperTradingEngine, TradingMode

#: Environment variable name patterns treated as live brokerage credentials
#: for the no-live-credentials check. A variable matches if its name contains
#: any of these substrings (case-insensitive).
LIVE_CREDENTIAL_ENV_PATTERNS = (
    "BROKER_API_KEY",
    "BROKER_API_SECRET",
    "BROKER_TOKEN",
    "LIVE_API_KEY",
    "LIVE_API_SECRET",
    "ATLAB_LIVE_",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def check_tests_green() -> bool:
    """Run the repo's pytest suite; True iff it passes (live-JEV skips OK).

    Tests marked ``promote_cli`` are deselected: they invoke this collector
    themselves, and running them inside the evidence subprocess would
    recurse. Everything else in the suite runs for real.
    """
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            "not promote_cli",
            "tests",
        ],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    return proc.returncode == 0


def check_paper_run_complete(workdir: Path) -> bool:
    """True iff the workdir holds a completed paper-run record."""
    record_path = workdir / "paper_run.json"
    if not record_path.exists():
        return False
    import json

    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(record.get("completed"))


def check_risk_limits_active() -> bool:
    """True iff the deterministic risk engine's limits are active."""
    limits = DeterministicRiskEngine().limits
    values = (
        limits.max_position_notional,
        limits.max_order_notional,
        limits.max_daily_loss,
        limits.max_drawdown,
        limits.max_leverage,
    )
    return all(math.isfinite(v) and v > 0 for v in values)


def _probe_decision() -> JEVDecision:
    return JEVDecision(
        decision_id="kill-switch-probe",
        strategy_id="probe",
        strategy_version="1",
        symbol="TEST",
        action=DecisionAction.ENTER,
        side=Side.BUY,
        confidence=1.0,
        rationale="promotion evidence probe",
        state_fingerprint="probe",
    )


def check_kill_switch_verified() -> bool:
    """Exercise the real kill switch: armed must block, disarmed must pass."""
    decision = _probe_decision()
    try:
        PaperExecution(kill_switch=True).submit(decision, 1.0, 100.0)
        return False
    except RuntimeError as exc:
        if str(exc) != "KILL_SWITCH_ACTIVE":
            return False
    order = PaperExecution(kill_switch=False).submit(decision, 1.0, 100.0)
    return order.status.value == "FILLED"


def check_replay_deterministic(
    csv_path: str, symbol: str, strategy: StrategyVersion, quantity: float
) -> bool:
    """Run the paper loop twice over the CSV; decisions and fills must match."""
    data = CsvMarketData(csv_path)
    observations = sorted(
        data.observations(symbol), key=lambda item: item.timestamp
    )
    if not observations:
        return False

    def run_once() -> list[tuple]:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            engine = PaperTradingEngine(
                CsvMarketData(csv_path),
                strategy,
                DeterministicRiskEngine(),
                PaperPortfolio(10_000.0),
                ImmutableLedger(workdir / "ledger.sqlite3"),
                quantity=quantity,
                mode=TradingMode.PAPER,
            )
            cycles = engine.run(symbol)
            # The persisted ledger must validate cleanly (sequence
            # contiguity and unique event IDs).
            validate_ledger_sequence(engine.ledger.read())
            return [
                (
                    cycle.decision.decision_id,
                    cycle.decision.action.value,
                    None
                    if cycle.order is None
                    else (
                        cycle.order.order_id,
                        cycle.order.side.value,
                        cycle.order.quantity,
                        cycle.order.fill_price,
                    ),
                )
                for cycle in cycles
            ]

    first, second = run_once(), run_once()
    return bool(first) and first == second


def live_credential_env_vars() -> list[str]:
    """Environment variable names matching live-credential patterns."""
    names = []
    for name in os.environ:
        upper = name.upper()
        if any(pattern in upper for pattern in LIVE_CREDENTIAL_ENV_PATTERNS):
            names.append(name)
    return sorted(names)


def live_broker_adapters() -> list[str]:
    """Concrete broker adapter classes in this repo advertising LIVE mode."""
    found = []
    for attr_name in dir(broker_module):
        attr = getattr(broker_module, attr_name)
        if (
            isinstance(attr, type)
            and isinstance(getattr(attr, "mode", None), BrokerMode)
            and attr.mode is BrokerMode.LIVE
        ):
            found.append(attr_name)
    return sorted(found)


def collect_evidence(
    *,
    workdir: Path,
    csv_path: str,
    symbol: str,
    strategy: StrategyVersion,
    quantity: float,
    human_approval_attestation: str | None,
) -> tuple[PromotionEvidence, dict[str, str]]:
    """Collect promotion evidence, each field tagged verified/attested."""
    provenance: dict[str, str] = {}

    def verified(name: str, value: bool) -> bool:
        provenance[name] = "verified"
        return value

    tests_green = verified("tests_green", check_tests_green())
    paper_run_complete = verified(
        "paper_run_complete", check_paper_run_complete(workdir)
    )
    risk_limits_active = verified("risk_limits_active", check_risk_limits_active())
    kill_switch_verified = verified(
        "kill_switch_verified", check_kill_switch_verified()
    )
    replay_deterministic = verified(
        "replay_deterministic",
        check_replay_deterministic(csv_path, symbol, strategy, quantity),
    )
    no_live_credentials = verified(
        "no_live_credentials", not live_credential_env_vars()
    )

    # There is no live broker adapter in this repo; this is verified by
    # scanning the broker module rather than assumed.
    live_adapters = live_broker_adapters()
    live_execution_implemented = verified(
        "live_execution_implemented", bool(live_adapters)
    )
    live_controls_verified = verified("live_controls_verified", False)
    if human_approval_attestation:
        provenance["human_approval"] = "attested"
        human_approval = True
    else:
        provenance["human_approval"] = "verified"
        human_approval = False
    # No credential-configuration mechanism exists in the CLI; verified
    # absent via the same environment scan.
    live_credentials_configured = verified(
        "live_credentials_configured", bool(live_credential_env_vars())
    )

    evidence = PromotionEvidence(
        tests_green=tests_green,
        paper_run_complete=paper_run_complete,
        risk_limits_active=risk_limits_active,
        kill_switch_verified=kill_switch_verified,
        replay_deterministic=replay_deterministic,
        no_live_credentials=no_live_credentials,
        live_execution_implemented=live_execution_implemented,
        live_controls_verified=live_controls_verified,
        human_approval=human_approval,
        live_credentials_configured=live_credentials_configured,
    )
    return evidence, provenance


def requirement_rows(target: PromotionMode) -> tuple[tuple[str, str], ...]:
    """(evidence field, failure reason) rows the gate evaluates for target."""
    from .promotion import PromotionGate

    rows = tuple(PromotionGate.BASE_REQUIREMENTS)
    if target is PromotionMode.PAPER:
        rows += tuple(PromotionGate.PAPER_REQUIREMENTS)
    elif target is PromotionMode.LIVE:
        rows += tuple(PromotionGate.LIVE_REQUIREMENTS)
    return rows


def live_evidence_notes() -> dict[str, str]:
    """Human-readable reasons for the non-verifiable LIVE evidence fields."""
    return {
        "live_execution_implemented": (
            "no BrokerMode.LIVE adapter exists in atlab.broker "
            f"(found: {live_broker_adapters() or 'none'})"
        ),
        "live_controls_verified": "no live control plane exists to verify",
        "human_approval": "requires explicit --attest-human-approval",
        "live_credentials_configured": (
            "no live credentials configured "
            f"(env scan: {live_credential_env_vars() or 'none found'})"
        ),
    }


def risk_limits_notes() -> str:
    limits: RiskLimits = DeterministicRiskEngine().limits
    return (
        f"max_position_notional={limits.max_position_notional} "
        f"max_order_notional={limits.max_order_notional} "
        f"max_daily_loss={limits.max_daily_loss} "
        f"max_drawdown={limits.max_drawdown} "
        f"max_leverage={limits.max_leverage}"
    )
