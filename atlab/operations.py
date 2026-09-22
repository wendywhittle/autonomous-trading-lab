from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .ledger import ImmutableLedger
from .observability import LedgerHealth, PortfolioReconciliation, inspect_ledger, reconcile_portfolio


@dataclass(frozen=True)
class OperationalHealth:
    healthy: bool
    ledger: LedgerHealth
    portfolio: PortfolioReconciliation
    errors: tuple[str, ...]


def inspect_portfolio_file(
    path: str | Path,
    orders,
    *,
    starting_cash: float,
) -> PortfolioReconciliation:
    target = Path(path)
    try:
        state = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return PortfolioReconciliation(False, ("PORTFOLIO_STATE_UNREADABLE",))
    return reconcile_portfolio(state, orders, starting_cash=starting_cash)


def inspect_operational_health(
    ledger: ImmutableLedger,
    portfolio_state_path: str | Path,
    orders,
    *,
    starting_cash: float,
) -> OperationalHealth:
    ledger_health = inspect_ledger(ledger)
    portfolio_health = inspect_portfolio_file(
        portfolio_state_path,
        orders,
        starting_cash=starting_cash,
    )
    errors = tuple(ledger_health.errors + portfolio_health.errors)
    return OperationalHealth(
        healthy=not errors,
        ledger=ledger_health,
        portfolio=portfolio_health,
        errors=errors,
    )
