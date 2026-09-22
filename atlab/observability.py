from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ledger import ImmutableLedger
from .models import PaperOrder, Side


@dataclass(frozen=True)
class LedgerHealth:
    healthy: bool
    event_count: int
    latest_sequence: int | None
    event_type_counts: tuple[tuple[str, int], ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class PortfolioReconciliation:
    reconciled: bool
    errors: tuple[str, ...]


def inspect_ledger(ledger: ImmutableLedger) -> LedgerHealth:
    errors: list[str] = []
    try:
        events = ledger.read()
    except Exception as exc:
        return LedgerHealth(
            healthy=False,
            event_count=0,
            latest_sequence=None,
            event_type_counts=(),
            errors=(f"LEDGER_READ_ERROR:{type(exc).__name__}",),
        )

    sequences = [event.sequence for event in events]
    if sequences != list(range(len(events))):
        errors.append("LEDGER_SEQUENCE_INVALID")

    event_ids = [event.event_id for event in events]
    if len(set(event_ids)) != len(event_ids):
        errors.append("LEDGER_EVENT_ID_DUPLICATE")

    counts: dict[str, int] = {}
    for event in events:
        counts[event.event_type] = counts.get(event.event_type, 0) + 1

    return LedgerHealth(
        healthy=not errors,
        event_count=len(events),
        latest_sequence=events[-1].sequence if events else None,
        event_type_counts=tuple(sorted(counts.items())),
        errors=tuple(errors),
    )


def reconcile_portfolio(
    portfolio_state: dict[str, Any],
    orders: list[PaperOrder],
    *,
    starting_cash: float,
) -> PortfolioReconciliation:
    errors: list[str] = []

    required = {
        "cash",
        "position_quantity",
        "average_cost",
        "realized_pnl",
        "applied_order_ids",
    }
    if set(portfolio_state) != required:
        return PortfolioReconciliation(False, ("INVALID_PORTFOLIO_STATE",))

    applied_ids = portfolio_state["applied_order_ids"]
    if not isinstance(applied_ids, list) or not all(
        isinstance(item, str) and item for item in applied_ids
    ):
        return PortfolioReconciliation(False, ("INVALID_APPLIED_ORDER_IDS",))

    if len(set(applied_ids)) != len(applied_ids):
        errors.append("APPLIED_ORDER_ID_DUPLICATE")

    order_by_id = {order.order_id: order for order in orders}
    if len(order_by_id) != len(orders):
        errors.append("LEDGER_ORDER_ID_DUPLICATE")

    missing = sorted(set(applied_ids) - set(order_by_id))
    if missing:
        errors.append("PORTFOLIO_ORDER_MISSING_FROM_LEDGER")

    expected_cash = float(starting_cash)
    expected_quantity = 0.0
    expected_average_cost = 0.0
    expected_realized_pnl = 0.0

    for order in orders:
        if order.order_id not in applied_ids:
            continue
        if order.side is Side.BUY:
            new_quantity = expected_quantity + order.quantity
            expected_average_cost = (
                expected_quantity * expected_average_cost + order.notional
            ) / new_quantity
            expected_quantity = new_quantity
            expected_cash -= order.notional
        else:
            if order.quantity > expected_quantity:
                errors.append("LEDGER_SELL_EXCEEDS_RECONCILED_POSITION")
                continue
            expected_realized_pnl += (
                order.fill_price - expected_average_cost
            ) * order.quantity
            expected_quantity -= order.quantity
            expected_cash += order.notional
            if expected_quantity == 0:
                expected_average_cost = 0.0

    comparisons = (
        ("cash", expected_cash),
        ("position_quantity", expected_quantity),
        ("average_cost", expected_average_cost),
        ("realized_pnl", expected_realized_pnl),
    )
    for field, expected in comparisons:
        actual = float(portfolio_state[field])
        if abs(actual - expected) > 1e-9:
            errors.append(f"PORTFOLIO_{field.upper()}_MISMATCH")

    return PortfolioReconciliation(not errors, tuple(errors))
