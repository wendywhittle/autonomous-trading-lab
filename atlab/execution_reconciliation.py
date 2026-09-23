from __future__ import annotations

from dataclasses import dataclass

from .broker import BrokerOrderStatus
from .execution import ExecutionIntentStore
from .ledger import ImmutableLedger


@dataclass(frozen=True)
class ExecutionConsistency:
    healthy: bool
    errors: tuple[str, ...]


_TERMINAL = {
    BrokerOrderStatus.FILLED,
    BrokerOrderStatus.CANCELED,
    BrokerOrderStatus.REJECTED,
}


def inspect_execution_consistency(
    store: ExecutionIntentStore,
    ledger: ImmutableLedger,
) -> ExecutionConsistency:
    """Check durable execution intents against their immutable audit trail.

    This function is deliberately read-only. It detects disagreement but never
    repairs or mutates execution state. Repair requires an explicit provider
    reconciliation decision.
    """
    errors: list[str] = []
    events = ledger.read()

    audit_by_key: dict[str, list] = {}
    for event in events:
        key = event.payload.get("idempotency_key")
        if key is not None and event.event_type.startswith("EXECUTION_"):
            audit_by_key.setdefault(key, []).append(event)

    # Read the SQLite table directly through the store's existing safe decoder.
    connection = store._connect()
    try:
        rows = connection.execute(
            "SELECT idempotency_key FROM execution_intents ORDER BY idempotency_key"
        ).fetchall()
        keys = [row[0] for row in rows]
    finally:
        connection.close()

    known_keys = set(keys)
    for key in sorted(set(audit_by_key) - known_keys):
        errors.append(f"EXECUTION_AUDIT_ORPHAN:{key}")

    broker_ids: dict[str, list[str]] = {}
    for key in keys:
        intent = store.get(key)
        if intent is not None and intent.result is not None:
            broker_id = intent.result.broker_order_id
            if broker_id:
                broker_ids.setdefault(broker_id, []).append(key)

    for broker_id, bound_keys in sorted(broker_ids.items()):
        if len(bound_keys) > 1:
            errors.append(f"EXECUTION_BROKER_ORDER_ID_DUPLICATE:{broker_id}")

    for key in keys:
        intent = store.get(key)
        if intent is None:
            errors.append("EXECUTION_INTENT_DISAPPEARED")
            continue

        events_for_key = audit_by_key.get(key, [])
        created = [
            event for event in events_for_key
            if event.event_type == "EXECUTION_INTENT_CREATED"
        ]
        result_events = [
            event for event in events_for_key
            if event.event_type in {
                "EXECUTION_RESULT",
                "EXECUTION_UNKNOWN",
                "EXECUTION_RECOVERED",
                "EXECUTION_RECONCILED",
            }
        ]

        if len(created) != 1:
            errors.append(
                "EXECUTION_INTENT_AUDIT_COUNT_MISMATCH:"
                f"{key}:CREATED={len(created)}"
            )

        if intent.result is None:
            if result_events:
                errors.append(f"EXECUTION_STATE_MISSING_RESULT:{key}")
            continue

        if not result_events:
            errors.append(f"EXECUTION_RESULT_AUDIT_MISSING:{key}")
            continue

        for event in result_events:
            payload = event.payload
            if payload.get("idempotency_key") != key:
                errors.append(f"EXECUTION_AUDIT_KEY_MISMATCH:{key}")
            if event.event_type == "EXECUTION_UNKNOWN" and payload.get("status") != BrokerOrderStatus.UNKNOWN.value:
                errors.append(f"EXECUTION_UNKNOWN_AUDIT_MISMATCH:{key}")
            if event.event_type == "EXECUTION_RESULT" and payload.get("status") == BrokerOrderStatus.UNKNOWN.value:
                errors.append(f"EXECUTION_RESULT_AUDIT_UNKNOWN:{key}")

        latest = result_events[-1].payload
        expected = {
            "idempotency_key": key,
            "broker_order_id": intent.result.broker_order_id,
            "status": intent.result.status.value,
            "accepted": intent.result.accepted,
            "message": intent.result.message,
        }
        for field, value in expected.items():
            if latest.get(field) != value:
                errors.append(
                    f"EXECUTION_AUDIT_STATE_MISMATCH:{key}:{field}"
                )

        if intent.status is not intent.result.status:
            errors.append(f"EXECUTION_STATUS_RESULT_MISMATCH:{key}")

        if intent.status is BrokerOrderStatus.UNKNOWN and any(
            event.event_type in {"EXECUTION_RECOVERED", "EXECUTION_RECONCILED"}
            for event in result_events
        ):
            errors.append(f"EXECUTION_UNKNOWN_HAS_RECONCILIATION:{key}")

        if intent.status in _TERMINAL and not latest.get("status") in {
            status.value for status in _TERMINAL
        }:
            errors.append(f"EXECUTION_TERMINAL_AUDIT_MISMATCH:{key}")

    return ExecutionConsistency(not errors, tuple(errors))

def execution_requires_halt(consistency: ExecutionConsistency) -> bool:
    """Return whether autonomous execution must remain blocked."""
    return not consistency.healthy


def reconciliation_halt_reason(consistency: ExecutionConsistency) -> str:
    """Build a stable, explicit reason for a durable execution halt."""
    if consistency.healthy:
        raise ValueError("EXECUTION_HALT_NOT_REQUIRED")
    return "RECONCILIATION_FAILURE:" + "|".join(consistency.errors)


def assert_execution_halt(
    store: ExecutionIntentStore,
    consistency: ExecutionConsistency,
) -> bool:
    """Durably assert a halt for an unhealthy reconciliation result.

    This operation is monotonic: healthy inspection never clears a halt.
    """
    if consistency.healthy:
        return False
    store.halt(reconciliation_halt_reason(consistency))
    return True
