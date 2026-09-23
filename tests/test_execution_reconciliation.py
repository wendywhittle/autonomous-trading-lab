import sqlite3

import pytest

from atlab.broker import (
    BrokerOrderRequest,
    BrokerOrderResult,
    BrokerOrderStatus,
    DisabledBroker,
)
from atlab.coordinator import ExecutionCoordinator
from atlab.execution import ExecutionIntentStore
from atlab.execution_reconciliation import assert_execution_halt, inspect_execution_consistency
from atlab.ledger import ImmutableLedger
from atlab.models import Side


def request():
    return BrokerOrderRequest(
        idempotency_key="consistency-1",
        symbol="TEST",
        side=Side.BUY,
        quantity=1,
        price=100,
    )


def setup(tmp_path):
    db = tmp_path / "execution.sqlite3"
    store = ExecutionIntentStore(db)
    ledger = ImmutableLedger(db)
    coordinator = ExecutionCoordinator(store, DisabledBroker(), ledger)
    return coordinator, store, ledger


def test_execution_consistency_is_healthy_for_durable_unknown(tmp_path):
    coordinator, store, ledger = setup(tmp_path)
    coordinator.submit(request())

    result = inspect_execution_consistency(store, ledger)

    assert result.healthy
    assert result.errors == ()


def test_execution_consistency_detects_missing_audit_result(tmp_path):
    coordinator, store, ledger = setup(tmp_path)
    coordinator.prepare(request())
    store.update_result(
        "consistency-1",
        BrokerOrderResult(
            accepted=True,
            broker_order_id="broker-1",
            status=BrokerOrderStatus.ACCEPTED,
            message="ACCEPTED",
        ),
    )

    result = inspect_execution_consistency(store, ledger)

    assert not result.healthy
    assert "EXECUTION_RESULT_AUDIT_MISSING:consistency-1" in result.errors


def test_execution_consistency_detects_audit_state_conflict(tmp_path):
    coordinator, store, ledger = setup(tmp_path)
    coordinator.submit(request())

    # Deliberately corrupt only the audit payload to model an external/manual
    # mutation. The consistency layer must report, never repair, the conflict.
    connection = sqlite3.connect(ledger.path)
    connection.execute(
        """
        UPDATE events
        SET payload = ?
        WHERE event_type = 'EXECUTION_RESULT'
        """,
        ('{"accepted":true,"broker_order_id":"wrong","idempotency_key":"consistency-1","message":"CORRUPTED","status":"FILLED"}',),
    )
    connection.commit()
    connection.close()

    result = inspect_execution_consistency(store, ledger)

    assert not result.healthy
    assert any(
        error.startswith("EXECUTION_AUDIT_STATE_MISMATCH:consistency-1:")
        for error in result.errors
    )


def test_execution_consistency_detects_missing_created_audit(tmp_path):
    coordinator, store, ledger = setup(tmp_path)
    coordinator.prepare(request())

    connection = sqlite3.connect(ledger.path)
    connection.execute(
        "DELETE FROM events WHERE event_type = 'EXECUTION_INTENT_CREATED'"
    )
    connection.commit()
    connection.close()

    result = inspect_execution_consistency(store, ledger)

    assert not result.healthy
    assert "EXECUTION_INTENT_AUDIT_COUNT_MISMATCH:consistency-1:CREATED=0" in result.errors


def test_execution_consistency_detects_orphan_audit(tmp_path):
    _coordinator, store, ledger = setup(tmp_path)
    ledger.append(
        "EXECUTION_RESULT",
        "execution-result-orphan",
        {
            "idempotency_key": "orphan-key",
            "broker_order_id": "broker-orphan",
            "status": "FILLED",
            "accepted": True,
            "message": "ORPHAN",
        },
    )

    result = inspect_execution_consistency(store, ledger)

    assert not result.healthy
    assert "EXECUTION_AUDIT_ORPHAN:orphan-key" in result.errors


def test_execution_consistency_detects_duplicate_broker_binding(tmp_path):
    coordinator, store, ledger = setup(tmp_path)
    first = request()
    second = BrokerOrderRequest(
        idempotency_key="consistency-2",
        symbol="TEST",
        side=Side.BUY,
        quantity=2,
        price=100,
    )
    coordinator.prepare(first)
    coordinator.prepare(second)
    shared = BrokerOrderResult(
        accepted=True,
        broker_order_id="shared-broker-id",
        status=BrokerOrderStatus.ACCEPTED,
        message="ACCEPTED",
    )
    coordinator._commit_result(first.idempotency_key, shared, "EXECUTION_RESULT")
    coordinator._commit_result(second.idempotency_key, shared, "EXECUTION_RESULT")

    result = inspect_execution_consistency(store, ledger)

    assert not result.healthy
    assert "EXECUTION_BROKER_ORDER_ID_DUPLICATE:shared-broker-id" in result.errors


def test_unhealthy_reconciliation_asserts_durable_halt(tmp_path):
    coordinator, store, ledger = setup(tmp_path)
    coordinator.prepare(request())
    store.update_result(
        "consistency-1",
        BrokerOrderResult(
            accepted=True,
            broker_order_id="broker-1",
            status=BrokerOrderStatus.ACCEPTED,
            message="ACCEPTED",
        ),
    )

    consistency = inspect_execution_consistency(store, ledger)
    assert assert_execution_halt(store, consistency)
    assert store.is_halted()
    assert store.halt_reason().startswith("RECONCILIATION_FAILURE:")


def test_healthy_reconciliation_does_not_clear_existing_halt(tmp_path):
    coordinator, store, ledger = setup(tmp_path)
    coordinator.submit(request())
    store.halt("MANUAL_HOLD")

    consistency = inspect_execution_consistency(store, ledger)
    assert consistency.healthy
    assert not assert_execution_halt(store, consistency)
    assert store.halt_reason() == "MANUAL_HOLD"


def test_clear_halt_requires_healthy_reconciliation(tmp_path):
    coordinator, store, ledger = setup(tmp_path)
    coordinator.prepare(request())
    store.update_result(
        "consistency-1",
        BrokerOrderResult(
            accepted=True,
            broker_order_id="broker-1",
            status=BrokerOrderStatus.ACCEPTED,
            message="ACCEPTED",
        ),
    )
    consistency = inspect_execution_consistency(store, ledger)
    assert assert_execution_halt(store, consistency)

    import pytest
    with pytest.raises(RuntimeError, match="EXECUTION_HALT_CLEAR_BLOCKED"):
        coordinator.clear_halt("operator-1")


def test_clear_halt_is_audited_after_clean_reconciliation(tmp_path):
    coordinator, store, ledger = setup(tmp_path)
    coordinator.submit(request())
    store.halt("MANUAL_HOLD")

    coordinator.clear_halt("operator-1")

    assert not store.is_halted()
    events = ledger.read()
    assert any(
        event.event_type == "EXECUTION_HALT_CLEARED"
        and event.payload["operator_reference"] == "operator-1"
        for event in events
    )


def test_provider_discovery_halts_on_order_missing_local_intent(tmp_path):
    from atlab.broker import BrokerDiscoveredOrder, BrokerMode

    class DiscoveringBroker(DisabledBroker):
        mode = BrokerMode.LIVE

        def discover_open_orders(self):
            return (
                BrokerDiscoveredOrder(
                    idempotency_key="provider-only",
                    result=BrokerOrderResult(
                        accepted=True,
                        broker_order_id="broker-provider-only",
                        status=BrokerOrderStatus.ACCEPTED,
                        message="ACCEPTED",
                    ),
                ),
            )

    db = tmp_path / "execution.sqlite3"
    store = ExecutionIntentStore(db)
    ledger = ImmutableLedger(db)
    coordinator = ExecutionCoordinator(store, DiscoveringBroker(), ledger)

    import pytest
    with pytest.raises(RuntimeError, match="EXECUTION_PROVIDER_ORPHAN_DETECTED"):
        coordinator.reconcile_discovered_orders()

    assert store.is_halted()
    assert "EXECUTION_PROVIDER_ORDER_ORPHAN:broker-provider-only" in store.halt_reason()
    assert any(
        event.event_type == "EXECUTION_HALT_ASSERTED"
        and "EXECUTION_PROVIDER_ORDER_ORPHAN:broker-provider-only" in event.payload["errors"]
        for event in ledger.read()
    )


def test_provider_discovery_reconciles_known_intent(tmp_path):
    from atlab.broker import BrokerDiscoveredOrder, BrokerMode

    class DiscoveringBroker(DisabledBroker):
        mode = BrokerMode.LIVE

        def discover_open_orders(self):
            return (
                BrokerDiscoveredOrder(
                    idempotency_key="consistency-1",
                    result=BrokerOrderResult(
                        accepted=True,
                        broker_order_id="broker-discovered",
                        status=BrokerOrderStatus.ACCEPTED,
                        message="ACCEPTED",
                    ),
                ),
            )

    db = tmp_path / "execution.sqlite3"
    store = ExecutionIntentStore(db)
    ledger = ImmutableLedger(db)
    coordinator = ExecutionCoordinator(store, DiscoveringBroker(), ledger)
    coordinator.prepare(request())

    reconciled = coordinator.reconcile_discovered_orders()

    assert len(reconciled) == 1
    assert reconciled[0].idempotency_key == "consistency-1"
    assert store.get("consistency-1").result.broker_order_id == "broker-discovered"


def test_execution_status_transitions_are_monotonic(tmp_path):
    coordinator, store, _ledger = setup(tmp_path)
    coordinator.prepare(request())

    accepted = BrokerOrderResult(
        accepted=True,
        broker_order_id="broker-1",
        status=BrokerOrderStatus.ACCEPTED,
        message="ACCEPTED",
    )
    partial = BrokerOrderResult(
        accepted=True,
        broker_order_id="broker-1",
        status=BrokerOrderStatus.PARTIALLY_FILLED,
        message="PARTIAL",
        filled_quantity=0.37,
        remaining_quantity=0.63,
    )
    filled = BrokerOrderResult(
        accepted=True,
        broker_order_id="broker-1",
        status=BrokerOrderStatus.FILLED,
        message="FILLED",
    )

    store.update_result("consistency-1", accepted)
    store.update_result("consistency-1", partial)
    store.update_result("consistency-1", filled)

    import pytest
    with pytest.raises(ValueError, match="EXECUTION_TERMINAL_STATE_CONFLICT"):
        store.update_result(
            "consistency-1",
            BrokerOrderResult(
                accepted=True,
                broker_order_id="broker-1",
                status=BrokerOrderStatus.CANCELED,
                message="CANCELED_AFTER_FILL",
            ),
        )


def test_execution_partial_fill_cannot_regress_to_unknown(tmp_path):
    coordinator, store, _ledger = setup(tmp_path)
    coordinator.prepare(request())
    store.update_result(
        "consistency-1",
        BrokerOrderResult(
            accepted=True,
            broker_order_id="broker-1",
            status=BrokerOrderStatus.PARTIALLY_FILLED,
            message="PARTIAL",
            filled_quantity=0.37,
            remaining_quantity=0.63,
        ),
    )

    import pytest
    with pytest.raises(ValueError, match="EXECUTION_STATE_TRANSITION_CONFLICT"):
        store.update_result(
            "consistency-1",
            BrokerOrderResult(
                accepted=False,
                broker_order_id="broker-1",
                status=BrokerOrderStatus.UNKNOWN,
                message="UNKNOWN",
            ),
        )


def test_partial_fill_quantities_progress_without_double_counting(tmp_path):
    coordinator, store, ledger = setup(tmp_path)
    coordinator.prepare(request())

    partial = BrokerOrderResult(
        accepted=True,
        broker_order_id="broker-1",
        status=BrokerOrderStatus.PARTIALLY_FILLED,
        message="PARTIAL_37",
        filled_quantity=0.37,
        remaining_quantity=0.63,
    )
    final_partial = BrokerOrderResult(
        accepted=True,
        broker_order_id="broker-1",
        status=BrokerOrderStatus.PARTIALLY_FILLED,
        message="PARTIAL_62",
        filled_quantity=0.62,
        remaining_quantity=0.38,
    )
    filled = BrokerOrderResult(
        accepted=True,
        broker_order_id="broker-1",
        status=BrokerOrderStatus.FILLED,
        message="FILLED",
        filled_quantity=1.0,
        remaining_quantity=0.0,
    )

    coordinator._commit_result("consistency-1", partial, "EXECUTION_RECONCILED")
    coordinator._commit_result("consistency-1", final_partial, "EXECUTION_RECONCILED")
    coordinator._commit_result("consistency-1", filled, "EXECUTION_RECONCILED")

    intent = store.get("consistency-1")
    assert intent.result == filled
    events = ledger.read()
    assert len(
        [event for event in events if event.event_type == "EXECUTION_RECONCILED"]
    ) == 3
    assert inspect_execution_consistency(store, ledger).healthy


def test_partial_fill_cannot_regress_quantity(tmp_path):
    coordinator, store, _ledger = setup(tmp_path)
    coordinator.prepare(request())
    coordinator._commit_result(
        "consistency-1",
        BrokerOrderResult(
            accepted=True,
            broker_order_id="broker-1",
            status=BrokerOrderStatus.PARTIALLY_FILLED,
            message="PARTIAL_37",
            filled_quantity=0.37,
            remaining_quantity=0.63,
        ),
        "EXECUTION_RECONCILED",
    )

    with pytest.raises(ValueError, match="EXECUTION_FILL_QUANTITY_REGRESSION"):
        store.update_result(
            "consistency-1",
            BrokerOrderResult(
                accepted=True,
                broker_order_id="broker-1",
                status=BrokerOrderStatus.PARTIALLY_FILLED,
                message="PARTIAL_20",
                filled_quantity=0.20,
                remaining_quantity=0.80,
            ),
        )


def test_filled_result_must_account_for_entire_requested_quantity(tmp_path):
    coordinator, store, _ledger = setup(tmp_path)
    coordinator.prepare(request())

    with pytest.raises(ValueError, match="INVALID_FILLED_QUANTITY"):
        store.update_result(
            "consistency-1",
            BrokerOrderResult(
                accepted=True,
                broker_order_id="broker-1",
                status=BrokerOrderStatus.FILLED,
                message="BAD_FILL",
                filled_quantity=0.80,
                remaining_quantity=0.20,
            ),
        )


def test_partial_fill_requires_exact_quantity_total(tmp_path):
    coordinator, store, _ledger = setup(tmp_path)
    coordinator.prepare(request())

    with pytest.raises(ValueError, match="EXECUTION_FILL_QUANTITY_TOTAL_MISMATCH"):
        store.update_result(
            "consistency-1",
            BrokerOrderResult(
                accepted=True,
                broker_order_id="broker-1",
                status=BrokerOrderStatus.PARTIALLY_FILLED,
                message="BAD_TOTAL",
                filled_quantity=0.37,
                remaining_quantity=0.62,
            ),
        )


def test_identical_partial_fill_is_idempotent(tmp_path):
    coordinator, _store, ledger = setup(tmp_path)
    coordinator.prepare(request())
    partial = BrokerOrderResult(
        accepted=True,
        broker_order_id="broker-1",
        status=BrokerOrderStatus.PARTIALLY_FILLED,
        message="PARTIAL",
        filled_quantity=0.37,
        remaining_quantity=0.63,
    )

    coordinator._commit_result("consistency-1", partial, "EXECUTION_RECONCILED")
    coordinator._commit_result("consistency-1", partial, "EXECUTION_RECONCILED")

    assert len(
        [event for event in ledger.read() if event.event_type == "EXECUTION_RECONCILED"]
    ) == 1


def test_fill_audit_quantity_corruption_forces_unhealthy_reconciliation(tmp_path):
    coordinator, store, ledger = setup(tmp_path)
    coordinator.prepare(request())
    coordinator._commit_result(
        "consistency-1",
        BrokerOrderResult(
            accepted=True,
            broker_order_id="broker-1",
            status=BrokerOrderStatus.PARTIALLY_FILLED,
            message="PARTIAL",
            filled_quantity=0.37,
            remaining_quantity=0.63,
        ),
        "EXECUTION_RECONCILED",
    )

    connection = sqlite3.connect(ledger.path)
    connection.execute(
        """
        UPDATE events
        SET payload = REPLACE(payload, '"filled_quantity":0.37', '"filled_quantity":0.20')
        WHERE event_type = 'EXECUTION_RECONCILED'
        """
    )
    connection.commit()
    connection.close()

    result = inspect_execution_consistency(store, ledger)
    assert not result.healthy
    assert "EXECUTION_AUDIT_STATE_MISMATCH:consistency-1:filled_quantity" in result.errors
