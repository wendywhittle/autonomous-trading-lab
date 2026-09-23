from atlab.broker import (
    BrokerOrderRequest,
    BrokerOrderResult,
    BrokerOrderStatus,
    DisabledBroker,
)
from atlab.execution import ExecutionIntentStore
from atlab.coordinator import ExecutionCoordinator
from atlab.execution_reconciliation import inspect_execution_consistency, assert_execution_halt
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
    import sqlite3

    connection = sqlite3.connect(ledger.path)
    connection.execute(
        """
        UPDATE events
        SET payload = ?
        WHERE event_type = 'EXECUTION_UNKNOWN'
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
    coordinator, store, ledger = setup(tmp_path)
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
