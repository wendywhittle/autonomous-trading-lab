from atlab.broker import (
    BrokerOrderRequest,
    BrokerOrderResult,
    BrokerOrderStatus,
    DisabledBroker,
)
from atlab.execution import ExecutionIntentStore
from atlab.coordinator import ExecutionCoordinator
from atlab.execution_reconciliation import inspect_execution_consistency
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
