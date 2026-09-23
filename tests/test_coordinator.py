import concurrent.futures

import pytest

from atlab.broker import (
    BrokerMode,
    BrokerOrderRequest,
    BrokerOrderResult,
    BrokerOrderStatus,
    BrokerRecoveryResult,
    DisabledBroker,
)
from atlab.coordinator import ExecutionCoordinator, IntentStatus
from atlab.execution import ExecutionIntentStore
from atlab.ledger import ImmutableLedger
from atlab.models import Side


class FailingBroker:
    mode = BrokerMode.LIVE

    def submit(self, request):
        raise TimeoutError("provider timeout")

    def cancel(self, broker_order_id):
        return False

    def get_order(self, broker_order_id):
        return BrokerOrderResult(
            accepted=False,
            broker_order_id=broker_order_id,
            status=BrokerOrderStatus.UNKNOWN,
            message="NOT_FOUND",
        )

    def reconcile(self, broker_order_ids):
        return ()

    def reconcile_by_idempotency_keys(self, idempotency_keys):
        return ()


class AcceptedBroker:
    mode = BrokerMode.LIVE

    def __init__(self):
        self.submissions = 0

    def submit(self, request):
        self.submissions += 1
        return BrokerOrderResult(
            accepted=True,
            broker_order_id="broker-1",
            status=BrokerOrderStatus.ACCEPTED,
            message="ACCEPTED",
        )

    def cancel(self, broker_order_id):
        return True

    def get_order(self, broker_order_id):
        return BrokerOrderResult(
            accepted=True,
            broker_order_id=broker_order_id,
            status=BrokerOrderStatus.FILLED,
            message="FILLED",
        )

    def reconcile(self, broker_order_ids):
        return tuple(self.get_order(order_id) for order_id in broker_order_ids)

    def reconcile_by_idempotency_keys(self, idempotency_keys):
        return tuple(
            BrokerRecoveryResult(
                idempotency_key=key,
                result=self.get_order("broker-1"),
            )
            for key in idempotency_keys
        )


class CrashAfterAcceptanceBroker:
    """Simulates provider durability followed by a client-visible timeout."""

    mode = BrokerMode.LIVE

    def __init__(self):
        self.submissions = 0
        self.orders = {}

    def submit(self, request):
        self.submissions += 1
        result = BrokerOrderResult(
            accepted=True,
            broker_order_id="broker-crash-1",
            status=BrokerOrderStatus.ACCEPTED,
            message="ACCEPTED_BEFORE_CLIENT_FAILURE",
        )
        self.orders[request.idempotency_key] = result
        raise TimeoutError("response lost after provider acceptance")

    def cancel(self, broker_order_id):
        return True

    def get_order(self, broker_order_id):
        for result in self.orders.values():
            if result.broker_order_id == broker_order_id:
                return result
        return BrokerOrderResult(
            accepted=False,
            broker_order_id=broker_order_id,
            status=BrokerOrderStatus.UNKNOWN,
            message="NOT_FOUND",
        )

    def reconcile(self, broker_order_ids):
        return tuple(
            self.get_order(order_id)
            for order_id in broker_order_ids
            if self.get_order(order_id).status is not BrokerOrderStatus.UNKNOWN
        )

    def reconcile_by_idempotency_keys(self, idempotency_keys):
        return tuple(
            BrokerRecoveryResult(
                idempotency_key=key,
                result=self.orders[key],
            )
            for key in idempotency_keys
            if key in self.orders
        )


class ConflictingRecoveryBroker(FailingBroker):
    def reconcile_by_idempotency_keys(self, idempotency_keys):
        return (
            BrokerRecoveryResult(
                idempotency_key="not-requested",
                result=BrokerOrderResult(
                    accepted=True,
                    broker_order_id="broker-conflict",
                    status=BrokerOrderStatus.ACCEPTED,
                    message="CONFLICT",
                ),
            ),
        )


def req():
    return BrokerOrderRequest(
        idempotency_key="intent-1",
        symbol="TEST",
        side=Side.BUY,
        quantity=1,
        price=100,
    )


def coordinator(tmp_path, broker):
    database = tmp_path / "execution.sqlite3"
    store = ExecutionIntentStore(database)
    ledger = ImmutableLedger(database)
    return ExecutionCoordinator(store, broker, ledger), store, ledger


def test_coordinator_rejects_split_execution_and_ledger_databases(tmp_path):
    store = ExecutionIntentStore(tmp_path / "intents.sqlite3")
    ledger = ImmutableLedger(tmp_path / "ledger.sqlite3")

    with pytest.raises(ValueError, match="EXECUTION_AND_LEDGER_MUST_SHARE_DATABASE"):
        ExecutionCoordinator(store, DisabledBroker(), ledger)


def test_coordinator_persists_intent_before_submission(tmp_path):
    coordinator_instance, store, ledger = coordinator(tmp_path, DisabledBroker())

    coordinator_instance.prepare(req())

    assert store.get("intent-1").request == req()
    events = ledger.read()
    assert [event.event_type for event in events] == ["EXECUTION_INTENT_CREATED"]


def test_coordinator_marks_provider_failure_unknown_and_audits(tmp_path):
    coordinator_instance, store, ledger = coordinator(tmp_path, FailingBroker())

    attempt = coordinator_instance.submit(req())

    assert attempt.status is IntentStatus.UNKNOWN
    assert attempt.result.status is BrokerOrderStatus.UNKNOWN
    assert attempt.result.message == "BROKER_SUBMISSION_UNKNOWN"
    intent = store.get("intent-1")
    assert intent.request == req()
    assert intent.status is BrokerOrderStatus.UNKNOWN
    assert intent.result == attempt.result
    assert [event.event_type for event in ledger.read()] == [
        "EXECUTION_INTENT_CREATED",
        "EXECUTION_UNKNOWN",
    ]


def test_coordinator_audits_accepted_submission(tmp_path):
    broker = AcceptedBroker()
    coordinator_instance, store, ledger = coordinator(tmp_path, broker)

    attempt = coordinator_instance.submit(req())

    assert attempt.status is IntentStatus.SUBMITTED
    assert attempt.result.status is BrokerOrderStatus.ACCEPTED
    assert broker.submissions == 1
    intent = store.get("intent-1")
    assert intent.status is BrokerOrderStatus.ACCEPTED
    assert intent.result == attempt.result
    events = ledger.read()
    assert [event.event_type for event in events] == [
        "EXECUTION_INTENT_CREATED",
        "EXECUTION_RESULT",
    ]
    assert events[1].payload["broker_order_id"] == "broker-1"


def test_coordinator_does_not_resubmit_existing_external_result(tmp_path):
    broker = AcceptedBroker()
    coordinator_instance, _, ledger = coordinator(tmp_path, broker)

    first = coordinator_instance.submit(req())
    second = coordinator_instance.submit(req())

    assert first.result == second.result
    assert first.status is IntentStatus.SUBMITTED
    assert second.status is IntentStatus.SUBMITTED
    assert broker.submissions == 1
    assert [event.event_type for event in ledger.read()] == [
        "EXECUTION_INTENT_CREATED",
        "EXECUTION_RESULT",
    ]


def test_coordinator_does_not_resubmit_unknown_result(tmp_path):
    broker = FailingBroker()
    coordinator_instance, _, ledger = coordinator(tmp_path, broker)

    first = coordinator_instance.submit(req())
    second = coordinator_instance.submit(req())

    assert first.result == second.result
    assert first.status is IntentStatus.UNKNOWN
    assert second.status is IntentStatus.UNKNOWN
    assert [event.event_type for event in ledger.read()] == [
        "EXECUTION_INTENT_CREATED",
        "EXECUTION_UNKNOWN",
    ]


def test_coordinator_reconciliation_closes_unknown_or_open_state(tmp_path):
    coordinator_instance, store, ledger = coordinator(tmp_path, AcceptedBroker())

    attempt = coordinator_instance.submit(req())
    assert attempt.status is IntentStatus.SUBMITTED

    reconciled = coordinator_instance.reconcile(["broker-1"])

    assert len(reconciled) == 1
    assert reconciled[0].idempotency_key == "intent-1"
    assert reconciled[0].result.status is BrokerOrderStatus.FILLED
    intent = store.get("intent-1")
    assert intent.status is BrokerOrderStatus.FILLED
    assert intent.result.status is BrokerOrderStatus.FILLED
    assert [event.event_type for event in ledger.read()] == [
        "EXECUTION_INTENT_CREATED",
        "EXECUTION_RESULT",
        "EXECUTION_RECONCILED",
    ]


def test_reconcile_unrequested_provider_order_durably_halts(tmp_path):
    class UnexpectedBroker(AcceptedBroker):
        def reconcile(self, broker_order_ids):
            return (
                BrokerOrderResult(
                    accepted=True,
                    broker_order_id="provider-unrequested",
                    status=BrokerOrderStatus.FILLED,
                    message="UNREQUESTED",
                ),
            )

    coordinator_instance, store, ledger = coordinator(tmp_path, UnexpectedBroker())
    coordinator_instance.prepare(req())
    store.update_result(
        "intent-1",
        BrokerOrderResult(
            accepted=True,
            broker_order_id="broker-1",
            status=BrokerOrderStatus.ACCEPTED,
            message="ACCEPTED",
        ),
    )

    with pytest.raises(
        RuntimeError, match="EXECUTION_RECONCILE_UNREQUESTED_ORDER"
    ):
        coordinator_instance.reconcile(["broker-1"])

    assert store.is_halted()
    assert any(
        event.event_type == "EXECUTION_HALT_ASSERTED"
        and "EXECUTION_RECONCILE_UNREQUESTED_ORDER:provider-unrequested"
        in event.payload["reason"]
        for event in ledger.read()
    )


def test_reconcile_duplicate_provider_result_durably_halts(tmp_path):
    class DuplicateBroker(AcceptedBroker):
        def reconcile(self, broker_order_ids):
            result = self.get_order("broker-1")
            return (result, result)

    coordinator_instance, store, ledger = coordinator(tmp_path, DuplicateBroker())
    coordinator_instance.submit(req())

    with pytest.raises(
        RuntimeError, match="EXECUTION_RECONCILE_DUPLICATE_RESULT"
    ):
        coordinator_instance.reconcile(["broker-1"])

    assert store.is_halted()
    assert any(
        event.event_type == "EXECUTION_HALT_ASSERTED"
        and "EXECUTION_RECONCILE_DUPLICATE_RESULT:broker-1"
        in event.payload["reason"]
        for event in ledger.read()
    )


def test_reconcile_duplicate_requested_order_id_durably_halts(tmp_path):
    coordinator_instance, store, ledger = coordinator(tmp_path, AcceptedBroker())

    with pytest.raises(
        RuntimeError, match="EXECUTION_RECONCILE_DUPLICATE_REQUESTED_ORDER_ID"
    ):
        coordinator_instance.reconcile(["broker-1", "broker-1"])

    assert store.is_halted()
    assert any(
        event.event_type == "EXECUTION_HALT_ASSERTED"
        and "EXECUTION_RECONCILE_DUPLICATE_REQUESTED_ORDER_ID"
        in event.payload["reason"]
        for event in ledger.read()
    )


def test_reconcile_unbound_requested_provider_order_durably_halts(tmp_path):
    class UnknownBindingBroker(AcceptedBroker):
        def reconcile(self, broker_order_ids):
            return (
                BrokerOrderResult(
                    accepted=True,
                    broker_order_id="broker-unbound",
                    status=BrokerOrderStatus.FILLED,
                    message="UNBOUND",
                ),
            )

    coordinator_instance, store, ledger = coordinator(tmp_path, UnknownBindingBroker())
    coordinator_instance.prepare(req())
    store.update_result(
        "intent-1",
        BrokerOrderResult(
            accepted=True,
            broker_order_id="broker-1",
            status=BrokerOrderStatus.ACCEPTED,
            message="ACCEPTED",
        ),
    )

    with pytest.raises(RuntimeError, match="EXECUTION_RECONCILE_UNREQUESTED_ORDER"):
        coordinator_instance.reconcile(["broker-unbound"])

    assert store.is_halted()
    assert any(event.event_type == "EXECUTION_HALT_ASSERTED" for event in ledger.read())


def test_coordinator_does_not_duplicate_intent_audit_on_retry(tmp_path):
    coordinator_instance, store, ledger = coordinator(tmp_path, DisabledBroker())

    coordinator_instance.prepare(req())
    coordinator_instance.prepare(req())

    events = ledger.read()
    assert [event.event_type for event in events] == ["EXECUTION_INTENT_CREATED"]
    assert store.pending_keys() == ("intent-1",)


def test_coordinator_prepare_rolls_back_intent_when_audit_fails(tmp_path, monkeypatch):
    coordinator_instance, store, ledger = coordinator(tmp_path, DisabledBroker())

    def fail_audit(*args, **kwargs):
        raise RuntimeError("AUDIT_WRITE_FAILED")

    monkeypatch.setattr(ledger, "_append_in_connection", fail_audit)

    with pytest.raises(RuntimeError, match="AUDIT_WRITE_FAILED"):
        coordinator_instance.prepare(req())

    assert store.get("intent-1") is None
    assert ledger.read() == []


def test_coordinator_result_and_audit_commit_atomically(tmp_path):
    broker = AcceptedBroker()
    coordinator_instance, store, ledger = coordinator(tmp_path, broker)
    coordinator_instance.prepare(req())

    original = ledger._append_in_connection

    def fail_audit(*args, **kwargs):
        raise RuntimeError("AUDIT_WRITE_FAILED")

    import types

    ledger._append_in_connection = types.MethodType(
        lambda self, *args, **kwargs: fail_audit(*args, **kwargs),
        ledger,
    )

    result = broker.submit(req())

    with pytest.raises(RuntimeError, match="AUDIT_WRITE_FAILED"):
        coordinator_instance._commit_result(
            req().idempotency_key, result, "EXECUTION_RESULT"
        )

    intent = store.get("intent-1")
    assert intent is not None
    assert intent.result is None
    events = ledger.read()
    assert [event.event_type for event in events] == ["EXECUTION_INTENT_CREATED"]

    ledger._append_in_connection = original


def test_concurrent_prepare_same_key_creates_one_intent_and_one_audit(tmp_path):
    coordinator_instance, store, ledger = coordinator(tmp_path, DisabledBroker())

    def prepare():
        coordinator_instance.prepare(req())

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: prepare(), range(8)))

    assert store.get("intent-1").request == req()
    events = ledger.read()
    assert [event.event_type for event in events] == ["EXECUTION_INTENT_CREATED"]


def test_provider_accepted_order_is_recovered_by_idempotency_key_after_unknown(tmp_path):
    broker = CrashAfterAcceptanceBroker()
    coordinator_instance, store, ledger = coordinator(tmp_path, broker)

    attempt = coordinator_instance.submit(req())

    assert attempt.status is IntentStatus.UNKNOWN
    assert attempt.result.broker_order_id is None
    assert store.unknown_keys() == ("intent-1",)

    recovered = coordinator_instance.recover_unknown()

    assert len(recovered) == 1
    assert recovered[0].idempotency_key == "intent-1"
    assert recovered[0].result.broker_order_id == "broker-crash-1"
    assert recovered[0].result.status is BrokerOrderStatus.ACCEPTED
    assert store.get("intent-1").result.broker_order_id == "broker-crash-1"
    assert [event.event_type for event in ledger.read()] == [
        "EXECUTION_INTENT_CREATED",
        "EXECUTION_UNKNOWN",
        "EXECUTION_RECOVERED",
    ]
    assert broker.submissions == 1

    coordinator_instance.submit(req())
    assert broker.submissions == 1


def test_unknown_without_provider_match_remains_unknown_and_is_not_resubmitted(tmp_path):
    broker = FailingBroker()
    coordinator_instance, store, ledger = coordinator(tmp_path, broker)

    coordinator_instance.submit(req())
    recovered = coordinator_instance.recover_unknown()

    assert recovered == ()
    assert store.get("intent-1").status is BrokerOrderStatus.UNKNOWN
    assert store.get("intent-1").result.status is BrokerOrderStatus.UNKNOWN
    assert [event.event_type for event in ledger.read()] == [
        "EXECUTION_INTENT_CREATED",
        "EXECUTION_UNKNOWN",
    ]


def test_provider_recovery_rejects_unrequested_idempotency_key(tmp_path):
    coordinator_instance, _, _ = coordinator(tmp_path, ConflictingRecoveryBroker())

    coordinator_instance.prepare(req())

    with pytest.raises(ValueError, match="EXECUTION_RECOVERY_KEY_CONFLICT"):
        coordinator_instance.recover_unknown()


def test_terminal_execution_state_cannot_regress(tmp_path):
    broker = AcceptedBroker()
    coordinator_instance, store, _ = coordinator(tmp_path, broker)
    coordinator_instance.submit(req())
    filled = broker.get_order("broker-1")
    coordinator_instance._commit_result("intent-1", filled, "EXECUTION_RECONCILED")

    with pytest.raises(ValueError, match="EXECUTION_TERMINAL_STATE_CONFLICT"):
        store.update_result(
            "intent-1",
            BrokerOrderResult(
                accepted=True,
                broker_order_id="broker-1",
                status=BrokerOrderStatus.ACCEPTED,
                message="REGRESSION",
            ),
        )


def test_coordinator_blocks_submission_when_durable_halt_is_active(tmp_path):
    broker = AcceptedBroker()
    coordinator_instance, store, ledger = coordinator(tmp_path, broker)
    store.halt("EXECUTION_AUDIT_ORPHAN:bad-key")

    with pytest.raises(
        RuntimeError,
        match="EXECUTION_HALTED:EXECUTION_AUDIT_ORPHAN:bad-key",
    ):
        coordinator_instance.submit(req())

    assert broker.submissions == 0
    assert store.get("intent-1") is None
    assert [event.event_type for event in ledger.read()] == []


def test_durable_halt_can_be_cleared_explicitly(tmp_path):
    _, store, _ = coordinator(tmp_path, DisabledBroker())
    store.halt("TEST_HALT")
    assert store.is_halted()
    assert store.halt_reason() == "TEST_HALT"

    store.clear_halt()

    assert not store.is_halted()
    assert store.halt_reason() is None


def test_coordinator_refuses_submission_when_reconciliation_is_unhealthy(tmp_path):
    broker = AcceptedBroker()
    coordinator_instance, store, ledger = coordinator(tmp_path, broker)
    coordinator_instance.prepare(req())
    store.update_result(
        "intent-1",
        BrokerOrderResult(
            accepted=True,
            broker_order_id="broker-1",
            status=BrokerOrderStatus.ACCEPTED,
            message="ACCEPTED",
        ),
    )

    with pytest.raises(RuntimeError, match="EXECUTION_HALTED"):
        coordinator_instance.submit(
            BrokerOrderRequest(
                idempotency_key="intent-2",
                symbol="TEST",
                side=Side.BUY,
                quantity=1,
                price=100,
            )
        )

    assert broker.submissions == 0
    assert store.is_halted()
    assert any(event.event_type == "EXECUTION_HALT_ASSERTED" for event in ledger.read())
