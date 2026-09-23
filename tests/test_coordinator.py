from atlab.broker import (
    BrokerMode,
    BrokerOrderRequest,
    BrokerOrderResult,
    BrokerOrderStatus,
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


def req():
    return BrokerOrderRequest(
        idempotency_key="intent-1",
        symbol="TEST",
        side=Side.BUY,
        quantity=1,
        price=100,
    )


def coordinator(tmp_path, broker):
    store = ExecutionIntentStore(tmp_path / "intents.json")
    ledger = ImmutableLedger(tmp_path / "ledger.sqlite3")
    return ExecutionCoordinator(store, broker, ledger), store, ledger


def test_coordinator_persists_intent_before_submission(tmp_path):
    coordinator_instance, store, ledger = coordinator(tmp_path, DisabledBroker())

    coordinator_instance.prepare(req())

    assert store.get("intent-1").request == req()
    events = ledger.read()
    assert [event.event_type for event in events] == ["EXECUTION_INTENT_CREATED"]


def test_coordinator_marks_provider_failure_unknown_and_audits(tmp_path):
    coordinator_instance, _, ledger = coordinator(tmp_path, FailingBroker())

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
    coordinator_instance, _, ledger = coordinator(tmp_path, AcceptedBroker())

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


def test_coordinator_does_not_duplicate_intent_audit_on_retry(tmp_path):
    coordinator_instance, _, ledger = coordinator(tmp_path, DisabledBroker())

    coordinator_instance.prepare(req())
    coordinator_instance.prepare(req())

    events = ledger.read()
    assert [event.event_type for event in events] == ["EXECUTION_INTENT_CREATED"]
    assert store.pending_keys() == ("intent-1",)
