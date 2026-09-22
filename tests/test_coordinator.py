from atlab.broker import BrokerMode, BrokerOrderRequest, BrokerOrderStatus, DisabledBroker
from atlab.coordinator import ExecutionCoordinator, IntentStatus
from atlab.execution import ExecutionIntentStore
from atlab.models import Side


class FailingBroker:
    mode = BrokerMode.LIVE

    def submit(self, request):
        raise TimeoutError("provider timeout")

    def cancel(self, broker_order_id):
        return False

    def get_order(self, broker_order_id):
        return None

    def reconcile(self, broker_order_ids):
        return ()


def req():
    return BrokerOrderRequest(
        idempotency_key="intent-1",
        symbol="TEST",
        side=Side.BUY,
        quantity=1,
        price=100,
    )


def test_coordinator_persists_intent_before_submission(tmp_path):
    store = ExecutionIntentStore(tmp_path / "intents.json")
    coordinator = ExecutionCoordinator(store, DisabledBroker())

    coordinator.prepare(req())

    assert store.get("intent-1").request == req()


def test_coordinator_marks_provider_failure_unknown(tmp_path):
    store = ExecutionIntentStore(tmp_path / "intents.json")
    coordinator = ExecutionCoordinator(store, FailingBroker())

    attempt = coordinator.submit(req())

    assert attempt.status is IntentStatus.UNKNOWN
    assert attempt.result.status is BrokerOrderStatus.UNKNOWN
    assert attempt.result.message == "BROKER_SUBMISSION_UNKNOWN"
    assert store.get("intent-1").request == req()
