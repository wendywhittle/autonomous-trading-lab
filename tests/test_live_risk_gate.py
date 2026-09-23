from datetime import UTC, datetime

import pytest

from atlab.broker import BrokerMode, BrokerOrderRequest, BrokerOrderResult, BrokerOrderStatus
from atlab.compliance import ComplianceEngine, CompliancePolicy
from atlab.coordinator import ExecutionCoordinator
from atlab.execution import ExecutionIntentStore
from atlab.ledger import ImmutableLedger
from atlab.models import DecisionAction, JEVDecision, Side
from atlab.promotion import PromotionEvidence, PromotionGate, PromotionMode
from atlab.risk import DeterministicRiskEngine


class CountingLiveBroker:
    mode = BrokerMode.LIVE

    def __init__(self):
        self.submissions = 0

    def submit(self, request):
        self.submissions += 1
        return BrokerOrderResult(
            accepted=True,
            broker_order_id=f"broker-{self.submissions}",
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
        return tuple(self.get_order(key) for key in broker_order_ids)

    def discover_open_orders(self):
        return ()

    def reconcile_by_idempotency_keys(self, idempotency_keys):
        return ()


def authorization():
    evidence = PromotionEvidence(
        tests_green=True,
        paper_run_complete=True,
        risk_limits_active=True,
        kill_switch_verified=True,
        replay_deterministic=True,
        no_live_credentials=False,
        live_execution_implemented=True,
        live_controls_verified=True,
        human_approval=True,
        live_credentials_configured=True,
    )
    return PromotionGate().authorize(evidence, PromotionMode.LIVE)


def request(symbol="TEST", quantity=1, price=100, reference_price=None, key="risk-1"):
    return BrokerOrderRequest(
        idempotency_key=key,
        symbol=symbol,
        side=Side.BUY,
        quantity=quantity,
        price=price,
        reference_price=reference_price,
    )


def risk_for(req):
    decision = JEVDecision(
        decision_id=req.decision_id,
        strategy_id="test-strategy",
        strategy_version="1",
        symbol=req.symbol,
        action=DecisionAction.ENTER,
        side=req.side,
        confidence=1,
        rationale="test",
        state_fingerprint="state-1",
        created_at=datetime.now(UTC),
    )
    return DeterministicRiskEngine().evaluate(decision, req.price or req.reference_price, req.quantity)


def make_make_coordinator(tmp_path, broker=None):
    broker = broker or CountingLiveBroker()
    db = tmp_path / "execution.sqlite3"
    store = ExecutionIntentStore(db)
    ledger = ImmutableLedger(db)
    instance = ExecutionCoordinator(
        store,
        broker,
        ledger,
        authorization=authorization(),
        compliance=ComplianceEngine(
            CompliancePolicy(policy_id="test-policy", version="1")
        ),
    )
    instance.activate_execution_authorization("test-activate")
    return instance, store, ledger, broker


def test_live_execution_without_risk_evidence_is_blocked(tmp_path):
    coordinator, store, ledger, broker = coordinator(tmp_path)
    req = request()

    with pytest.raises(RuntimeError, match="LIVE_RISK_DECISION_REQUIRED"):
        coordinator.submit(req)

    assert broker.submissions == 0
    assert store.get(req.idempotency_key) is None
    assert ledger.read() == []


def test_valid_risk_evidence_is_required_and_persisted(tmp_path):
    coordinator, store, ledger, broker = coordinator(tmp_path)
    req = request()

    result = coordinator.submit(req, risk_for(req))

    assert result.result.status is BrokerOrderStatus.ACCEPTED
    assert broker.submissions == 1
    intent = store.get(req.idempotency_key)
    assert intent is not None
    assert intent.risk_decision is not None
    assert intent.risk_fingerprint == intent.risk_decision.risk_fingerprint


@pytest.mark.parametrize(
    "mutated",
    [
        lambda r: request(symbol="EVIL"),
        lambda r: request(quantity=2),
        lambda r: request(price=101),
        lambda r: BrokerOrderRequest(
            idempotency_key=r.idempotency_key,
            symbol=r.symbol,
            side=Side.SELL,
            quantity=r.quantity,
            price=r.price,
        ),
    ],
)
def test_material_order_mutation_invalidates_risk_evidence(tmp_path, mutated):
    coordinator, _, _, broker = coordinator(tmp_path)
    original = request()
    evidence = risk_for(original)

    with pytest.raises(RuntimeError):
        coordinator.submit(mutated(original), evidence)

    assert broker.submissions == 0


def test_tampered_risk_fingerprint_is_blocked(tmp_path):
    coordinator, _, _, broker = coordinator(tmp_path)
    req = request()
    evidence = risk_for(req).model_copy(update={"risk_fingerprint": "tampered"})

    with pytest.raises(RuntimeError, match="RISK_EVIDENCE_TAMPERED"):
        coordinator.submit(req, evidence)

    assert broker.submissions == 0


def test_market_order_without_reference_price_is_blocked(tmp_path):
    coordinator, _, _, broker = coordinator(tmp_path)
    req = request(price=None)

    with pytest.raises(RuntimeError, match="LIVE_MARKET_ORDER_REFERENCE_PRICE_REQUIRED"):
        coordinator.submit(req, None)

    assert broker.submissions == 0


def test_market_order_uses_deterministic_reference_price(tmp_path):
    coordinator, store, _, broker = coordinator(tmp_path)
    req = request(price=None, reference_price=100)
    evidence = risk_for(req)

    result = coordinator.submit(req, evidence)

    assert result.result.status is BrokerOrderStatus.ACCEPTED
    assert broker.submissions == 1
    assert store.get(req.idempotency_key).risk_decision.price == 100


def test_hold_risk_evidence_cannot_execute(tmp_path):
    coordinator, _, _, broker = coordinator(tmp_path)
    req = request()
    decision = JEVDecision(
        decision_id=req.decision_id,
        strategy_id="test-strategy",
        strategy_version="1",
        symbol=req.symbol,
        action=DecisionAction.HOLD,
        side=None,
        confidence=1,
        rationale="hold",
        state_fingerprint="state-1",
        created_at=datetime.now(UTC),
    )
    evidence = DeterministicRiskEngine().evaluate(decision, 100, req.quantity)

    with pytest.raises(RuntimeError, match="RISK_DECISION_NOT_APPROVED"):
        coordinator.submit(req, evidence)

    assert broker.submissions == 0
