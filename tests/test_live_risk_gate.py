from datetime import UTC, datetime

import pytest

from atlab.broker import BrokerMode, BrokerOrderRequest, BrokerOrderResult, BrokerOrderStatus
from atlab.compliance import ComplianceEngine, CompliancePolicy
from atlab.coordinator import ExecutionCoordinator
from atlab.execution import ExecutionIntentStore
from atlab.ledger import ImmutableLedger
from atlab.models import DecisionAction, JEVDecision, Side
from atlab.promotion import PromotionEvidence, PromotionGate, PromotionMode
from atlab.risk import DeterministicRiskEngine, RiskLimits
from atlab.risk_state import RiskStateSnapshot


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


def request(
    symbol="TEST",
    quantity=1,
    price=100,
    reference_price=None,
    key="risk-1",
    decision_id="decision-1",
):
    return BrokerOrderRequest(
        idempotency_key=key,
        symbol=symbol,
        side=Side.BUY,
        quantity=quantity,
        price=price,
        reference_price=reference_price,
        decision_id=decision_id,
    )


def state(
    *,
    position=0,
    cash=1000,
    equity=1000,
    session_start=1000,
    high_water=1000,
):
    return RiskStateSnapshot(
        symbol="TEST",
        current_position_notional=position,
        equity=equity,
        session_start_equity=session_start,
        high_water_mark=high_water,
        available_cash=cash,
    )


def risk_for(req, snapshot):
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
    effective_price = req.price if req.price is not None else req.reference_price
    return DeterministicRiskEngine().evaluate(
        decision,
        effective_price,
        req.quantity,
        snapshot.current_position_notional,
        equity=snapshot.equity,
        session_start_equity=snapshot.session_start_equity,
        high_water_mark=snapshot.high_water_mark,
        available_cash=snapshot.available_cash,
        risk_state=snapshot,
    )


def make_coordinator(tmp_path, broker=None, provider=None, risk=None):
    broker = broker or CountingLiveBroker()
    snapshot = state()
    provider = provider or (lambda request: snapshot)
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
        risk=risk,
        risk_state_provider=provider,
    )
    instance.activate_execution_authorization("test-activate")
    return instance, store, ledger, broker, snapshot


def test_live_execution_without_risk_evidence_is_blocked(tmp_path):
    coordinator, store, ledger, broker, _ = make_coordinator(tmp_path)
    req = request()

    with pytest.raises(RuntimeError, match="LIVE_RISK_DECISION_REQUIRED"):
        coordinator.submit(req)

    assert broker.submissions == 0
    assert store.get(req.idempotency_key) is None
    assert [event.event_type for event in ledger.read()] == ["EXECUTION_AUTHORIZATION_ACTIVATED"]


def test_live_risk_evidence_is_state_bound_and_persisted(tmp_path):
    coordinator, store, _, broker, snapshot = make_coordinator(tmp_path)
    req = request()

    result = coordinator.submit(req, risk_for(req, snapshot))

    assert result.result.status is BrokerOrderStatus.ACCEPTED
    assert broker.submissions == 1
    intent = store.get(req.idempotency_key)
    assert intent is not None
    assert intent.risk_decision is not None
    assert intent.risk_decision.risk_state_fingerprint == snapshot.fingerprint()
    assert intent.risk_fingerprint == intent.risk_decision.risk_fingerprint


@pytest.mark.parametrize(
    "mutation",
    [
        lambda: state(position=101),
        lambda: state(cash=800),
        lambda: state(equity=900, high_water=1000),
    ],
)
def test_authoritative_risk_state_mutation_blocks_before_broker(tmp_path, mutation):
    current = state()
    coordinator, store, _, broker, _ = make_coordinator(
        tmp_path, provider=lambda request: current
    )
    req = request()
    evidence = risk_for(req, current)

    current = mutation()

    with pytest.raises(RuntimeError, match="RISK_STATE_"):
        coordinator.submit(req, evidence)

    assert broker.submissions == 0
    assert store.get(req.idempotency_key) is None


def test_final_risk_state_race_blocks_and_leaves_intent_auditable(tmp_path):
    original = state()
    mutated = state(cash=800)
    calls = 0

    def provider(request):
        nonlocal calls
        calls += 1
        return original if calls == 1 else mutated

    coordinator, store, ledger, broker, _ = make_coordinator(
        tmp_path, provider=provider
    )
    req = request()
    evidence = risk_for(req, original)

    with pytest.raises(RuntimeError, match="RISK_STATE_CHANGED_AFTER_APPROVAL"):
        coordinator.submit(req, evidence)

    assert calls >= 2
    assert broker.submissions == 0
    assert store.get(req.idempotency_key) is not None
    assert any(
        event.event_type == "EXECUTION_INTENT_CREATED"
        for event in ledger.read()
    )


def test_risk_limit_mutation_blocks(tmp_path):
    current = state()
    risk = DeterministicRiskEngine(RiskLimits(max_order_notional=2500))
    coordinator, _, _, broker, _ = make_coordinator(
        tmp_path, provider=lambda request: current, risk=risk
    )
    req = request()
    evidence = risk_for(req, current)

    risk.limits = RiskLimits(max_order_notional=99)

    with pytest.raises(RuntimeError, match="RISK_LIMITS_FINGERPRINT_CONFLICT"):
        coordinator.submit(req, evidence)

    assert broker.submissions == 0


def test_material_order_mutation_invalidates_risk_evidence(tmp_path):
    snapshot = state()
    coordinator, _, _, broker, _ = make_coordinator(
        tmp_path, provider=lambda request: snapshot
    )
    original = request()
    evidence = risk_for(original, snapshot)

    mutated = request(symbol="EVIL", decision_id=original.decision_id)

    with pytest.raises(RuntimeError, match="RISK_SYMBOL_CONFLICT"):
        coordinator.submit(mutated, evidence)

    assert broker.submissions == 0


def test_tampered_risk_fingerprint_is_blocked(tmp_path):
    snapshot = state()
    coordinator, _, _, broker, _ = make_coordinator(
        tmp_path, provider=lambda request: snapshot
    )
    req = request()
    evidence = risk_for(req, snapshot).model_copy(update={"risk_fingerprint": "tampered"})

    with pytest.raises(RuntimeError, match="RISK_EVIDENCE_TAMPERED"):
        coordinator.submit(req, evidence)

    assert broker.submissions == 0


def test_market_order_without_reference_price_is_blocked(tmp_path):
    coordinator, _, _, broker, _ = make_coordinator(tmp_path)
    req = request(price=None)

    with pytest.raises(RuntimeError, match="LIVE_RISK_DECISION_REQUIRED"):
        coordinator.submit(req)

    assert broker.submissions == 0


def test_market_order_uses_deterministic_reference_price(tmp_path):
    snapshot = state()
    coordinator, store, _, broker, _ = make_coordinator(
        tmp_path, provider=lambda request: snapshot
    )
    req = request(price=None, reference_price=100)
    evidence = risk_for(req, snapshot)

    result = coordinator.submit(req, evidence)

    assert result.result.status is BrokerOrderStatus.ACCEPTED
    assert broker.submissions == 1
    assert store.get(req.idempotency_key).risk_decision.price == 100


def test_hold_risk_evidence_cannot_execute(tmp_path):
    snapshot = state()
    coordinator, _, _, broker, _ = make_coordinator(
        tmp_path, provider=lambda request: snapshot
    )
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
    evidence = DeterministicRiskEngine().evaluate(
        decision,
        100,
        req.quantity,
        risk_state=snapshot,
        equity=snapshot.equity,
        session_start_equity=snapshot.session_start_equity,
        high_water_mark=snapshot.high_water_mark,
        available_cash=snapshot.available_cash,
    )

    with pytest.raises(RuntimeError, match="RISK_DECISION_NOT_APPROVED"):
        coordinator.submit(req, evidence)

    assert broker.submissions == 0


def test_live_missing_decision_identity_is_rejected(tmp_path):
    snapshot = state()
    coordinator, _, _, broker, _ = make_coordinator(
        tmp_path, provider=lambda request: snapshot
    )
    req = BrokerOrderRequest(
        idempotency_key="risk-1",
        symbol="TEST",
        side=Side.BUY,
        quantity=1,
        price=100,
    )
    evidence = risk_for(
        BrokerOrderRequest(
            idempotency_key="risk-1",
            symbol="TEST",
            side=Side.BUY,
            quantity=1,
            price=100,
            decision_id="decision-1",
        ),
        snapshot,
    )

    with pytest.raises(RuntimeError, match="LIVE_DECISION_ID_REQUIRED"):
        coordinator.submit(req, evidence)

    assert broker.submissions == 0


def test_live_requires_distinct_decision_identity(tmp_path):
    snapshot = state()
    coordinator, _, _, broker, _ = make_coordinator(
        tmp_path, provider=lambda request: snapshot
    )
    req = request(decision_id="risk-1")
    evidence = risk_for(req, snapshot)

    bad = BrokerOrderRequest(
        idempotency_key=req.idempotency_key,
        symbol=req.symbol,
        side=req.side,
        quantity=req.quantity,
        price=req.price,
        decision_id=req.idempotency_key,
    )

    with pytest.raises(RuntimeError, match="LIVE_DECISION_ID_REQUIRED"):
        coordinator.submit(bad, evidence)

    assert broker.submissions == 0
