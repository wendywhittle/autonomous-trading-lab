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


def make_coordinator(tmp_path, broker=None, risk=None):
    broker = broker or CountingLiveBroker()
    snapshot = state()
    risk = risk or DeterministicRiskEngine()
    db = tmp_path / "execution.sqlite3"
    store = ExecutionIntentStore(db)
    store.initialize_live_risk_state(snapshot)
    store.initialize_live_risk_limits(risk.limits)
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


@pytest.mark.parametrize("mutation",[lambda: state(position=101),lambda: state(cash=800),lambda: state(equity=900,high_water=1000)])
def test_authoritative_risk_state_mutation_blocks_before_broker(tmp_path, mutation):
    coordinator, store, _, broker, current = make_coordinator(tmp_path)
    req=request(); evidence=risk_for(req,current); store.mutate_live_risk_state(mutation(),0)
    with pytest.raises(RuntimeError, match="RISK_STATE_CHANGED_AFTER_APPROVAL"): coordinator.submit(req,evidence)
    assert broker.submissions==0
    assert store.get(req.idempotency_key) is None


def test_live_execution_authority_blocks_risk_mutation(tmp_path):
    coordinator, store, _, broker, current = make_coordinator(tmp_path)
    req=request(); evidence=risk_for(req,current); result=coordinator.submit(req,evidence)
    assert result.result.status is BrokerOrderStatus.ACCEPTED
    assert broker.submissions==1
    with pytest.raises(RuntimeError, match="MUTATION_BLOCKED_BY_ACTIVE_AUTHORITY"):
        store.mutate_live_risk_state(state(cash=800),0)

def test_risk_limit_mutation_blocks(tmp_path):
    current=state(); coordinator,store,_,broker,_=make_coordinator(tmp_path); req=request(); evidence=risk_for(req,current)
    store.mutate_live_risk_limits(RiskLimits(max_order_notional=99),0)
    with pytest.raises(RuntimeError,match="RISK_LIMITS_FINGERPRINT_CONFLICT"): coordinator.submit(req,evidence)
    assert broker.submissions==0

def test_material_order_mutation_invalidates_risk_evidence(tmp_path):
    snapshot = state()
    coordinator, _, _, broker, _ = make_coordinator(
        tmp_path
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
        tmp_path
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
        tmp_path
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
        tmp_path
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
        tmp_path
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
        tmp_path
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


# --- H12: parametrized LIVE risk-binding mutation tests (end-to-end via
# submit, asserting the specific reason and zero broker submissions). ---

def _sell_request(original):
    return BrokerOrderRequest(
        idempotency_key=original.idempotency_key,
        symbol=original.symbol,
        side=Side.SELL,
        quantity=original.quantity,
        price=original.price,
        decision_id=original.decision_id,
    )


@pytest.mark.parametrize(
    ("mutate", "expected_reason"),
    [
        (
            lambda original: request(symbol="EVIL", decision_id=original.decision_id),
            "RISK_SYMBOL_CONFLICT",
        ),
        (
            lambda original: request(decision_id="decision-tampered"),
            "RISK_DECISION_ID_CONFLICT",
        ),
        (
            _sell_request,
            "RISK_SIDE_CONFLICT",
        ),
        (
            lambda original: request(quantity=2, decision_id=original.decision_id),
            "RISK_QUANTITY_CONFLICT",
        ),
        (
            lambda original: request(price=101, decision_id=original.decision_id),
            "RISK_PRICE_CONFLICT",
        ),
        (
            lambda original: request(decision_id=""),
            "LIVE_DECISION_ID_REQUIRED",
        ),
        (
            lambda original: request(decision_id=original.idempotency_key),
            "LIVE_DECISION_ID_REQUIRED",
        ),
        (
            lambda original: request(price=None, reference_price=None),
            "LIVE_MARKET_ORDER_REFERENCE_PRICE_REQUIRED",
        ),
    ],
)
def test_live_risk_binding_mutations_are_rejected_before_broker(
    tmp_path, mutate, expected_reason
):
    coordinator, store, _, broker, snapshot = make_coordinator(tmp_path)
    original = request()
    evidence = risk_for(original, snapshot)
    mutated = mutate(original)

    with pytest.raises(RuntimeError, match=expected_reason):
        coordinator.submit(mutated, evidence)

    assert broker.submissions == 0
    assert store.get(mutated.idempotency_key) is None


def test_live_risk_evidence_without_state_fingerprint_is_rejected(tmp_path):
    coordinator, store, _, broker, snapshot = make_coordinator(tmp_path)
    original = request()
    # No risk_state passed: the signed evidence carries no state fingerprint.
    evidence = DeterministicRiskEngine().evaluate(
        JEVDecision(
            decision_id=original.decision_id,
            strategy_id="test-strategy",
            strategy_version="1",
            symbol=original.symbol,
            action=DecisionAction.ENTER,
            side=original.side,
            confidence=1,
            rationale="test",
            state_fingerprint="state-1",
            created_at=datetime.now(UTC),
        ),
        original.price,
        original.quantity,
        snapshot.current_position_notional,
        equity=snapshot.equity,
        session_start_equity=snapshot.session_start_equity,
        high_water_mark=snapshot.high_water_mark,
        available_cash=snapshot.available_cash,
    )

    with pytest.raises(RuntimeError, match="RISK_STATE_EVIDENCE_REQUIRED"):
        coordinator.submit(original, evidence)

    assert broker.submissions == 0
    assert store.get(original.idempotency_key) is None


def test_live_hold_evidence_is_not_executable(tmp_path):
    coordinator, store, _, broker, snapshot = make_coordinator(tmp_path)
    original = request()
    # Synthetic approved+HOLD evidence: the engine never approves HOLD, so a
    # directly-signed artifact exercises the defense-in-depth branch.
    hold_jev = JEVDecision(
        decision_id=original.decision_id,
        strategy_id="test-strategy",
        strategy_version="1",
        symbol=original.symbol,
        action=DecisionAction.HOLD,
        side=original.side,
        confidence=1,
        rationale="test",
        state_fingerprint="state-1",
        created_at=datetime.now(UTC),
    )
    hold_evidence = DeterministicRiskEngine()._result(
        hold_jev,
        original.price,
        original.quantity,
        snapshot.current_position_notional,
        True,
        "APPROVED",
        2500,
        equity=snapshot.equity,
        session_start_equity=snapshot.session_start_equity,
        high_water_mark=snapshot.high_water_mark,
        available_cash=snapshot.available_cash,
        risk_state_fingerprint=snapshot.fingerprint(),
    )

    with pytest.raises(RuntimeError, match="RISK_HOLD_NOT_EXECUTABLE"):
        coordinator.submit(original, hold_evidence)

    assert broker.submissions == 0
    assert store.get(original.idempotency_key) is None


def test_live_tampered_risk_evidence_is_rejected_before_broker(tmp_path):
    coordinator, store, _, broker, snapshot = make_coordinator(tmp_path)
    original = request()
    evidence = risk_for(original, snapshot)
    tampered = evidence.model_copy(update={"quantity": 2})

    with pytest.raises(RuntimeError, match="RISK_EVIDENCE_TAMPERED"):
        coordinator.submit(original, tampered)

    assert broker.submissions == 0
    assert store.get(original.idempotency_key) is None


# --- H15: revocation during provider uncertainty is blocked. ---

class UncertainLiveBroker(CountingLiveBroker):
    """Broker that drops the connection at submit time: the order may or may
    not exist provider-side, so the outcome is genuinely unknown."""

    def submit(self, request):
        self.submissions += 1
        raise ConnectionError("PROVIDER_TIMEOUT")


def test_revocation_is_blocked_during_provider_uncertainty(tmp_path):
    coordinator, store, _, broker, snapshot = make_coordinator(
        tmp_path, broker=UncertainLiveBroker()
    )
    req = request()
    attempt = coordinator.submit(req, risk_for(req, snapshot))
    assert broker.submissions == 1

    # The broker was contacted but the outcome is unknown: the intent holds an
    # UNKNOWN-status result. Revoking now could orphan a live order.
    from atlab.coordinator import IntentStatus

    assert attempt.status is IntentStatus.UNKNOWN
    intent = store.get(req.idempotency_key)
    assert intent is not None and intent.result is not None

    with pytest.raises(
        RuntimeError,
        match="EXECUTION_AUTHORIZATION_REVOCATION_BLOCKED_BY_ACTIVE_AUTHORITY",
    ):
        coordinator.revoke_execution_authorization("operator-uncertain")

    # The authorization stays active; revocation did not go through.
    assert store.authorization_is_active(coordinator.authorization.authorization_id)


def test_revocation_succeeds_once_outcome_is_known(tmp_path):
    coordinator, store, _, _broker, snapshot = make_coordinator(tmp_path)
    req = request()
    coordinator.submit(req, risk_for(req, snapshot))
    store.update_result(
        req.idempotency_key,
        BrokerOrderResult(
            accepted=True,
            broker_order_id="broker-1",
            status=BrokerOrderStatus.FILLED,
            message="FILLED",
            filled_quantity=req.quantity,
            remaining_quantity=0,
        ),
    )

    coordinator.revoke_execution_authorization("operator-known")

    assert not store.authorization_is_active(
        coordinator.authorization.authorization_id
    )


# --- H4: atomic LIVE aggregate risk-state updates on fills. ---

class ScriptedFillBroker(CountingLiveBroker):
    """LIVE broker returning a scripted result per submission."""

    def __init__(self, results):
        super().__init__()
        self._results = list(results)

    def submit(self, request):
        self.submissions += 1
        return self._results[self.submissions - 1]


def _fill_result(filled, remaining, status=BrokerOrderStatus.FILLED):
    return BrokerOrderResult(
        accepted=True,
        broker_order_id="broker-1",
        status=status,
        message="FILL",
        filled_quantity=filled,
        remaining_quantity=remaining,
    )


def _live_risk_state(store):
    snapshot, version = store.get_live_risk_state()
    return snapshot, version


def test_live_full_buy_fill_updates_aggregate_risk_state(tmp_path):
    broker = ScriptedFillBroker([_fill_result(1, 0)])
    coordinator, store, _, _, snapshot = make_coordinator(tmp_path, broker=broker)
    _, initial_version = _live_risk_state(store)
    req = request()
    coordinator.submit(req, risk_for(req, snapshot))

    updated, version = _live_risk_state(store)
    assert updated.current_position_notional == 100
    assert updated.available_cash == 900
    assert version == initial_version + 1
    assert broker.submissions == 1


def test_live_incremental_partial_fills_apply_delta_only(tmp_path):
    broker = ScriptedFillBroker(
        [_fill_result(0.4, 0.6, BrokerOrderStatus.PARTIALLY_FILLED)]
    )
    coordinator, store, _, _, snapshot = make_coordinator(tmp_path, broker=broker)
    req = request()
    coordinator.submit(req, risk_for(req, snapshot))

    partial, _ = _live_risk_state(store)
    assert partial.current_position_notional == pytest.approx(40)
    assert partial.available_cash == pytest.approx(960)

    # Cumulative fill of 1.0: only the 0.6 delta is applied, not the full 1.0.
    coordinator._commit_result(
        req.idempotency_key, _fill_result(1.0, 0), "EXECUTION_RESULT"
    )
    final, _ = _live_risk_state(store)
    assert final.current_position_notional == pytest.approx(100)
    assert final.available_cash == pytest.approx(900)


def test_live_sell_fill_reduces_position_and_restores_cash(tmp_path):
    broker = ScriptedFillBroker([_fill_result(1, 0)])
    coordinator, store, _, _, _ = make_coordinator(tmp_path, broker=broker)
    funded = state(position=500, cash=500, equity=1000)
    store.mutate_live_risk_state(funded, 0)

    sell_jev = JEVDecision(
        decision_id="decision-sell-1",
        strategy_id="test-strategy",
        strategy_version="1",
        symbol="TEST",
        action=DecisionAction.EXIT,
        side=Side.SELL,
        confidence=1,
        rationale="test",
        state_fingerprint="state-1",
        created_at=datetime.now(UTC),
    )
    sell_risk = DeterministicRiskEngine().evaluate(
        sell_jev, 100, 1, funded.current_position_notional,
        equity=funded.equity,
        session_start_equity=funded.session_start_equity,
        high_water_mark=funded.high_water_mark,
        available_cash=funded.available_cash,
        risk_state=funded,
    )
    assert sell_risk.approved
    sell_req = BrokerOrderRequest(
        idempotency_key="risk-sell-1",
        symbol="TEST",
        side=Side.SELL,
        quantity=1,
        price=100,
        decision_id="decision-sell-1",
    )
    coordinator.submit(sell_req, sell_risk)

    updated, _ = _live_risk_state(store)
    assert updated.current_position_notional == 400
    assert updated.available_cash == 600
    assert broker.submissions == 1


def test_live_fill_rolls_back_atomically_on_risk_state_corruption(tmp_path):
    broker = ScriptedFillBroker(
        [BrokerOrderResult(accepted=True, broker_order_id="broker-1",
                           status=BrokerOrderStatus.ACCEPTED, message="ACCEPTED")]
    )
    coordinator, store, ledger, _, snapshot = make_coordinator(
        tmp_path, broker=broker
    )
    before, initial_version = _live_risk_state(store)
    req = request()
    coordinator.submit(req, risk_for(req, snapshot))

    connection = store._connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "UPDATE live_risk_state SET fingerprint = ? WHERE id = 1", ("corrupt",)
        )
        connection.execute("COMMIT")
    finally:
        connection.close()

    result_events_before = [
        event for event in ledger.read() if event.event_type == "EXECUTION_RESULT"
    ]
    assert len(result_events_before) == 1  # the ACCEPTED result from submit

    with pytest.raises(RuntimeError, match="LIVE_RISK_STATE_CORRUPT"):
        coordinator._commit_result(
            req.idempotency_key, _fill_result(1, 0), "EXECUTION_RESULT"
        )

    # Atomic rollback: the fill result was not committed, the risk-state row
    # is untouched by the failed commit, and no second ledger result event
    # was appended.
    intent = store.get(req.idempotency_key)
    assert intent.result.status is BrokerOrderStatus.ACCEPTED
    row = store._connect().execute(
        "SELECT current_position_notional, available_cash, version "
        "FROM live_risk_state WHERE id = 1"
    ).fetchone()
    assert tuple(row) == (
        before.current_position_notional,
        before.available_cash,
        initial_version,
    )
    result_events_after = [
        event for event in ledger.read() if event.event_type == "EXECUTION_RESULT"
    ]
    assert [e.event_id for e in result_events_after] == [
        e.event_id for e in result_events_before
    ]


def test_paper_fill_does_not_touch_live_risk_state(tmp_path):
    broker = ScriptedFillBroker([_fill_result(1, 0)])
    broker.mode = BrokerMode.PAPER
    db = tmp_path / "paper.sqlite3"
    store = ExecutionIntentStore(db)
    ledger = ImmutableLedger(db)
    instance = ExecutionCoordinator(store, broker, ledger)
    req = BrokerOrderRequest(
        idempotency_key="paper-1", symbol="TEST", side=Side.BUY, quantity=1, price=100
    )
    instance.submit(req)

    with pytest.raises(RuntimeError, match="LIVE_RISK_STATE_NOT_INITIALIZED"):
        store.get_live_risk_state()
