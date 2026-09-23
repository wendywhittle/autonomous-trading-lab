from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .adapters import MarketDataAdapter
from .jev import JEVAdapter
from .ledger import ImmutableLedger
from .models import JEVDecision, PaperOrder, StrategyVersion
from .paper import PaperExecution
from .portfolio import PaperPortfolio
from .risk import DeterministicRiskEngine
from .risk_state import RiskSessionState


class TradingMode(str, Enum):
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE_DISABLED = "LIVE_DISABLED"


@dataclass(frozen=True)
class CycleResult:
    decision: JEVDecision
    order: PaperOrder | None
    risk_reason: str
    equity: float


class PaperTradingEngine:
    """Sequential autonomous paper loop with persisted portfolio and risk state."""

    @staticmethod
    def _validate_jev_decision(
        decision: JEVDecision,
        strategy: StrategyVersion,
        state,
        proposal: JEVDecision,
    ) -> None:
        expected_side = {
            "HOLD": None,
            "ENTER": "BUY",
            "EXIT": "SELL",
        }[decision.action.value]
        if decision.strategy_id != strategy.strategy_id:
            raise RuntimeError("JEV_DECISION_STRATEGY_MISMATCH")
        if decision.strategy_version != strategy.version:
            raise RuntimeError("JEV_DECISION_STRATEGY_VERSION_MISMATCH")
        if decision.symbol != state.symbol:
            raise RuntimeError("JEV_DECISION_SYMBOL_MISMATCH")
        if decision.state_fingerprint != state.state_fingerprint:
            raise RuntimeError("JEV_DECISION_STATE_MISMATCH")
        if decision.side is not None and getattr(decision.side, "value", decision.side) != expected_side:
            raise RuntimeError("JEV_DECISION_SIDE_MISMATCH")
        if decision.action.value != "HOLD" and decision.side is None:
            raise RuntimeError("JEV_DECISION_SIDE_MISSING")
        if not decision.decision_id or decision.decision_id == proposal.decision_id:
            raise RuntimeError("JEV_DECISION_ID_COLLISION")

    def __init__(
        self,
        adapter: MarketDataAdapter,
        strategy: StrategyVersion,
        risk: DeterministicRiskEngine,
        portfolio: PaperPortfolio,
        ledger: ImmutableLedger,
        *,
        quantity: float = 1.0,
        kill_switch: bool = False,
        mode: TradingMode = TradingMode.PAPER,
        portfolio_state_path: str | Path | None = None,
        risk_state_path: str | Path | None = None,
        jev: JEVAdapter | None = None,
    ):
        if mode is not TradingMode.PAPER:
            raise ValueError("PAPER_ENGINE_REQUIRES_PAPER_MODE")
        if quantity <= 0:
            raise ValueError("INVALID_QUANTITY")
        self.adapter = adapter
        self.strategy = strategy
        self.risk = risk
        self.portfolio = portfolio
        self.ledger = ledger
        self.quantity = quantity
        self.jev = jev
        self.execution = PaperExecution(kill_switch=kill_switch)
        self.portfolio_state_path = (
            Path(portfolio_state_path) if portfolio_state_path else None
        )
        self.risk_state_path = Path(risk_state_path) if risk_state_path else None
        self.risk_session = (
            RiskSessionState.load(self.risk_state_path)
            if self.risk_state_path and self.risk_state_path.exists()
            else None
        )
        if self.portfolio_state_path and self.portfolio_state_path.exists():
            self.portfolio.load_state(self.portfolio_state_path)
        elif self.portfolio_state_path:
            self.portfolio.save_state(self.portfolio_state_path)

    def _persist_state(self) -> None:
        if self.portfolio_state_path:
            self.portfolio.save_state(self.portfolio_state_path)
        if self.risk_state_path and self.risk_session is not None:
            self.risk_session.save(self.risk_state_path)

    def run(self, symbol: str) -> tuple[CycleResult, ...]:
        from .state import build_state
        from .strategy import make_decision

        observations = sorted(
            self.adapter.observations(symbol),
            key=lambda item: item.timestamp,
        )
        results: list[CycleResult] = []

        for index in range(1, len(observations)):
            state = build_state(
                observations[: index + 1],
                as_of=observations[index].timestamp,
            )
            proposal = make_decision(self.strategy, state)
            proposal_events = self.ledger.events_for_proposal(proposal.decision_id)
            evaluation_events = [
                event
                for event in proposal_events
                if event.event_type == "JEV_EVALUATION"
            ]
            if evaluation_events:
                if len(evaluation_events) != 1:
                    raise RuntimeError("JEV_EVALUATION_DUPLICATE")
                evaluation_payload = evaluation_events[0].payload
                if evaluation_payload.get("proposal_id") != proposal.decision_id:
                    raise RuntimeError("JEV_EVALUATION_PROPOSAL_MISMATCH")
                try:
                    decision = JEVDecision.model_validate(
                        evaluation_payload["decision"]
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError("JEV_EVALUATION_INVALID") from exc
                self._validate_jev_decision(
                    decision, self.strategy, state, proposal
                )
            elif self.jev:
                decision = self.jev.evaluate(self.strategy, state, proposal)
                self._validate_jev_decision(
                    decision, self.strategy, state, proposal
                )
                self.ledger.append(
                    "JEV_EVALUATION",
                    f"jev-{proposal.decision_id}",
                    {
                        "proposal_id": proposal.decision_id,
                        "decision": decision.model_dump(mode="json"),
                    },
                )
            else:
                decision = proposal
            decision_events = self.ledger.events_for_decision(decision.decision_id)
            if decision_events:
                matching_decision = any(
                    event.event_type == "DECISION"
                    and event.payload == decision.model_dump(mode="json")
                    for event in decision_events
                )
                if not matching_decision:
                    raise RuntimeError("JEV_DECISION_ID_COLLISION")
            event_types = {event.event_type for event in decision_events}

            if "ORDER" in event_types or "RISK_BLOCK" in event_types:
                continue
            if "KILL_SWITCH" in event_types:
                raise RuntimeError("KILL_SWITCH_ACTIVE")

            price = state.price
            pre_trade_snapshot = self.portfolio.snapshot(price)

            if self.risk_session is None:
                self.risk_session = RiskSessionState(
                    session_start_equity=pre_trade_snapshot.equity,
                    high_water_mark=pre_trade_snapshot.equity,
                    current_equity=pre_trade_snapshot.equity,
                )
            else:
                self.risk_session = RiskSessionState(
                    session_start_equity=self.risk_session.session_start_equity,
                    high_water_mark=max(
                        self.risk_session.high_water_mark, pre_trade_snapshot.equity
                    ),
                    current_equity=pre_trade_snapshot.equity,
                )

            current_position_notional = self.portfolio.position_quantity * price
            risk_result = self.risk.evaluate(
                decision,
                price,
                self.quantity,
                current_position_notional,
                equity=pre_trade_snapshot.equity,
                session_start_equity=self.risk_session.session_start_equity,
                high_water_mark=self.risk_session.high_water_mark,
            )
            order = None

            if "DECISION" not in event_types:
                try:
                    self.ledger.append(
                        "DECISION",
                        decision.decision_id,
                        decision.model_dump(mode="json"),
                    )
                except ValueError as exc:
                    if str(exc) != "LEDGER_EVENT_ID_EXISTS":
                        raise
                    decision_events = self.ledger.events_for_decision(
                        decision.decision_id
                    )
                    event_types = {event.event_type for event in decision_events}
                    if "ORDER" in event_types or "RISK_BLOCK" in event_types:
                        continue
                    if "KILL_SWITCH" in event_types:
                        raise RuntimeError("KILL_SWITCH_ACTIVE")

            if risk_result.approved:
                try:
                    order = self.execution.submit(decision, self.quantity, price)
                except RuntimeError as exc:
                    if str(exc) == "KILL_SWITCH_ACTIVE":
                        self.ledger.append(
                            "KILL_SWITCH",
                            f"kill-{decision.decision_id}",
                            {"decision_id": decision.decision_id, "reason": str(exc)},
                        )
                    raise

                snapshot = self.portfolio.apply(order)
                equity = snapshot.equity
                self.risk_session = RiskSessionState(
                    session_start_equity=self.risk_session.session_start_equity,
                    high_water_mark=max(self.risk_session.high_water_mark, equity),
                    current_equity=equity,
                )
                self._persist_state()

                try:
                    self.ledger.append(
                        "ORDER",
                        order.order_id,
                        order.model_dump(mode="json"),
                    )
                except ValueError as exc:
                    if str(exc) != "LEDGER_EVENT_ID_EXISTS":
                        raise
            else:
                equity = pre_trade_snapshot.equity
                self.ledger.append(
                    "RISK_BLOCK",
                    f"risk-{decision.decision_id}",
                    {
                        "decision_id": decision.decision_id,
                        "reason": risk_result.reason,
                    },
                )
                self.risk_session = RiskSessionState(
                    session_start_equity=self.risk_session.session_start_equity,
                    high_water_mark=max(self.risk_session.high_water_mark, equity),
                    current_equity=equity,
                )
                self._persist_state()

            results.append(
                CycleResult(
                    decision=decision,
                    order=order,
                    risk_reason=risk_result.reason,
                    equity=equity,
                )
            )

        return tuple(results)
