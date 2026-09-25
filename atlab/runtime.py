from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .adapters import MarketDataAdapter
from .jev import JEVAdapter
from .ledger import ImmutableLedger
from .models import DecisionAction, JEVDecision, PaperOrder, Side, StrategyVersion
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
        quantity_policy: Callable[[float], float] | None = None,
        stop_loss: float | None = None,
        stop_cooldown_bars: int = 5,
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
        if stop_loss is not None and stop_loss <= 0:
            raise ValueError("INVALID_STOP_LOSS")
        if stop_cooldown_bars < 0:
            raise ValueError("INVALID_STOP_COOLDOWN")
        self.adapter = adapter
        self.strategy = strategy
        self.risk = risk
        self.portfolio = portfolio
        self.ledger = ledger
        self.quantity = quantity
        # Optional price -> quantity policy (e.g. the risk charter's $10
        # target-notional sizing). When set it wins over the fixed quantity,
        # so every order notional tracks the target regardless of price.
        self.quantity_policy = quantity_policy
        # Optional per-position stop-loss (USD of unrealized loss). When set,
        # a position bleeding past this level is force-exited: the strategy
        # proposal is overridden with a deterministic EXIT/SELL decision so
        # the cage — not the strategy — decides when a loser is cut.
        self.stop_loss = stop_loss
        # After a stop-loss exit, new entries are suppressed for this many
        # bars so the strategy cannot immediately re-buy the same falling
        # knife. In-memory only: a restart resets the cooldown, which is a
        # bounded behavioral deviation (one extra gated $10 entry at most),
        # never a safety-gate bypass -- the loss halts stay restart-safe.
        self.stop_cooldown_bars = stop_cooldown_bars
        self._stop_cooldown_remaining = 0
        # Latched while a stop-loss exit is in progress: once the stop
        # fires, every subsequent cycle keeps exiting until the position is
        # flat, so a shrinking remainder can never strand a stub whose total
        # unrealized sits just inside the threshold.
        self._stop_exiting = False
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
            # M6: fail closed if the ledger database was replaced after this
            # portfolio state was written.
            self.portfolio.load_state(
                self.portfolio_state_path,
                expected_ledger_identity=self.ledger.identity,
            )
        elif self.portfolio_state_path:
            self.portfolio.save_state(
                self.portfolio_state_path, ledger_identity=self.ledger.identity
            )

    def _persist_state(self) -> None:
        if self.portfolio_state_path:
            self.portfolio.save_state(
                self.portfolio_state_path, ledger_identity=self.ledger.identity
            )
        if self.risk_state_path and self.risk_session is not None:
            self.risk_session.save(self.risk_state_path)

    def _quantity_for(self, price: float) -> float:
        """Resolve the order quantity for a fill price.

        A quantity policy (e.g. charter $10 target-notional sizing) wins over
        the fixed quantity. The resolved quantity must always be positive;
        the risk engine still gates the resulting notional.
        """
        quantity = self.quantity_policy(price) if self.quantity_policy else self.quantity
        if quantity <= 0:
            raise ValueError("INVALID_QUANTITY")
        return quantity

    def _stop_loss_decision(self, state) -> JEVDecision | None:
        """Force an EXIT when the open position's unrealized loss hits the stop.

        Returns None when no stop is configured, no position is open, or the
        unrealized loss is within tolerance. The synthetic decision is
        deterministic in the state fingerprint (plus a stop-loss marker), so
        crash recovery treats it exactly like any other decision.
        """
        if self.stop_loss is None:
            return None
        position_quantity = self.portfolio.position_quantity
        if position_quantity <= 0:
            return None
        unrealized = (state.price - self.portfolio.average_cost) * position_quantity
        if not self._stop_exiting and unrealized > -self.stop_loss:
            return None
        identity = {
            "stop_loss": True,
            "strategy_id": self.strategy.strategy_id,
            "strategy_version": self.strategy.version,
            "symbol": state.symbol,
            "state_fingerprint": state.state_fingerprint,
            "unrealized": round(unrealized, 6),
        }
        decision_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return JEVDecision(
            decision_id=decision_id,
            strategy_id=self.strategy.strategy_id,
            strategy_version=self.strategy.version,
            symbol=state.symbol,
            action=DecisionAction.EXIT,
            side=Side.SELL,
            confidence=1.0,
            rationale=(
                f"Stop-loss: unrealized {unrealized:.2f} <= -{self.stop_loss:.2f}; "
                "cutting the loser regardless of the strategy signal."
            ),
            state_fingerprint=state.state_fingerprint,
        )

    def _stop_cooldown_decision(self, state) -> JEVDecision:
        # Synthetic HOLD suppressing entries during the post-stop cooldown.
        # Deterministic in the state fingerprint so crash recovery treats it
        # like any other decision.
        decision_id = hashlib.sha256(
            json.dumps(
                {
                    "stop_cooldown": True,
                    "strategy_id": self.strategy.strategy_id,
                    "strategy_version": self.strategy.version,
                    "symbol": state.symbol,
                    "state_fingerprint": state.state_fingerprint,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return JEVDecision(
            decision_id=decision_id,
            strategy_id=self.strategy.strategy_id,
            strategy_version=self.strategy.version,
            symbol=state.symbol,
            action=DecisionAction.HOLD,
            side=None,
            confidence=1.0,
            rationale=(
                "Stop-loss cooldown: no new entries for "
                f"{self.stop_cooldown_bars} bars after being stopped out."
            ),
            state_fingerprint=state.state_fingerprint,
        )

    def run(self, symbol: str) -> tuple[CycleResult, ...]:
        from .state import build_state
        from .strategy import make_decision

        observations = sorted(
            self.adapter.observations(symbol),
            key=lambda item: item.timestamp,
        )
        results: list[CycleResult] = []

        for index in range(1, len(observations)):
            # H1/H7: the kill switch is global. Any KILL_SWITCH event in the
            # ledger halts the loop at the next cycle boundary; halting no
            # longer depends on kill events being scoped to one decision.
            if self.ledger.has_event_type("KILL_SWITCH"):
                raise RuntimeError("KILL_SWITCH_ACTIVE")
            state = build_state(
                observations[: index + 1],
                as_of=observations[index].timestamp,
            )
            if self._stop_cooldown_remaining > 0:
                self._stop_cooldown_remaining -= 1
            proposal = make_decision(self.strategy, state)
            # The stop-loss overrides the strategy: cutting losers is the
            # cage's job, and it must work even when the strategy says HOLD.
            stop_loss_decision = self._stop_loss_decision(state)
            if stop_loss_decision is not None:
                proposal = stop_loss_decision
            elif (
                self._stop_cooldown_remaining > 0
                and proposal.action is DecisionAction.ENTER
            ):
                # Cooling down after a stop: sit out instead of re-buying.
                proposal = self._stop_cooldown_decision(state)
            proposal_events = self.ledger.events_for_proposal(proposal.decision_id)
            evaluation_events = [
                event
                for event in proposal_events
                if event.event_type == "JEV_EVALUATION"
            ]
            decision_from_jev = False
            if evaluation_events:
                decision_from_jev = True
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
                decision_from_jev = True
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
            if decision_from_jev and decision_events:
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
            quantity = self._quantity_for(price)
            if stop_loss_decision is not None:
                # A stop-loss exit must always be able to flatten the stub:
                # never let the $10 sizing strand a smaller remainder.
                quantity = min(quantity, self.portfolio.position_quantity)
            persisted_evaluation = self._risk_evaluation_for(decision)
            if persisted_evaluation is None:
                # H6: persist the risk outcome BEFORE any state mutation, so a
                # crash later in the cycle resumes from the recorded outcome
                # instead of re-evaluating against already-mutated state.
                fresh_result = self.risk.evaluate(
                    decision,
                    price,
                    quantity,
                    current_position_notional,
                    equity=pre_trade_snapshot.equity,
                    session_start_equity=self.risk_session.session_start_equity,
                    high_water_mark=self.risk_session.high_water_mark,
                    # M3: the cash check was dead because available_cash was
                    # never supplied; a BUY larger than cash is now blocked
                    # with INSUFFICIENT_CASH before it can drive cash
                    # negative.
                    available_cash=pre_trade_snapshot.cash,
                )
                self._append_risk_evaluation(
                    decision,
                    approved=fresh_result.approved,
                    reason=fresh_result.reason,
                )
                risk_approved, risk_reason = (
                    fresh_result.approved,
                    fresh_result.reason,
                )
            else:
                risk_approved, risk_reason = persisted_evaluation
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

            if risk_approved:
                order_id = f"paper-{decision.decision_id}"
                if self.portfolio.has_applied_order(order_id):
                    # H6: the fill was applied and portfolio state was
                    # persisted, but the crash happened before the ORDER event
                    # was appended. Backfill the ORDER event from the
                    # deterministic order instead of re-executing and
                    # double-applying the fill.
                    order = self._reconstruct_applied_order(decision, price)
                    self._append_order_event(order)
                    equity = self.portfolio.snapshot(price).equity
                    self._persist_state()
                else:
                    try:
                        order = self.execution.submit(decision, quantity, price)
                    except RuntimeError as exc:
                        if str(exc) == "KILL_SWITCH_ACTIVE":
                            self.ledger.append(
                                "KILL_SWITCH",
                                f"kill-{decision.decision_id}",
                                {"decision_id": decision.decision_id, "reason": str(exc)},
                            )
                        raise

                    snapshot = self.portfolio.apply(order)
                    if stop_loss_decision is not None:
                        self._stop_cooldown_remaining = self.stop_cooldown_bars
                        self._stop_exiting = True
                    if snapshot.position_quantity <= 0:
                        self._stop_exiting = False
                    equity = snapshot.equity
                    self.risk_session = RiskSessionState(
                        session_start_equity=self.risk_session.session_start_equity,
                        high_water_mark=max(self.risk_session.high_water_mark, equity),
                        current_equity=equity,
                    )
                    self._persist_state()

                    self._append_order_event(order)
            else:
                equity = pre_trade_snapshot.equity
                self.ledger.append(
                    "RISK_BLOCK",
                    f"risk-{decision.decision_id}",
                    {
                        "decision_id": decision.decision_id,
                        "reason": risk_reason,
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
                    risk_reason=risk_reason,
                    equity=equity,
                )
            )

        return tuple(results)

    # -- H6 crash-recovery helpers -------------------------------------
    def _risk_evaluation_for(self, decision) -> tuple[bool, str] | None:
        """Return the persisted (approved, reason) for a decision, if any."""
        for event in self.ledger.events_for_decision(decision.decision_id):
            if event.event_type == "RISK_EVALUATION":
                payload = event.payload
                return (bool(payload["approved"]), str(payload["reason"]))
        return None

    def _append_risk_evaluation(
        self, decision, *, approved: bool, reason: str
    ) -> None:
        try:
            self.ledger.append(
                "RISK_EVALUATION",
                f"risk-eval-{decision.decision_id}",
                {
                    "decision_id": decision.decision_id,
                    "approved": approved,
                    "reason": reason,
                },
            )
        except ValueError as exc:
            if str(exc) != "LEDGER_EVENT_ID_EXISTS":
                raise
            # Another writer recorded the outcome first; the deterministic
            # risk engine produces the same outcome for the same inputs, so
            # the caller's freshly computed outcome remains valid.

    def _append_order_event(self, order: PaperOrder) -> None:
        try:
            self.ledger.append(
                "ORDER",
                order.order_id,
                order.model_dump(mode="json"),
            )
        except ValueError as exc:
            if str(exc) != "LEDGER_EVENT_ID_EXISTS":
                raise

    def _reconstruct_applied_order(self, decision, price: float) -> PaperOrder:
        """Rebuild the exact order ``PaperExecution.submit`` produced.

        The paper fill is deterministic in (decision, quantity, price), so
        an order applied before a crash can be reconstructed bit-for-bit for
        ORDER backfill without re-executing.
        """
        quantity = self._quantity_for(price)
        return PaperOrder(
            order_id=f"paper-{decision.decision_id}",
            decision_id=decision.decision_id,
            symbol=decision.symbol,
            side=decision.side,
            quantity=quantity,
            fill_price=price,
            notional=quantity * price,
            status="FILLED",
        )
