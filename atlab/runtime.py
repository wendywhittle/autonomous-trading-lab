from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .adapters import MarketDataAdapter
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
            decision = make_decision(self.strategy, state)
            if self.ledger.contains_event_id(decision.decision_id):
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

            self.ledger.append(
                "DECISION",
                decision.decision_id,
                decision.model_dump(mode="json"),
            )

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
                self.ledger.append(
                    "ORDER",
                    order.order_id,
                    order.model_dump(mode="json"),
                )
                equity = snapshot.equity
            else:
                equity = pre_trade_snapshot.equity
                self.ledger.append(
                    "RISK_BLOCK",
                    f"risk-{decision.decision_id}",
                    {"decision_id": decision.decision_id, "reason": risk_result.reason},
                )

            self.risk_session = RiskSessionState(
                session_start_equity=self.risk_session.session_start_equity,
                high_water_mark=max(self.risk_session.high_water_mark, equity),
                current_equity=equity,
            )
            if self.portfolio_state_path:
                self.portfolio.save_state(self.portfolio_state_path)
            if self.risk_state_path:
                self.risk_session.save(self.risk_state_path)

            results.append(
                CycleResult(
                    decision=decision,
                    order=order,
                    risk_reason=risk_result.reason,
                    equity=equity,
                )
            )

        return tuple(results)
