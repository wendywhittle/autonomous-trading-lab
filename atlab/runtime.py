from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .adapters import MarketDataAdapter
from .ledger import ImmutableLedger
from .models import JEVDecision, PaperOrder, StrategyVersion
from .paper import PaperExecution
from .portfolio import PaperPortfolio
from .risk import DeterministicRiskEngine


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
    """Sequential autonomous paper loop with no live execution path."""

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
            price = state.price
            current_position_notional = self.portfolio.position_quantity * price
            risk_result = self.risk.evaluate(
                decision,
                price,
                self.quantity,
                current_position_notional,
            )
            order = None

            self.ledger.append(
                "DECISION",
                decision.decision_id,
                decision.model_dump(mode="json"),
            )

            if risk_result.approved:
                order = self.execution.submit(decision, self.quantity, price)
                snapshot = self.portfolio.apply(order)
                self.ledger.append(
                    "ORDER",
                    order.order_id,
                    order.model_dump(mode="json"),
                )
                equity = snapshot.equity
            else:
                equity = self.portfolio.snapshot(price).equity
                self.ledger.append(
                    "RISK_BLOCK",
                    f"risk-{decision.decision_id}",
                    {"decision_id": decision.decision_id, "reason": risk_result.reason},
                )

            results.append(
                CycleResult(
                    decision=decision,
                    order=order,
                    risk_reason=risk_result.reason,
                    equity=equity,
                )
            )

        return tuple(results)
