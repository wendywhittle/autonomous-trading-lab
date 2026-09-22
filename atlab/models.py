from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class DecisionAction(str, Enum):
    HOLD = "HOLD"
    ENTER = "ENTER"
    EXIT = "EXIT"


class OrderStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    FILLED = "FILLED"


class MarketObservation(BaseModel):
    symbol: str
    timestamp: datetime
    price: float = Field(gt=0)
    volume: float = Field(default=0, ge=0)
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MarketState(BaseModel):
    symbol: str
    as_of: datetime
    price: float = Field(gt=0)
    returns: list[float] = Field(default_factory=list)
    regime: str = "UNKNOWN"
    state_fingerprint: str


class StrategyVersion(BaseModel):
    strategy_id: str
    version: str
    hypothesis: str
    parameters: dict[str, float] = Field(default_factory=dict)
    immutable: bool = True


class JEVDecision(BaseModel):
    decision_id: str
    strategy_id: str
    strategy_version: str
    symbol: str
    action: DecisionAction
    side: Side | None = None
    confidence: float = Field(ge=0, le=1)
    rationale: str
    state_fingerprint: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RiskDecision(BaseModel):
    approved: bool
    reason: str
    max_notional: float = Field(ge=0)


class PaperOrder(BaseModel):
    order_id: str
    decision_id: str
    symbol: str
    side: Side
    quantity: float = Field(gt=0)
    fill_price: float = Field(gt=0)
    notional: float = Field(gt=0)
    status: OrderStatus


class LedgerEvent(BaseModel):
    sequence: int = Field(ge=0)
    event_type: str
    event_id: str
    timestamp: datetime
    payload: dict[str, Any]
