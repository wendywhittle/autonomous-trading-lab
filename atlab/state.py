from __future__ import annotations

import hashlib
import json
from datetime import datetime

from .models import MarketObservation, MarketState


def build_state(observations: list[MarketObservation], *, as_of: datetime | None = None) -> MarketState:
    if not observations:
        raise ValueError("NO_MARKET_DATA")
    symbols = {item.symbol for item in observations}
    if len(symbols) != 1:
        raise ValueError("MULTIPLE_SYMBOLS")
    ordered = sorted(observations, key=lambda item: item.timestamp)
    cutoff = as_of or ordered[-1].timestamp
    available = [item for item in ordered if item.timestamp <= cutoff]
    if not available:
        raise ValueError("NO_DATA_AT_CUTOFF")
    prices = [item.price for item in available]
    returns = [(prices[i] / prices[i - 1]) - 1 for i in range(1, len(prices))]
    payload = [(item.symbol, item.timestamp.isoformat(), item.price, item.volume, item.source) for item in available]
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return MarketState(symbol=available[-1].symbol, as_of=available[-1].timestamp, price=available[-1].price, returns=returns, state_fingerprint=fingerprint)
