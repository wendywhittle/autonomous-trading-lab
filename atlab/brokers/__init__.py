"""Broker adapters for real providers (paper-first).

Every adapter here implements the provider-independent ``BrokerAdapter``
protocol from :mod:`atlab.broker`. The default broker everywhere remains
``DisabledBroker``; nothing in this package can move real money without an
explicit, loud opt-in at the call site.
"""

from .alpaca import (
    ALPACA_LIVE_BASE_URL,
    ALPACA_PAPER_BASE_URL,
    AlpacaBroker,
    alpaca_idempotency_key,
)

__all__ = [
    "ALPACA_LIVE_BASE_URL",
    "ALPACA_PAPER_BASE_URL",
    "AlpacaBroker",
    "alpaca_idempotency_key",
]
