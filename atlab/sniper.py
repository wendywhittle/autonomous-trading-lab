"""Sniper strategy v1: hunt small, rapid mean-reversion edges.

An honest scope note: this is not cross-venue arbitrage in the HFT sense —
no retail system can snipe those spreads; they vanish in microseconds. This
hunts the small, frequent version of the same idea: sharp short-horizon dips
that snap back, one charter-sized ($10) scale at a time.

The strategy is deliberately stateless: it proposes ENTER/EXIT from the
latest bar only. Position sizing, loss halts, and drawdown stops belong to
the deterministic risk engine, which disposes. A losing streak therefore
cannot override the charter — the cage binds no matter what the strategy
proposes.
"""

from __future__ import annotations

import hashlib
import json

from .models import DecisionAction, JEVDecision, MarketState, Side, StrategyVersion

SNIPER_STRATEGY_ID = "sniper-mean-reversion"
SNIPER_VERSION = "1.0.0"
SNIPER_HYPOTHESIS = (
    "Sharp short-horizon dips in liquid instruments snap back within a few "
    "bars often enough that buying dips and selling bounces is positive "
    "expectancy at small size; the risk charter bounds the downside."
)


def make_sniper_decision(strategy: StrategyVersion, state: MarketState) -> JEVDecision:
    """Propose a sniper decision from the latest bar's return.

    Parameters (all in ``strategy.parameters``):
      - ``entry_dip``: ENTER/BUY when latest return <= -entry_dip.
      - ``exit_bounce``: EXIT/SELL when latest return >= exit_bounce.
      - otherwise HOLD.
    """
    if strategy.strategy_id != SNIPER_STRATEGY_ID:
        raise ValueError("SNIPER_STRATEGY_ID_MISMATCH")
    entry_dip = float(strategy.parameters.get("entry_dip", 0.004))
    exit_bounce = float(strategy.parameters.get("exit_bounce", 0.003))
    if entry_dip <= 0 or exit_bounce <= 0:
        raise ValueError("SNIPER_INVALID_PARAMETERS")

    latest_return = state.returns[-1] if state.returns else 0.0
    if latest_return <= -entry_dip:
        action, side = DecisionAction.ENTER, Side.BUY
        rationale = (
            f"Sniper dip: latest return {latest_return:.4%} <= -{entry_dip:.2%}; "
            "buying the dip for the snap-back."
        )
    elif latest_return >= exit_bounce:
        action, side = DecisionAction.EXIT, Side.SELL
        rationale = (
            f"Sniper bounce: latest return {latest_return:.4%} >= {exit_bounce:.2%}; "
            "selling into strength."
        )
    else:
        action, side = DecisionAction.HOLD, None
        rationale = "No sniper trigger: latest return inside the dead zone."

    confidence = min(1.0, 0.5 + abs(latest_return) * 50.0)
    identity = {
        "strategy_id": strategy.strategy_id,
        "strategy_version": strategy.version,
        "symbol": state.symbol,
        "action": action.value,
        "side": side.value if side else None,
        "confidence": confidence,
        "state_fingerprint": state.state_fingerprint,
        "entry_dip": entry_dip,
        "exit_bounce": exit_bounce,
    }
    decision_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return JEVDecision(
        decision_id=decision_id,
        strategy_id=strategy.strategy_id,
        strategy_version=strategy.version,
        symbol=state.symbol,
        action=action,
        side=side,
        confidence=confidence,
        rationale=rationale,
        state_fingerprint=state.state_fingerprint,
    )
