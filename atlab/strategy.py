from __future__ import annotations

import hashlib
import json

from .models import DecisionAction, JEVDecision, MarketState, Side, StrategyVersion


def make_decision(strategy: StrategyVersion, state: MarketState) -> JEVDecision:
    threshold = strategy.parameters.get("entry_return", 0.0)
    latest_return = state.returns[-1] if state.returns else 0.0
    if latest_return > threshold:
        action, side = DecisionAction.ENTER, Side.BUY
        confidence = min(1.0, 0.5 + abs(latest_return) * 10)
        rationale = "Versioned strategy threshold condition satisfied."
    elif latest_return < -threshold and threshold > 0:
        action, side = DecisionAction.EXIT, Side.SELL
        confidence = min(1.0, 0.5 + abs(latest_return) * 10)
        rationale = "Versioned strategy exit condition satisfied."
    else:
        action, side, confidence = DecisionAction.HOLD, None, 0.5
        rationale = "No strategy condition satisfied."
    identity = {"strategy_id": strategy.strategy_id, "strategy_version": strategy.version, "symbol": state.symbol, "action": action.value, "side": side.value if side else None, "confidence": confidence, "state_fingerprint": state.state_fingerprint}
    decision_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return JEVDecision(decision_id=decision_id, strategy_id=strategy.strategy_id, strategy_version=strategy.version, symbol=state.symbol, action=action, side=side, confidence=confidence, rationale=rationale, state_fingerprint=state.state_fingerprint)
