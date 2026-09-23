from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from .models import JEVDecision, MarketState, StrategyVersion


class JEVAdapter(Protocol):
    """Contract for the real JEV evaluation capability.

    This module does not emulate JEV. A connected implementation must supply
    the actual JEV evaluation capability and return a typed JEVDecision.
    """

    def evaluate(
        self,
        strategy: StrategyVersion,
        state: MarketState,
        proposal: JEVDecision,
    ) -> JEVDecision:
        """Evaluate a strategy proposal using the connected JEV capability."""
        ...


class UnconfiguredJEV:
    """Fail-closed adapter until a real JEV implementation is connected."""

    def evaluate(
        self,
        strategy: StrategyVersion,
        state: MarketState,
        proposal: JEVDecision,
    ) -> JEVDecision:
        raise RuntimeError("JEV_NOT_CONFIGURED")


class TypeSafeJEV:
    """Real TypeSafe/Jev adapter.

    The official TypeSafe Python SDK reads TYPESAFE_API_KEY from the environment.
    The adapter asks Jev for a typed action judgment and confidence, then maps
    that judgment back into the repository's immutable decision contract.

    Jev supplies evaluation; deterministic code retains execution authority.
    """

    def __init__(self, client: Any | None = None, *, model: str = "jev-latest"):
        self._client = client
        self.model = model

    def _evaluate_with_client(
        self,
        client: Any,
        strategy: StrategyVersion,
        state: MarketState,
        proposal: JEVDecision,
    ) -> JEVDecision:
        from typesafe_sdk import Choice

        response = client.system_one(
            state={
                "strategy": strategy.model_dump(mode="json"),
                "market_state": state.model_dump(mode="json"),
                "strategy_proposal": proposal.model_dump(mode="json"),
            },
            questions={
                "action": Choice(
                    instructions=(
                        "Evaluate the strategy proposal against the supplied market "
                        "state. Choose the action Jev judges most appropriate."
                    ),
                    criteria={
                        "HOLD": "Do not open or close a position.",
                        "ENTER": "Open the proposed position.",
                        "EXIT": "Close the proposed position.",
                    },
                )
            },
            model=self.model,
        )

        answer = response.choices["action"]
        action = str(answer.choice).upper()
        if action not in {"HOLD", "ENTER", "EXIT"}:
            raise RuntimeError("JEV_INVALID_ACTION")

        confidence = float(answer.confidence)
        if not 0 <= confidence <= 1:
            raise RuntimeError("JEV_INVALID_CONFIDENCE")

        probabilities = getattr(answer, "probabilities", {})
        rationale = (
            f"TypeSafe Jev action={action}; confidence={confidence:.6f}; "
            f"probabilities={json.dumps(probabilities, sort_keys=True, default=str)}"
        )

        decision_material = {
            "proposal_id": proposal.decision_id,
            "state_fingerprint": state.state_fingerprint,
            "action": action,
            "confidence": confidence,
            "probabilities": probabilities,
        }
        decision_id = "jev-" + hashlib.sha256(
            json.dumps(decision_material, sort_keys=True, default=str).encode()
        ).hexdigest()[:24]

        return proposal.model_copy(
            update={
                "decision_id": decision_id,
                "action": action,
                "confidence": confidence,
                "rationale": rationale,
                "created_at": proposal.created_at,
            }
        )

    def evaluate(
        self,
        strategy: StrategyVersion,
        state: MarketState,
        proposal: JEVDecision,
    ) -> JEVDecision:
        if self._client is not None:
            return self._evaluate_with_client(self._client, strategy, state, proposal)

        try:
            from typesafe_sdk import TypeSafeClient
        except ImportError as exc:
            raise RuntimeError("TYPESAFE_SDK_NOT_INSTALLED") from exc

        with TypeSafeClient(model=self.model) as client:
            return self._evaluate_with_client(client, strategy, state, proposal)
