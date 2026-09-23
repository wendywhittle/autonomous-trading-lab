from dataclasses import dataclass
from enum import Enum


class PromotionMode(str, Enum):
    RESEARCH = "RESEARCH"
    PAPER = "PAPER"
    LIVE = "LIVE"


@dataclass(frozen=True)
class PromotionEvidence:
    tests_green: bool
    paper_run_complete: bool
    risk_limits_active: bool
    kill_switch_verified: bool
    replay_deterministic: bool
    no_live_credentials: bool
    live_execution_implemented: bool = False
    live_controls_verified: bool = False
    human_approval: bool = False


@dataclass(frozen=True)
class PromotionResult:
    eligible: bool
    target: PromotionMode
    failed_requirements: tuple[str, ...]


class PromotionGate:
    """Explicit promotion gate; LIVE also requires a real, verified execution path."""

    REQUIREMENTS = (
        ("tests_green", "TESTS_NOT_GREEN"),
        ("paper_run_complete", "PAPER_RUN_INCOMPLETE"),
        ("risk_limits_active", "RISK_LIMITS_INACTIVE"),
        ("kill_switch_verified", "KILL_SWITCH_NOT_VERIFIED"),
        ("replay_deterministic", "REPLAY_NOT_DETERMINISTIC"),
        ("no_live_credentials", "LIVE_CREDENTIALS_PRESENT"),
        ("human_approval", "HUMAN_APPROVAL_REQUIRED"),
    )

    LIVE_REQUIREMENTS = (
        ("live_execution_implemented", "LIVE_EXECUTION_NOT_IMPLEMENTED"),
        ("live_controls_verified", "LIVE_CONTROLS_NOT_VERIFIED"),
    )

    def evaluate(
        self, evidence: PromotionEvidence, target: PromotionMode
    ) -> PromotionResult:
        if target is PromotionMode.PAPER:
            failed = tuple(
                reason
                for field, reason in self.REQUIREMENTS[:-1]
                if not getattr(evidence, field)
            )
            return PromotionResult(not failed, target, failed)

        failed = tuple(
            reason for field, reason in self.REQUIREMENTS if not getattr(evidence, field)
        )
        if target is PromotionMode.LIVE:
            failed += tuple(
                reason
                for field, reason in self.LIVE_REQUIREMENTS
                if not getattr(evidence, field)
            )
        return PromotionResult(not failed, target, failed)
