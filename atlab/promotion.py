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
    human_approval: bool = False


@dataclass(frozen=True)
class PromotionResult:
    eligible: bool
    target: PromotionMode
    failed_requirements: tuple[str, ...]


class PromotionGate:
    """Explicit promotion gate. It never grants live authority automatically."""

    REQUIREMENTS = (
        ("tests_green", "TESTS_NOT_GREEN"),
        ("paper_run_complete", "PAPER_RUN_INCOMPLETE"),
        ("risk_limits_active", "RISK_LIMITS_INACTIVE"),
        ("kill_switch_verified", "KILL_SWITCH_NOT_VERIFIED"),
        ("replay_deterministic", "REPLAY_NOT_DETERMINISTIC"),
        ("no_live_credentials", "LIVE_CREDENTIALS_PRESENT"),
        ("human_approval", "HUMAN_APPROVAL_REQUIRED"),
    )

    def evaluate(
        self, evidence: PromotionEvidence, target: PromotionMode
    ) -> PromotionResult:
        failed = tuple(
            reason for field, reason in self.REQUIREMENTS if not getattr(evidence, field)
        )
        if target is PromotionMode.LIVE and not evidence.human_approval:
            return PromotionResult(False, target, failed)
        if target is PromotionMode.PAPER:
            paper_requirements = tuple(
                reason
                for field, reason in self.REQUIREMENTS[:-1]
                if not getattr(evidence, field)
            )
            return PromotionResult(not paper_requirements, target, paper_requirements)
        return PromotionResult(not failed, target, failed)
