from atlab.promotion import PromotionEvidence, PromotionGate, PromotionMode


def evidence(**overrides):
    values = {
        "tests_green": True,
        "paper_run_complete": True,
        "risk_limits_active": True,
        "kill_switch_verified": True,
        "replay_deterministic": True,
        "no_live_credentials": True,
    }
    values.update(overrides)
    return PromotionEvidence(**values)


def test_promotion_gate_allows_paper_with_required_evidence():
    result = PromotionGate().evaluate(evidence(), PromotionMode.PAPER)
    assert result.eligible
    assert result.failed_requirements == ()


def test_promotion_gate_blocks_missing_evidence():
    result = PromotionGate().evaluate(
        evidence(replay_deterministic=False), PromotionMode.PAPER
    )
    assert not result.eligible
    assert result.failed_requirements == ("REPLAY_NOT_DETERMINISTIC",)


def test_live_promotion_requires_explicit_human_approval():
    result = PromotionGate().evaluate(evidence(), PromotionMode.LIVE)
    assert not result.eligible
    assert result.failed_requirements == ("HUMAN_APPROVAL_REQUIRED",)
