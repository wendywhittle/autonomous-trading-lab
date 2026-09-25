"""Tests for the all-in cost model."""

import pytest

from atlab.research.costs import CostModel


def costs(**overrides):
    params = {
        "name": "test",
        "fee_flat": 1.0,
        "fee_bps": 10.0,
        "slippage_bps": 20.0,
        "spread_bps": 5.0,
    }
    params.update(overrides)
    return CostModel(**params)


def test_per_trade_cost_combines_flat_and_proportional():
    model = costs()
    # 1.00 flat + 1000 * (10 + 20 + 5) / 10000 = 1.00 + 3.50
    assert model.per_trade_cost(1000.0) == pytest.approx(4.50)


def test_total_cost_scales_with_trade_count():
    model = costs()
    assert model.total_cost(4, 1000.0) == pytest.approx(4 * 4.50)
    assert model.total_cost(0, 1000.0) == pytest.approx(0.0)


def test_fingerprint_is_stable_and_sensitive():
    assert costs().fingerprint() == costs().fingerprint()
    assert costs().fingerprint() != costs(slippage_bps=21.0).fingerprint()
    assert len(costs().fingerprint()) == 64


def test_rejects_negative_fields_and_inputs():
    with pytest.raises(ValueError, match="COST_MODEL_NAME_REQUIRED"):
        CostModel(name="")
    with pytest.raises(ValueError, match="COST_MODEL_NEGATIVE_FEE_BPS"):
        costs(fee_bps=-1.0)
    with pytest.raises(ValueError, match="COST_MODEL_NEGATIVE_NOTIONAL"):
        costs().per_trade_cost(-5.0)
    with pytest.raises(ValueError, match="COST_MODEL_NEGATIVE_TRADES"):
        costs().total_cost(-2, 100.0)
