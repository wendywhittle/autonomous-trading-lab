import json

import pytest

from atlab.models import OrderStatus, PaperOrder, Side
from atlab.portfolio import PaperPortfolio


def _buy(order_id: str, quantity: float, fill_price: float) -> PaperOrder:
    return PaperOrder(
        order_id=order_id,
        decision_id="decision-1",
        symbol="TEST",
        side=Side.BUY,
        quantity=quantity,
        fill_price=fill_price,
        notional=quantity * fill_price,
        status=OrderStatus.FILLED,
    )


def test_apply_buy_rejects_insufficient_cash():
    # M3: apply() itself must never let cash go negative, no matter who
    # calls it — the risk engine's INSUFFICIENT_CASH check is the first
    # line of defense, this is the backstop.
    portfolio = PaperPortfolio(100.0)
    order = _buy("order-1", 2, 100.0)  # notional 200 > cash 100

    with pytest.raises(ValueError, match="INSUFFICIENT_CASH"):
        portfolio.apply(order)

    assert portfolio.cash == 100.0
    assert not portfolio.has_applied_order("order-1")


def test_apply_buy_with_exact_cash_succeeds():
    portfolio = PaperPortfolio(200.0)
    snapshot = portfolio.apply(_buy("order-1", 2, 100.0))

    assert snapshot.cash == 0.0
    assert snapshot.equity == 200.0  # 0 cash + 2 shares @ 100


def test_load_state_rejects_ledger_identity_mismatch(tmp_path):
    # M6: the portfolio state is bound to the ledger database it was
    # written alongside; a replaced database fails closed.
    path = tmp_path / "portfolio.json"
    portfolio = PaperPortfolio(1000.0)
    portfolio.save_state(path, ledger_identity="ledger-a")

    restored = PaperPortfolio(0)
    with pytest.raises(ValueError, match="LEDGER_IDENTITY_MISMATCH"):
        restored.load_state(path, expected_ledger_identity="ledger-b")

    assert restored.cash == 0  # untouched by the failed load


def test_save_load_state_round_trip_with_identity(tmp_path):
    path = tmp_path / "portfolio.json"
    portfolio = PaperPortfolio(1000.0)
    portfolio.save_state(path, ledger_identity="ledger-a")

    restored = PaperPortfolio(0)
    restored.load_state(path, expected_ledger_identity="ledger-a")

    assert restored.cash == 1000.0
    assert restored.state("ledger-a")["ledger_identity"] == "ledger-a"


def test_load_state_without_identity_still_loads(tmp_path):
    # Files written before the identity record existed carry no identity
    # and cannot be verified — they load as before.
    path = tmp_path / "portfolio.json"
    PaperPortfolio(500.0).save_state(path)

    restored = PaperPortfolio(0)
    restored.load_state(path, expected_ledger_identity="ledger-a")

    assert restored.cash == 500.0


def test_load_state_rejects_malformed_identity(tmp_path):
    path = tmp_path / "portfolio.json"
    PaperPortfolio(500.0).save_state(path, ledger_identity="ledger-a")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["ledger_identity"] = ""
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="INVALID_PORTFOLIO_STATE"):
        PaperPortfolio(0).load_state(path)
