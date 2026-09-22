from atlab.ledger import ImmutableLedger
from atlab.models import PaperOrder, OrderStatus, Side
from atlab.observability import inspect_ledger, reconcile_portfolio


def order(order_id, side, quantity, price):
    return PaperOrder(
        order_id=order_id,
        decision_id=f"decision-{order_id}",
        symbol="TEST",
        side=side,
        quantity=quantity,
        fill_price=price,
        notional=quantity * price,
        status=OrderStatus.FILLED,
    )


def test_inspect_ledger_reports_healthy_state(tmp_path):
    ledger = ImmutableLedger(tmp_path / "ledger.sqlite3")
    ledger.append("DECISION", "decision-1", {"decision_id": "decision-1"})
    ledger.append("ORDER", "order-1", {"decision_id": "decision-1"})

    health = inspect_ledger(ledger)

    assert health.healthy
    assert health.event_count == 2
    assert health.latest_sequence == 1
    assert health.event_type_counts == (("DECISION", 1), ("ORDER", 1))
    assert health.errors == ()


def test_inspect_ledger_reports_read_failure():
    class BrokenLedger:
        def read(self):
            raise OSError("broken")

    health = inspect_ledger(BrokenLedger())

    assert not health.healthy
    assert health.errors == ("LEDGER_READ_ERROR:OSError",)


def test_reconcile_portfolio_matches_ledger_orders():
    buy = order("order-1", Side.BUY, 2, 100)
    sell = order("order-2", Side.SELL, 1, 110)

    state = {
        "cash": 910,
        "position_quantity": 1,
        "average_cost": 100,
        "realized_pnl": 10,
        "applied_order_ids": ["order-1", "order-2"],
    }

    result = reconcile_portfolio(state, [buy, sell], starting_cash=1000)

    assert result.reconciled
    assert result.errors == ()


def test_reconcile_portfolio_detects_mismatch():
    buy = order("order-1", Side.BUY, 2, 100)
    state = {
        "cash": 700,
        "position_quantity": 2,
        "average_cost": 100,
        "realized_pnl": 0,
        "applied_order_ids": ["order-1"],
    }

    result = reconcile_portfolio(state, [buy], starting_cash=1000)

    assert not result.reconciled
    assert "PORTFOLIO_CASH_MISMATCH" in result.errors


def test_reconcile_portfolio_detects_order_missing_from_ledger():
    state = {
        "cash": 800,
        "position_quantity": 2,
        "average_cost": 100,
        "realized_pnl": 0,
        "applied_order_ids": ["missing-order"],
    }

    result = reconcile_portfolio(state, [], starting_cash=1000)

    assert not result.reconciled
    assert "PORTFOLIO_ORDER_MISSING_FROM_LEDGER" in result.errors
