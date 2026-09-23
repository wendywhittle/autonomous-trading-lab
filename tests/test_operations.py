from atlab.ledger import ImmutableLedger
from atlab.models import OrderStatus, PaperOrder, Side
from atlab.operations import inspect_operational_health, inspect_portfolio_file


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


def state(cash=1000, quantity=0, average_cost=0, realized_pnl=0, ids=None):
    return {
        "cash": cash,
        "position_quantity": quantity,
        "average_cost": average_cost,
        "realized_pnl": realized_pnl,
        "applied_order_ids": ids or [],
    }


def test_inspect_portfolio_file_reports_unreadable(tmp_path):
    result = inspect_portfolio_file(
        tmp_path / "missing.json",
        [],
        starting_cash=1000,
    )
    assert not result.reconciled
    assert result.errors == ("PORTFOLIO_STATE_UNREADABLE",)


def test_inspect_operational_health_aggregates_components(tmp_path):
    ledger = ImmutableLedger(tmp_path / "ledger.sqlite3")
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        '{"applied_order_ids":[],"average_cost":0,"cash":1000,'
        '"position_quantity":0,"realized_pnl":0}',
        encoding="utf-8",
    )

    health = inspect_operational_health(
        ledger,
        portfolio_path,
        [],
        starting_cash=1000,
    )

    assert health.healthy
    assert health.errors == ()


def test_inspect_operational_health_reports_corrupt_portfolio(tmp_path):
    ledger = ImmutableLedger(tmp_path / "ledger.sqlite3")
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text("{not-json", encoding="utf-8")

    health = inspect_operational_health(
        ledger,
        portfolio_path,
        [order("order-1", Side.BUY, 1, 100)],
        starting_cash=1000,
    )

    assert not health.healthy
    assert "PORTFOLIO_STATE_UNREADABLE" in health.errors
