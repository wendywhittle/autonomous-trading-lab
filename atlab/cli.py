"""Command-line interface for the Autonomous Trading Lab.

Paper/research only. There is no live broker adapter; the default execution
path is simulated paper fills. ``atlab run`` drives the paper trading engine
end-to-end from a CSV file and persists portfolio/risk state plus the event
ledger. ``atlab backtest`` runs the same loop and prints deterministic
performance metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .adapters import CsvMarketData
from .backtest import run_paper_backtest
from .ledger import ImmutableLedger
from .models import StrategyVersion
from .portfolio import PaperPortfolio
from .risk import DeterministicRiskEngine
from .runtime import PaperTradingEngine, TradingMode


def _strategy(args: argparse.Namespace) -> StrategyVersion:
    return StrategyVersion(
        strategy_id=args.strategy_id,
        version=args.strategy_version,
        hypothesis=args.hypothesis,
        parameters={"entry_return": args.entry_return},
    )


def _engine(args: argparse.Namespace, workdir: Path) -> tuple[PaperTradingEngine, str]:
    data = CsvMarketData(args.csv)
    symbol = args.symbol
    observations = list(data.observations(symbol))
    if not observations:
        raise ValueError(f"CSV_NO_OBSERVATIONS_FOR_SYMBOL:{symbol}")
    engine = PaperTradingEngine(
        data,
        _strategy(args),
        DeterministicRiskEngine(),
        PaperPortfolio(args.starting_cash),
        ImmutableLedger(workdir / "ledger.sqlite3"),
        quantity=args.quantity,
        mode=TradingMode.PAPER,
        portfolio_state_path=workdir / "portfolio.json",
        risk_state_path=workdir / "risk_session.json",
    )
    return engine, symbol


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV file.")
    parser.add_argument("--symbol", required=True, help="Symbol to trade.")
    parser.add_argument("--workdir", default="lab-workdir", help="State directory.")
    parser.add_argument("--starting-cash", type=float, default=10_000.0)
    parser.add_argument("--quantity", type=float, default=1.0)
    parser.add_argument("--strategy-id", default="momentum")
    parser.add_argument("--strategy-version", default="1.0.0")
    parser.add_argument("--hypothesis", default="CLI momentum threshold strategy.")
    parser.add_argument("--entry-return", type=float, default=0.001)


def cmd_run(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    engine, symbol = _engine(args, workdir)
    cycles = engine.run(symbol)
    trades = sum(1 for cycle in cycles if cycle.order is not None)
    print(f"symbol={symbol} cycles={len(cycles)} trades={trades}")
    for cycle in cycles:
        order = cycle.order
        if order is not None:
            print(
                f"  {cycle.decision.action.value} {order.side.value} "
                f"qty={order.quantity} price={order.fill_price:.4f} "
                f"equity={cycle.equity:.2f}"
            )
    print(f"final equity={cycles[-1].equity:.2f}" if cycles else "no cycles")
    print(f"state persisted under {workdir}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    data = CsvMarketData(args.csv)
    observations = list(data.observations(args.symbol))
    if not observations:
        raise ValueError(f"CSV_NO_OBSERVATIONS_FOR_SYMBOL:{args.symbol}")
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    result = run_paper_backtest(
        observations,
        _strategy(args),
        starting_cash=args.starting_cash,
        quantity=args.quantity,
        workdir=workdir,
    )
    metrics = result.metrics
    assert metrics is not None
    print(f"symbol={args.symbol} strategy={args.strategy_id}@{args.strategy_version}")
    print(f"observations={metrics.num_observations} decisions={metrics.num_decisions}")
    print(f"trades={metrics.num_trades} round_trips={metrics.num_round_trips}")
    print(f"starting_equity={metrics.starting_equity:.2f}")
    print(f"ending_equity={metrics.ending_equity:.2f}")
    print(f"total_return={metrics.total_return:.4%}")
    print(f"max_drawdown={metrics.max_drawdown:.4%}")
    print(f"sharpe_ratio={metrics.sharpe_ratio:.4f} (rf=0, {252:.0f} periods/yr)")
    print(f"win_rate={metrics.win_rate:.2%}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlab", description="Autonomous Trading Lab (paper/research only).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Drive the paper trading engine end-to-end from CSV.")
    _add_common(run_parser)
    run_parser.set_defaults(func=cmd_run)

    backtest_parser = subparsers.add_parser("backtest", help="Backtest a strategy on CSV data and print metrics.")
    _add_common(backtest_parser)
    backtest_parser.set_defaults(func=cmd_backtest)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
