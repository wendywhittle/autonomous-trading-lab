"""Command-line interface for the Autonomous Trading Lab.

Paper/research only. There is no live broker adapter; the default execution
path is simulated paper fills. ``atlab run`` drives the paper trading engine
end-to-end from a CSV file and persists portfolio/risk state plus the event
ledger. ``atlab backtest`` runs the same loop and prints deterministic
performance metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from .adapters import CsvMarketData
from .backtest import run_paper_backtest
from .charter import CHARTER_STOP_LOSS, charter_limits, charter_quantity_policy
from .evidence import (
    collect_evidence,
    live_evidence_notes,
    requirement_rows,
    risk_limits_notes,
)
from .ledger import ImmutableLedger
from .models import StrategyVersion
from .portfolio import PaperPortfolio
from .promotion import PromotionGate, PromotionMode
from .risk import DeterministicRiskEngine
from .runtime import PaperTradingEngine, TradingMode


def _strategy(args: argparse.Namespace) -> StrategyVersion:
    return StrategyVersion(
        strategy_id=args.strategy_id,
        version=args.strategy_version,
        hypothesis=args.hypothesis,
        parameters={
            "entry_return": args.entry_return,
            "entry_dip": args.entry_dip,
            "exit_bounce": args.exit_bounce,
        },
    )


def _risk_engine(args: argparse.Namespace) -> DeterministicRiskEngine:
    if args.risk_charter:
        return DeterministicRiskEngine(charter_limits())
    return DeterministicRiskEngine()


def _quantity_policy(args: argparse.Namespace):
    if args.risk_charter:
        return charter_quantity_policy()
    return None


def _stop_loss(args: argparse.Namespace):
    if args.risk_charter:
        return CHARTER_STOP_LOSS
    return None


def _engine(args: argparse.Namespace, workdir: Path) -> tuple[PaperTradingEngine, str]:
    data = CsvMarketData(args.csv)
    symbol = args.symbol
    observations = list(data.observations(symbol))
    if not observations:
        raise ValueError(f"CSV_NO_OBSERVATIONS_FOR_SYMBOL:{symbol}")
    engine = PaperTradingEngine(
        data,
        _strategy(args),
        _risk_engine(args),
        PaperPortfolio(args.starting_cash),
        ImmutableLedger(workdir / "ledger.sqlite3"),
        quantity=args.quantity,
        quantity_policy=_quantity_policy(args),
        stop_loss=_stop_loss(args),
        mode=TradingMode.PAPER,
        portfolio_state_path=workdir / "portfolio.json",
        risk_state_path=workdir / "risk_session.json",
    )
    return engine, symbol


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV file. Naive timestamps are interpreted as UTC.")
    parser.add_argument("--symbol", required=True, help="Symbol to trade.")
    parser.add_argument("--workdir", default="lab-workdir", help="State directory.")
    parser.add_argument("--starting-cash", type=float, default=10_000.0)
    parser.add_argument("--quantity", type=float, default=1.0)
    parser.add_argument("--strategy-id", default="momentum")
    parser.add_argument("--strategy-version", default="1.0.0")
    parser.add_argument("--hypothesis", default="CLI momentum threshold strategy.")
    parser.add_argument("--entry-return", type=float, default=0.001)
    parser.add_argument("--entry-dip", type=float, default=0.004,
                        help="Sniper: ENTER/BUY when latest return <= -entry-dip.")
    parser.add_argument("--exit-bounce", type=float, default=0.003,
                        help="Sniper: EXIT/SELL when latest return >= exit-bounce.")
    parser.add_argument("--risk-charter", action="store_true",
                        help="Enforce Wendy's risk charter v1 ($10/trade, $40 deployed, "
                             "$15 daily-loss halt, $50 drawdown stop, $5 per-position "
                             "stop-loss) with $10 target-notional sizing.")


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
    # H9: completion record consumed as promotion evidence by `atlab promote`.
    (workdir / "paper_run.json").write_text(
        json.dumps(
            {
                "completed": True,
                "symbol": symbol,
                "strategy_id": args.strategy_id,
                "strategy_version": args.strategy_version,
                "cycles": len(cycles),
                "trades": trades,
                "final_equity": cycles[-1].equity if cycles else None,
                "finished_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
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
        quantity_policy=_quantity_policy(args),
        stop_loss=_stop_loss(args),
        risk=_risk_engine(args),
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


def cmd_promote(args: argparse.Namespace) -> int:
    """Evaluate promotion evidence through the PromotionGate (H9).

    The gate is no longer advisory: this command constructs it, collects
    evidence that is actually verified locally (never assumed), keeps
    attested-but-unverified evidence explicit, and records the decision.
    A LIVE authorization is minted only if the evidence is fully eligible;
    with no live broker adapter in this repo, LIVE is unreachable and the
    command reports exactly which requirements fail.
    """
    target = PromotionMode(args.target.upper())
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    strategy = _strategy(args)

    print(f"collecting {target.value} promotion evidence (local verification)...")
    evidence, provenance = collect_evidence(
        workdir=workdir,
        csv_path=args.csv,
        symbol=args.symbol,
        strategy=strategy,
        quantity=args.quantity,
        human_approval_attestation=args.attest_human_approval,
    )

    gate = PromotionGate()
    result = gate.evaluate(evidence, target)

    notes = live_evidence_notes()
    print(f"target={target.value} eligible={result.eligible}")
    print("evidence:")
    for field, reason in requirement_rows(target):
        value = getattr(evidence, field)
        status = "pass" if value else "FAIL"
        line = f"  [{status}] {field}={value} ({provenance[field]})"
        if field in notes:
            line += f" -- {notes[field]}"
        if field == "risk_limits_active" and value:
            line += f" -- {risk_limits_notes()}"
        print(line)
    if not result.eligible:
        print("failed_requirements=" + "|".join(result.failed_requirements))

    record = {
        "target": target.value,
        "eligible": result.eligible,
        "failed_requirements": list(result.failed_requirements),
        "evidence": {
            field: {"value": getattr(evidence, field), "provenance": provenance[field]}
            for field, _reason in requirement_rows(target)
        },
        "human_approval_attestation": args.attest_human_approval,
        "workdir": str(workdir),
        "evaluated_at": datetime.now(UTC).isoformat(),
    }
    record_path = workdir / "promotion_record.json"
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"promotion record written to {record_path}")

    if not result.eligible:
        return 1
    if target is PromotionMode.LIVE:
        # Reachable only with fully eligible LIVE evidence. Minting requires
        # the HMAC signing key (H2); without it this fails closed.
        authorization = gate.authorize(evidence, target)
        print("LIVE execution authorization minted:")
        print(f"  authorization_id={authorization.authorization_id}")
        print(f"  issued_at={authorization.issued_at}")
        print(f"  expires_at={authorization.expires_at}")
        print(
            "  signature_present="
            f"{bool(authorization.signature)} (HMAC-SHA256, key from "
            "ATLAB_AUTH_SIGNING_KEY; the key is never printed)"
        )
    else:
        print("PROMOTION_ELIGIBLE:paper")
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

    promote_parser = subparsers.add_parser(
        "promote",
        help="Evaluate promotion evidence through the PromotionGate.",
    )
    _add_common(promote_parser)
    promote_parser.add_argument(
        "--target",
        required=True,
        choices=["paper", "live"],
        help="Promotion target mode.",
    )
    promote_parser.add_argument(
        "--attest-human-approval",
        default=None,
        help=(
            "Explicit human approval attestation (e.g. 'Name <role>: reason'). "
            "Recorded as attested, NOT independently verified."
        ),
    )
    promote_parser.set_defaults(func=cmd_promote)
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
