"""Confidence-bucket reporting.

The Sniper v1 spec calls this "probably the most important research table":
confidence bucket x trades x gross edge x costs x net edge. It answers the
question every confidence-threshold strategy must answer honestly — does
higher model confidence actually correspond to higher realized net
expectancy, or is the model just confident and wrong?

Inputs are declared explicitly: one :class:`ScoredTrade` per completed trade
with the model confidence behind it and its gross P&L and cost. Attribution
of a trade's P&L to the confidence of its entry decision is the caller's
documented choice.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoredTrade:
    """One completed trade with the confidence that opened it."""

    confidence: float  #: model confidence in [0, 1] behind the entry decision
    gross_pnl: float  #: trade P&L before execution costs
    cost: float  #: all-in execution cost attributed to this trade

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("SCORED_TRADE_CONFIDENCE_OUT_OF_RANGE")
        if self.cost < 0:
            raise ValueError("SCORED_TRADE_NEGATIVE_COST")

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.cost


@dataclass(frozen=True)
class ConfidenceBucketRow:
    """Aggregated net-expectancy row for one confidence bucket."""

    bucket: str  #: human label, e.g. "70-75%"
    low: float  #: inclusive lower bound
    high: float  #: exclusive upper bound
    trades: int
    gross_edge: float
    costs: float
    net_edge: float

    @property
    def expectancy_per_trade(self) -> float:
        return self.net_edge / self.trades if self.trades else 0.0


def confidence_bucket_table(
    trades: list[ScoredTrade] | tuple[ScoredTrade, ...],
    *,
    bucket_width: float = 0.05,
    min_confidence: float = 0.5,
) -> tuple[ConfidenceBucketRow, ...]:
    """Aggregate scored trades into confidence buckets.

    Buckets span ``[min_confidence, 1.0]`` in steps of ``bucket_width``; the
    final bucket includes 1.0. A trade with confidence exactly on a boundary
    falls in the higher bucket (except 1.0, which stays in the last bucket).
    Empty buckets are included with zero trades so gaps in the evidence are
    visible rather than hidden.
    """
    if bucket_width <= 0 or bucket_width > 1:
        raise ValueError("BUCKET_INVALID_WIDTH")
    if not 0.0 <= min_confidence < 1.0:
        raise ValueError("BUCKET_INVALID_MIN_CONFIDENCE")

    bounds: list[float] = []
    low = min_confidence
    while low < 1.0:
        bounds.append(low)
        low = round(low + bucket_width, 10)
    bounds.append(1.0)

    rows: list[ConfidenceBucketRow] = []
    for index in range(len(bounds) - 1):
        bucket_low, bucket_high = bounds[index], bounds[index + 1]
        bucket_trades = [
            trade
            for trade in trades
            if (bucket_low <= trade.confidence < bucket_high)
            or (bucket_high == 1.0 and trade.confidence == 1.0)
        ]
        label = f"{round(bucket_low * 100)}-{round(bucket_high * 100)}%"
        rows.append(
            ConfidenceBucketRow(
                bucket=label,
                low=bucket_low,
                high=bucket_high,
                trades=len(bucket_trades),
                gross_edge=sum(t.gross_pnl for t in bucket_trades),
                costs=sum(t.cost for t in bucket_trades),
                net_edge=sum(t.net_pnl for t in bucket_trades),
            )
        )
    return tuple(rows)


def format_bucket_table(rows: tuple[ConfidenceBucketRow, ...]) -> str:
    """Render bucket rows as a plain-text table for research reports."""
    header = (
        f"{'bucket':>8} | {'trades':>6} | {'gross_edge':>10} | "
        f"{'costs':>10} | {'net_edge':>10} | {'exp/trade':>10}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.bucket:>8} | {row.trades:>6} | {row.gross_edge:>10.2f} | "
            f"{row.costs:>10.2f} | {row.net_edge:>10.2f} | "
            f"{row.expectancy_per_trade:>10.4f}"
        )
    return "\n".join(lines)
