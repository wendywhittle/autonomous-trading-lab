# Autonomous Trading Lab

Research-first foundation for an autonomous trading system.

## Architecture

Market data → normalization → causal state → versioned strategy → JEV decision → deterministic risk → paper execution → immutable event ledger → replay/research loop.

Intelligence may propose decisions, typed evaluation may assess them, deterministic risk controls may block them, and execution remains a separately controlled capability.

## M0 boundary

M0 provides deterministic state construction, versioned decisions, hard risk limits, a kill switch, simulated paper fills, append-only event recording, and replay validation.

**Live brokerage execution is not implemented or enabled in M0.** No API keys or broker credentials belong in source control. A future live adapter must be an explicit promotion step behind independent tests and risk gates.

## Safety properties

- No future data may enter a state built at a historical cutoff.
- Strategy decisions carry an explicit immutable strategy version.
- Risk is a deterministic gate, not an LLM judgment.
- Paper execution is simulation only.
- The ledger is append-only at the application interface and currently uses a process-local writer lock; durable multi-process transactional storage is a later hardening step.
- Replay validates contiguous event sequencing.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Next layers

1. Provider-independent market-data adapters and normalization.
2. Strategy registry with immutable versions and experiment records.
3. Backtest/replay runner with explicit data snapshots.
4. Portfolio/P&L accounting and daily-loss controls.
5. Promotion gates and observability.
6. Broker adapter interface, initially disabled, followed by controlled paper-to-live promotion after evidence and review.
