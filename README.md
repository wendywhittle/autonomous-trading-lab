# Autonomous Trading Lab

Research-first foundation for an autonomous trading system.

## Architecture

Market data → normalization → causal state → versioned strategy → JEV decision → deterministic risk → paper execution → persistent portfolio/risk state → transactional event ledger → replay/research loop.

Intelligence may propose decisions, typed evaluation may assess them, deterministic risk controls may block them, and execution remains a separately controlled capability.

## M0 boundary

M0 provides deterministic state construction, immutable strategy-version handling, hard risk limits, a kill switch with audit events, simulated paper fills, persistent paper portfolio and risk-session state, restart-safe decision/order idempotency, transactional SQLite event recording, replay validation, and explicit promotion gates.

**Live brokerage execution is not implemented or enabled in M0.** No API keys or broker credentials belong in source control. The promotion gate refuses LIVE eligibility until a real live execution path and its controls are implemented and independently verified.

## Safety properties

- No future data may enter a state built at a historical cutoff.
- Strategy decisions carry an explicit immutable version.
- Registry storage is protected from caller mutation.
- Risk is a deterministic gate, not an LLM judgment.
- Daily-loss and drawdown checks use persisted session state when configured.
- Paper execution is simulation only.
- Portfolio state is atomically replaced on persistence and records applied order IDs for restart-safe paper recovery.
- The ledger uses SQLite WAL mode, transactional sequencing, and unique event IDs.
- Runtime recovery distinguishes an incomplete DECISION from an already completed ORDER or RISK_BLOCK.
- Duplicate paper order application is idempotent.
- Kill-switch activation is recorded in the ledger.
- Replay validates contiguous event sequencing and unique event IDs.
- Promotion cannot manufacture live execution capability; LIVE requires explicit human approval plus verified live execution controls.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Next layers

1. Add explicit experiment records and reproducible data snapshots.
2. Add concurrency/recovery stress tests around the transactional ledger.
3. Add observability, reconciliation, and operational failure handling.
4. Add provider-independent broker interfaces, initially disabled.
5. Add compliance/governance policy boundaries, audit requirements, and controlled promotion evidence.
6. Build the live execution boundary only after controls, permissions, reconciliation, and independent verification exist.

## Design principle

The system may become autonomous in execution, but it must never become autonomous in defining its own authority.
