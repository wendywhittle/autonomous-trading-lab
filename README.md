# Autonomous Trading Lab

Research-first foundation for an autonomous trading system.

## Architecture

Market data → normalization → causal state → versioned strategy → JEV decision → deterministic risk → paper execution → persistent portfolio/risk state → transactional event ledger → replay/research loop.

Intelligence may propose decisions, typed evaluation may assess them, deterministic risk controls may block them, and execution remains a separately controlled capability.

## M0 boundary

M0 provides deterministic state construction, immutable strategy-version handling, hard risk limits, a kill switch with audit events, simulated paper fills, persistent paper portfolio and risk-session state, restart-safe decision/order idempotency, provider-independent broker contracts, durable execution intents, atomic execution-state-plus-audit transactions, operational observability and reconciliation, replay validation, reproducible data snapshots and experiment identity, concurrency stress coverage, and explicit promotion gates.

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
- Data snapshots are content-addressed so identical observations produce identical snapshot identity.
- Experiment identity binds a strategy version and parameters to an exact data snapshot.
- Concurrent ledger appends and same-key intent preparation are covered by stress tests.
- Execution intent state and its corresponding audit event commit or roll back as one SQLite transaction.
- Promotion cannot manufacture live execution capability; LIVE requires explicit human approval plus verified live execution controls.

## Development

python -m pip install -e ".[dev]"
pytest
ruff check .

## Next layers

1. Harden provider-side recovery so an UNKNOWN submission can be reconciled by client/idempotency key, not only broker order ID.
2. Add cash/leverage-aware risk limits and operational failure handling.
3. Add compliance/governance policy boundaries, audit requirements, and controlled promotion evidence.
4. Build the live execution boundary only after controls, permissions, provider-side idempotency, reconciliation, and independent verification exist.

## Design principle

The system may become autonomous in execution, but it must never become autonomous in defining its own authority.
