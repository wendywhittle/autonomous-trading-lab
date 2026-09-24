# Autonomous Trading Lab

Research-first foundation for an autonomous trading system.

## Architecture

Market data → normalization → causal state → versioned strategy → JEV decision → deterministic risk → paper execution → persistent portfolio/risk state → transactional event ledger → replay/research loop.

Intelligence may propose decisions, typed evaluation may assess them, deterministic risk controls may block them, and execution remains a separately controlled capability.

JEV is an explicit evaluation boundary, not a simulated capability. The repository defines a provider-independent JEV adapter contract plus a real TypeSafe/Jev implementation. The real adapter reads TYPESAFE_API_KEY from the environment, sends typed Choice judgments to Jev, and maps the returned choice/confidence into the trading decision contract. No local evaluator is presented as JEV.

## M0 boundary

M0 provides deterministic state construction, immutable strategy-version handling, hard risk limits, a kill switch with audit events, simulated paper fills, persistent paper portfolio and risk-session state, restart-safe decision/order idempotency, provider-independent broker contracts, durable execution intents, atomic execution-state-plus-audit transactions, operational observability and reconciliation, replay validation, reproducible data snapshots and experiment identity, concurrency stress coverage, explicit promotion gates, a CLI (`atlab run`, `atlab backtest`, `atlab promote`), a CSV OHLCV market-data adapter, and deterministic backtest metrics (equity curve, max drawdown, Sharpe ratio, total return, win rate, trade counts).

**Live brokerage execution is not implemented or enabled in M0.** No API keys or broker credentials belong in source control. The promotion gate refuses LIVE eligibility until a real live execution path and its controls are implemented and independently verified.

The LIVE submission path itself **is implemented**: `ExecutionCoordinator.submit()` supports a broker whose mode is `LIVE`, enforcing a signed execution authorization (HMAC, keyed from the `ATLAB_AUTH_SIGNING_KEY` environment variable), durable intent binding, persisted LIVE risk authority with optimistic concurrency, revocation semantics, and atomic result/ledger/risk-state persistence. What is absent is **a real broker adapter**: no `BrokerMode.LIVE` adapter class exists for any actual brokerage, and the default broker remains `DisabledBroker`. The LIVE path can therefore only ever be exercised with test doubles; it cannot reach a real broker. This is a deliberate paper/research boundary, not a placeholder description: the controls are real and tested, the provider endpoint is missing and must remain an explicit, separate decision.

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

### CLI

Paper/research only. No JEV API or live broker is required.

```bash
atlab backtest --csv data/sample_ohlcv.csv --symbol SYNTH --workdir ./lab-backtest
atlab run --csv data/sample_ohlcv.csv --symbol SYNTH --workdir ./lab-run
```

`data/sample_ohlcv.csv` is a small synthetic daily OHLCV series (seeded PRNG, clearly labeled) for smoke-testing the CSV adapter and backtest metrics. `atlab run` drives the paper trading engine end-to-end and persists the portfolio state, risk-session state, and event ledger under `--workdir`. `atlab backtest` runs the same loop and prints deterministic metrics: trade counts, equity, total return, max drawdown, Sharpe ratio (risk-free 0), and win rate over completed round trips.

### Live Jev connectivity

The repository includes a manual GitHub Actions workflow, `Jev live smoke`, that uses the repository secret `TYPESAFE_API_KEY` to make one real TypeSafe/Jev evaluation. It is deliberately manual so ordinary CI never spends API calls. Live brokerage execution remains disabled.

## Next layers

1. Strategy research surface: parameter sweeps and experiment tracking on top of the backtest metrics, reusing the content-addressed snapshots and experiment identity.
2. Richer market-data adapters (additional venues/formats) behind the same provider-independent contract; no future leakage by construction.
3. Cash/leverage-aware risk limit calibration tooling and operational runbooks for paper deployment.
4. Build the live execution boundary only after controls, permissions, provider-side idempotency, reconciliation, and independent verification exist. The LIVE control-plane scaffold documents the intended shape; a real broker adapter remains a separate, explicit decision.

## Design principle

The system may become autonomous in execution, but it must never become autonomous in defining its own authority.
