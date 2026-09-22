# Autonomous Trading Lab

Research-first foundation for an autonomous trading system.

## Architecture

Market data → normalization → causal state → versioned strategy → JEV decision → deterministic risk → paper execution → immutable event ledger → replay/research loop.

Intelligence may propose decisions, typed evaluation may assess them, deterministic risk controls may block them, and execution remains a separately controlled capability.

## M0 boundary

M0 provides deterministic state construction, versioned decisions, hard risk limits, a kill switch, simulated paper fills, append-only event recording, replay validation, persistent paper portfolio state, persisted risk-session state, and explicit promotion gates.

**Live brokerage execution is not implemented or enabled in M0.** No API keys or broker credentials belong in source control. The promotion gate also refuses LIVE eligibility until a real live execution path and its controls are implemented and independently verified.

## Safety properties

- No future data may enter a state built at a historical cutoff.
- Strategy decisions carry an explicit version.
- Risk is a deterministic gate, not an LLM judgment.
- Daily-loss and drawdown checks are deterministic and use persisted session state when configured.
- Paper execution is simulation only.
- Portfolio state is atomically persisted when a state path is configured, so a process restart does not reset the paper account.
- The ledger is append-only at the application interface and currently uses a process-local writer lock; durable multi-process transactional storage is a later hardening step.
- Replay validates contiguous event sequencing.
- Promotion cannot manufacture live execution capability; LIVE requires explicit human approval plus verified live execution controls.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Next layers

1. Harden immutable strategy objects and registry storage.
2. Add explicit experiment records and reproducible data snapshots.
3. Strengthen ledger durability for multi-process operation.
4. Add observability, failure recovery, and kill-switch audit events.
5. Define a provider-independent broker interface, initially disabled.
6. Build controlled paper-to-live promotion evidence only after the execution path, controls, and independent verification exist.
