# Sniper v1 — system specification

*(Received from Wendy 2026-09-24. Saved as reference; not yet approved for build. See assessment in chat.)*

## 1. Objective

Goal: identify rare, short-horizon opportunities where predicted price movement exceeds all-in execution costs by a sufficient margin.

Initial scope

| Parameter | V1 |
|---|---|
| Asset | 1 highly liquid instrument |
| Venue | 1 exchange/broker |
| Horizon | 100 ms–5 min |
| Direction | Long + short |
| Position | One position at a time |
| Strategy | Order-flow/microstructure |
| Execution | Limit-first, market fallback |
| Mode | Backtest → replay → paper → guarded live |
| Primary metric | Net expectancy after costs |

The system should be allowed to say NO TRADE most of the time.

## 2. Data layer

Capture raw market data without transforming it before storage.

Required streams: timestamp_exchange, timestamp_receive, sequence_number, bid_px[1..5], bid_sz[1..5], ask_px[1..5], ask_sz[1..5], last_trade_px, last_trade_sz, trade_side, halt/status messages.

Also record: order acknowledgements, order submissions, cancellations, fills, rejections, latency, fees.

Every event gets both exchange_timestamp and local_monotonic_timestamp (distinguishes market alpha from latency artifacts).

## 3. Feature engine

Book imbalance per level k: I_k = (sum B_i − sum A_i) / (sum B_i + sum A_i); generate imbalance_L1/L3/L5.

Microprice = (ask·bidSize + bid·askSize) / (bidSize+askSize); microprice_edge, microprice_edge_bps.

Trade flow over rolling windows (100ms, 250ms, 500ms, 1s, 3s, 10s, 30s): buy/sell/signed volume, trade_count, avg_trade_size, aggressive_buy_ratio.

Book dynamics: bid/ask add/cancel rates, depth_acceleration, spread_change, liquidity_withdrawal/replenishment.

## 4. Regime engine

Classify: TREND, MEAN_REVERT, HIGH_VOL, LOW_VOL, THIN_LIQUIDITY, NORMAL, STRESSED (simple classifier OK for V1: realized_volatility, spread_percentile, depth_percentile, trade_intensity, short_term_autocorrelation, price_displacement).

Hard vetoes: spread > max_spread, depth < min_depth, volatility > max_vol, data/sequence gap, exchange_status != NORMAL, latency > max_latency. Any veto → NO TRADE.

## 5. Alpha model

Two models: A (direction) P(ΔP > threshold | X); B (magnitude) E[|ΔP| | X]. Interpretable first: logistic regression, gradient-boosted trees, calibrated ensemble. No LSTM/Transformer unless simpler models leave residual alpha.

## 6. Target definition

Label tradeability, not direction. For a long: R_net = P_target − P_entry − fees − spread − slippage, including adverse excursion. LONG_WIN = future_high ≥ entry + target AND future_low > entry − stop. Horizons: 100ms → 5min.

## 7. Sniper score

Score = P(win) × ExpectedMove − ExecutionCost; expected_edge = P_win·expected_profit − (1−P_win)·expected_loss − fees − slippage − adverse_selection. Trade only if expected_edge > minimum_edge, model_confidence > threshold, regime permitted.

## 8. Entry logic

Long requires: REGIME_OK, P(up) ≥ 0.70 (initial research parameter, not sacred), expected_net_edge ≥ 2× estimated_cost, microprice_edge > threshold, L1 imbalance > threshold, trade_flow > threshold, spread ≤ max, liquidity ≥ minimum, no risk veto. Short symmetrical.

## 9. Execution engine

Signal generation separated from execution. Policy: check book → estimate fill probability → estimate adverse selection → passive order if justified → reprice/cancel after timeout → cross spread only if alpha exceeds crossing cost → abort on material change. Every order records signal_id, decision/submit/ack/fill timestamps, price, quantity, fee, slippage.

## 10. Position management

Max 1 open position, 1 outstanding entry order, 1 outstanding exit order. Exit types: TAKE_PROFIT, STOP_LOSS, ALPHA_DECAY, TIME_STOP, REGIME_BREAK, EMERGENCY. Alpha-decay exit: exit if P(win) falls below threshold, microprice reverses, or flow advantage disappears — even without TP/SL.

## 11. Risk engine

Sits outside the strategy; can veto anything. Limits: position size, notional, daily loss, consecutive losses, orders/sec, cancel rate, slippage, spread, latency, data age. Circuit breakers (disable + require operator intervention): sequence gap, unexpected position, duplicate order, ack timeout, abnormal fill, latency spike, spread explosion, liquidity collapse, daily loss limit, confidence anomaly.

## 12. Backtester

Event-driven, not candle-based: historical L2/L5 → event replay → features → model → simulated book → queue/fill model → fees+latency+slippage → portfolio → metrics. Must model latency, queue position, partial fills, cancellations, spread crossing, market impact, fees, missed fills.

## 13. Research protocol

Chronological splits only (TRAIN / VALIDATION / TEST / FORWARD-PAPER); never shuffle time series. Walk-forward evaluation.

## 14. Metrics

Trading: net P&L, expectancy/trade, profit factor, max drawdown, Sharpe, Sortino, win rate, avg winner/loser. Execution: implementation shortfall, slippage, fill probability, maker/taker ratio, queue loss, latency, adverse selection. Model: AUC, precision at high-confidence threshold, calibration, Brier score, prediction decay, performance by regime/time of day. Key table: confidence bucket × trades × gross edge × costs × net edge.

## 15. Autonomous research loop

Market Data → Feature Lab → Candidate Models → walk-forward test → cost-aware test → robustness tests → model registry → paper trading → promotion gate → restricted live. Every experiment: experiment_id, dataset_version, feature_version, model_version, hyperparameters, random_seed, cost_model, evaluation_period, results.

## 16. Model promotion gate

Require: positive net expectancy, survives realistic fees/slippage/latency, walk-forward, multiple regimes, no feature leakage, parameter stability, paper agrees with sim, risk limits tested. Stages: RESEARCH → SHADOW → PAPER → MICRO-LIVE → LIMITED LIVE, each with separate config and capital ceiling.

## 17. Suggested repository

sniper/ with data/, features/, models/, execution/, risk/, backtest/, research/, monitoring/, config/ (research.yaml, paper.yaml, live.yaml).

## 18. V1 research hypothesis

"Can short-term order-flow imbalance plus microprice displacement predict a sufficiently large next move to overcome realistic execution costs?" Run through the entire pipeline before adding features; add one feature family at a time, measuring incremental out-of-sample improvement.
