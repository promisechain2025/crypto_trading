# Profitability Backtest — `crypto_bot.py`

Reproduce with `python3 backtest.py`. Outputs land in `backtest_output/`.

## TL;DR

**The bot is not profitable.** Across four 800-day price scenarios (bull, bear,
sideways, high-vol alt), the bot loses money in absolute terms in every run
that produces any trades, with negative Sharpe in every case. With its real
trading rule unmodified (sentiment + on-chain feeds disabled, as in any
historical backtest) it cannot trade at all — the weighted score is
mathematically incapable of clearing the ±0.5 threshold.

## How the test was constructed

External data hosts (CoinGecko, Yahoo) are blocked from this sandbox, so the
harness generates seeded GBM price paths with simple volatility clustering.
The harness implements the bot's deterministic signals as ports of the
methods in `crypto_bot.py`:

| Signal | Source in bot | Implementation in backtest |
|---|---|---|
| MA crossover | `compute_moving_averages` (50/200) | Same |
| RSI | `compute_rsi` (14, 30/70) | Same |
| Trend | `fetch_price_action` daily-return sign | Same on daily bars |
| AI | LSTM trained per call with Optuna | Mean-log-return drift proxy (cheap deterministic stand-in for `sign(predicted - price)`) |
| Sentiment | Twitter+Reddit RoBERTa average | (a) zeroed; (b) momentum-correlated noisy proxy in [-1, 1] |
| On-chain | Etherscan tx count threshold | zeroed (no historical replay) |

Position sizing matches the bot: **50% of cash USDT per buy, full liquidation
on sell**, 10 bps fee per side, $10,000 starting capital.

## Run A — sentiment & on-chain neutralized (faithful offline backtest)

| scenario | buy & hold | bot | alpha | trades | score range |
|---|---:|---:|---:|---:|---|
| bull_btc_like   | -75.24% | **0.00%** | +75.24% | 0 / 0 | -0.40 to +0.35 |
| bear_btc_like   | -66.11% | **0.00%** | +66.11% | 0 / 0 | -0.40 to +0.25 |
| sideways_choppy | +7.56%  | **0.00%** |  -7.56% | 0 / 0 | -0.40 to +0.40 |
| high_vol_alt    | -82.15% | **0.00%** | +82.15% | 0 / 0 | -0.40 to +0.40 |

**Why zero trades.** Without sentiment (30% weight) and on-chain (15%), the
remaining four signals contribute at most `0.15 + 0.10 + 0.15 + 0.15 = 0.55`
when every one of them agrees. To exceed the ±0.5 threshold requires:

1. MA crossover on this exact bar (rare — typically 1–5 events per 800 days), **and**
2. RSI is in extreme territory (<30 or >70), **and**
3. Daily trend agrees, **and**
4. AI prediction direction agrees.

In 600+ evaluated bars per scenario, this confluence never occurred. The
"alpha" column above is misleading — it just reflects the bot sitting in
cash while buy-and-hold paths drifted down.

## Run B — with synthetic sentiment stream

To exercise the trading branch, sentiment is replaced with a momentum-tracking
noisy proxy (7-day return z-score × 0.6 + N(0, 0.4²), clipped to [-1, 1]).
This is a charitable stand-in for live social-media sentiment.

| scenario | buy & hold | bot | alpha | DD | Sharpe | buys/sells | win rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| bull_btc_like   | -75.24% | **-19.59%** | +55.65% | -45.87% | -0.24 | 14 / 6 | 83.3% |
| bear_btc_like   | -66.11% | **-51.51%** | +14.60% | -63.15% | -0.84 | 22 / 8 | 62.5% |
| sideways_choppy | +7.56%  | **-34.80%** | -42.37% | -53.99% | -0.47 | 28 / 9 | 77.8% |
| high_vol_alt    | -82.15% | **-46.22%** | +35.93% | -74.20% | -0.52 | 14 / 7 | 42.9% |

**Observations.**

- **Absolute returns are negative in all scenarios.** Even the "best" outcome
  is a 19.6% loss.
- **Sharpe is negative in every scenario.** The strategy carries equity
  drawdowns of 45–74% with no compensating return.
- **Sideways-market loss is the most damning.** Buy-and-hold made +7.6%; the
  bot lost 34.8%. Choppy markets generate signal whipsaw, and the 50% capital
  commitment per buy compounds the bleed.
- **High win rate is misleading.** 83.3% in the bull case, 77.8% in sideways
  — yet the bot still loses, because losing trades are far larger than
  winners. The MA-crossover and RSI-mean-reversion signals tend to take small
  profits and let losers run, classic for this signal mix.
- **Apparent "alpha" vs. buy-and-hold is path-dependent.** Three of the four
  GBM seeds drifted down hard with σ ≥ 0.7, so any cash-heavy strategy looks
  like alpha. That's not skill, that's a denominator effect.

## Run C — threshold sensitivity (bull scenario, with sentiment)

Lowering the buy/sell threshold to fire more often does **not** turn this
strategy profitable.

| threshold | bot return | DD | Sharpe | buys / sells | win rate |
|---:|---:|---:|---:|---:|---:|
| ±0.50 | -19.59% | -45.87% | -0.24 | 14 / 6   | 83.3% |
| ±0.40 | -38.98% | -50.35% | -0.64 | 34 / 12  | 83.3% |
| ±0.30 | -28.21% | -42.90% | -0.31 | 92 / 18  | 77.8% |
| ±0.20 | -16.95% | -42.25% | -0.09 | 141 / 33 | 60.6% |
| ±0.10 | -24.54% | -50.53% | -0.28 | 203 / 54 | 66.7% |
| ±0.05 | -27.81% | -48.79% | -0.34 | 224 / 62 | 66.1% |

No threshold yields positive return or positive Sharpe.

## Caveats

- **Backtest can't replicate live edge.** If the real edge lives in the
  sentiment + on-chain components (combined 45% weight), a price-only
  backtest will understate it. But Run B's noisy momentum-correlated
  sentiment is a reasonable stand-in, and it didn't help.
- **Synthetic prices, not historical.** The harness uses GBM, not real BTC
  data, because data hosts are blocked here. Re-run with real OHLCV by
  replacing `generate_prices()` with a CSV loader; the rest of the harness
  is data-agnostic.
- **AI signal is a stand-in, not the actual LSTM.** The LSTM in the bot
  re-trains itself per decision call (with 20 Optuna trials × 10 epochs),
  which is too expensive to backtest 600× and would overfit the GBM path
  anyway. The drift proxy reproduces the sign behavior the rest of the bot
  consumes (`ai_val = sign(predicted - price)`).

## Concrete issues found in the bot itself

These are independent of the backtest result and worth flagging:

1. **`decide_trade()` re-trains the LSTM and runs 20 Optuna trials on every
   call, for every crypto.** At the configured 5-minute interval that's
   tens of minutes of GPU/CPU work per cycle, on top of all the API calls.
2. **`execute_trade_on_dex` sell branch computes "profit" wrong:**
   `profit = amount_usdt - (self.capital.get('usdt', 10000) * (amount_crypto / sum(self.positions.values() or [1])))`
   uses current cash and fraction-of-position rather than the cost basis at
   entry, so reported P&L will not match actual P&L.
3. **`execute_grid_trade`, `execute_market_making`, `execute_scalp_trade` are
   `print()` stubs**, not actual orders, despite being called in the live
   loop.
4. **50% of cash per buy with no per-trade stop-loss.** Two unlucky entries
   draw the account down 75%.
5. **`predict_next_price()` calls `self.scaler.transform()` after the scaler
   was fit on the same series in `train_ai_model()`.** If `train_ai_model`
   isn't called first (e.g. on insufficient data), `transform` will raise
   on an unfit scaler.
6. **Aster path uses placeholder `'0xYourAsterContractAddress'` and
   `'0x...'` calldata** — sending such transactions live would burn gas
   without trading.

## Bottom line

On price-driven signals alone, the bot doesn't trade. With a plausible
sentiment proxy added, it trades but loses money in every regime tested,
with negative risk-adjusted return. The strategy needs work before any live
deployment is justified — at minimum: (a) decouple training from the trading
loop, (b) add a stop-loss and shrink position size, (c) re-tune signal
weights so the score can reach the threshold without sentiment carrying it,
(d) fix the P&L accounting and the placeholder DEX calls.
