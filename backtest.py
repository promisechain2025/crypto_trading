"""
Profitability backtest for CryptoTradingBot in crypto_bot.py.

The bot fuses six signals into one weighted score:

    sentiment * 0.30 + MA * 0.15 + trend * 0.10 + on_chain * 0.15
                     + RSI * 0.15 + AI    * 0.15

Buy if score > 0.5, sell if < -0.5, else hold. On buy, 50% of cash USDT is
deployed; on sell, the entire position is liquidated.

This harness replays the deterministic, price-only signals (MA crossover,
RSI, daily-return trend, an LSTM-substitute momentum proxy) bar by bar.
Sentiment and on-chain are fixed at 0 because they cannot be reconstructed
from historical price alone; that means the backtest exercises 60% of the
score weight (MA + trend + RSI + AI) and the threshold of |score| > 0.5
becomes "all four price-derived signals must agree".

Three regimes are tested with seeded GBM + volatility clustering so results
are reproducible without network access to a price API.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Synthetic price generation
# --------------------------------------------------------------------------

def generate_prices(
    n_days: int,
    mu_annual: float,
    sigma_annual: float,
    start_price: float,
    seed: int,
    vol_cluster: float = 0.15,
) -> pd.DataFrame:
    """Geometric Brownian Motion with simple GARCH-like volatility clustering."""
    rng = np.random.default_rng(seed)
    dt = 1 / 365.0
    drift = (mu_annual - 0.5 * sigma_annual ** 2) * dt
    base_diffusion = sigma_annual * math.sqrt(dt)

    sigma_t = np.empty(n_days)
    sigma_t[0] = base_diffusion
    z = rng.standard_normal(n_days)
    log_returns = np.empty(n_days)
    for t in range(n_days):
        if t > 0:
            sigma_t[t] = math.sqrt(
                (1 - vol_cluster) * base_diffusion ** 2
                + vol_cluster * (sigma_t[t - 1] * z[t - 1]) ** 2
            )
        log_returns[t] = drift + sigma_t[t] * z[t]

    prices = start_price * np.exp(np.cumsum(log_returns))
    idx = pd.date_range("2022-01-01", periods=n_days, freq="D")
    return pd.DataFrame({"price": prices}, index=idx)


# --------------------------------------------------------------------------
# Bot signal replicas (lifted from crypto_bot.py)
# --------------------------------------------------------------------------

SHORT_MA_PERIOD = 50
LONG_MA_PERIOD = 200
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
BUY_THRESHOLD = 0.5
SELL_THRESHOLD = -0.5
RISK_FRACTION = 0.5  # bot risks 50% of cash on buy
FEE_BPS = 10  # 10 bps per side; round-trip = 20 bps


def ma_signal(prices: pd.Series) -> str:
    """Replicates compute_moving_averages: 'buy' on golden cross, 'sell' on death cross."""
    if len(prices) < LONG_MA_PERIOD + 1:
        return "hold"
    short_ma = prices.rolling(SHORT_MA_PERIOD).mean()
    long_ma = prices.rolling(LONG_MA_PERIOD).mean()
    if short_ma.iloc[-1] > long_ma.iloc[-1] and short_ma.iloc[-2] <= long_ma.iloc[-2]:
        return "buy"
    if short_ma.iloc[-1] < long_ma.iloc[-1] and short_ma.iloc[-2] >= long_ma.iloc[-2]:
        return "sell"
    return "hold"


def rsi_value(prices: pd.Series) -> float:
    """Replicates compute_rsi exactly (simple moving average of gains/losses)."""
    if len(prices) < RSI_PERIOD + 1:
        return 50.0
    delta = prices.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(RSI_PERIOD).mean().iloc[-1]
    avg_loss = loss.rolling(RSI_PERIOD).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def trend_signal(prices: pd.Series) -> int:
    """Replicates fetch_price_action's 24h trend on a daily series."""
    if len(prices) < 2:
        return 0
    ret = prices.iloc[-1] / prices.iloc[-2] - 1
    return 1 if ret > 0 else -1 if ret < 0 else 0


def ai_prediction(prices: pd.Series, lookback: int = 20) -> float:
    """Stand-in for the LSTM: predicts next price as last + mean recent log-return.

    The real bot trains a 60-step LSTM with Optuna tuning on each call. For
    a backtest we need something deterministic and cheap that captures the
    'AI says price is rising/falling' direction, which is what the bot uses
    (ai_val = sign(predicted - current)).
    """
    if len(prices) < lookback + 1:
        return float(prices.iloc[-1])
    log_rets = np.log(prices).diff().iloc[-lookback:]
    drift = log_rets.mean()
    return float(prices.iloc[-1] * math.exp(drift))


def combined_score(
    sentiment: float,
    ma_val: int,
    price_val: int,
    on_chain_val: int,
    rsi_val: int,
    ai_val: int,
) -> float:
    return (
        sentiment * 0.30
        + ma_val * 0.15
        + price_val * 0.10
        + on_chain_val * 0.15
        + rsi_val * 0.15
        + ai_val * 0.15
    )


# --------------------------------------------------------------------------
# Backtest engine
# --------------------------------------------------------------------------

@dataclass
class BacktestResult:
    name: str
    starting_capital: float
    ending_equity: float
    buy_hold_equity: float
    total_return_pct: float
    buy_hold_return_pct: float
    max_drawdown_pct: float
    sharpe: float
    n_buys: int
    n_sells: int
    win_rate_pct: float
    trades: list = field(default_factory=list)
    equity_curve: pd.Series = None


def run_backtest(
    name: str,
    prices: pd.DataFrame,
    starting_capital: float = 10_000.0,
    sentiment_stream: pd.Series | None = None,
    buy_threshold: float = BUY_THRESHOLD,
    sell_threshold: float = SELL_THRESHOLD,
    diagnostics: dict | None = None,
) -> BacktestResult:
    cash = starting_capital
    position = 0.0
    last_buy_value = 0.0
    trades = []
    equity_curve = []

    fee = FEE_BPS / 10_000.0

    p = prices["price"]
    if diagnostics is not None:
        diagnostics.setdefault("ma_buy", 0)
        diagnostics.setdefault("ma_sell", 0)
        diagnostics.setdefault("rsi_oversold", 0)
        diagnostics.setdefault("rsi_overbought", 0)
        diagnostics.setdefault("ai_up", 0)
        diagnostics.setdefault("ai_down", 0)
        diagnostics.setdefault("scores", [])
        diagnostics.setdefault("bars", 0)

    for i in range(LONG_MA_PERIOD + 1, len(p)):
        window = p.iloc[: i + 1]
        price = window.iloc[-1]

        ma = ma_signal(window)
        rsi = rsi_value(window)
        trend = trend_signal(window)
        predicted = ai_prediction(window)

        ma_v = 1 if ma == "buy" else -1 if ma == "sell" else 0
        rsi_v = 1 if rsi < RSI_OVERSOLD else -1 if rsi > RSI_OVERBOUGHT else 0
        ai_v = 1 if predicted > price else -1 if predicted < price else 0

        sentiment = 0.0 if sentiment_stream is None else float(sentiment_stream.iloc[i])
        score = combined_score(sentiment, ma_v, trend, 0, rsi_v, ai_v)

        if diagnostics is not None:
            diagnostics["bars"] += 1
            if ma_v == 1: diagnostics["ma_buy"] += 1
            if ma_v == -1: diagnostics["ma_sell"] += 1
            if rsi_v == 1: diagnostics["rsi_oversold"] += 1
            if rsi_v == -1: diagnostics["rsi_overbought"] += 1
            if ai_v == 1: diagnostics["ai_up"] += 1
            if ai_v == -1: diagnostics["ai_down"] += 1
            diagnostics["scores"].append(score)

        decision = "hold"
        if score > buy_threshold:
            decision = "buy"
        elif score < sell_threshold:
            decision = "sell"

        if decision == "buy" and cash > 1.0:
            spend = cash * RISK_FRACTION
            qty = (spend * (1 - fee)) / price
            position += qty
            cash -= spend
            last_buy_value = spend
            trades.append({
                "date": window.index[-1].isoformat(),
                "side": "buy",
                "price": float(price),
                "qty": float(qty),
                "score": float(score),
                "rsi": float(rsi),
            })
        elif decision == "sell" and position > 0:
            proceeds = position * price * (1 - fee)
            pnl = proceeds - last_buy_value
            cash += proceeds
            trades.append({
                "date": window.index[-1].isoformat(),
                "side": "sell",
                "price": float(price),
                "qty": float(position),
                "score": float(score),
                "rsi": float(rsi),
                "pnl": float(pnl),
            })
            position = 0.0
            last_buy_value = 0.0

        equity_curve.append(cash + position * price)

    # Mark-to-market liquidate at end (informational, not a real trade)
    final_price = p.iloc[-1]
    ending_equity = cash + position * final_price

    eq = pd.Series(equity_curve, index=p.index[LONG_MA_PERIOD + 1:])
    daily_ret = eq.pct_change().dropna()
    sharpe = (
        float(daily_ret.mean() / daily_ret.std() * math.sqrt(365))
        if daily_ret.std() > 0
        else 0.0
    )
    running_max = eq.cummax()
    drawdown = (eq - running_max) / running_max
    max_dd = float(drawdown.min())

    bh_qty = starting_capital / p.iloc[LONG_MA_PERIOD + 1]
    bh_equity = bh_qty * final_price

    sells = [t for t in trades if t["side"] == "sell"]
    wins = sum(1 for t in sells if t.get("pnl", 0) > 0)
    win_rate = (wins / len(sells) * 100) if sells else 0.0

    return BacktestResult(
        name=name,
        starting_capital=starting_capital,
        ending_equity=ending_equity,
        buy_hold_equity=bh_equity,
        total_return_pct=(ending_equity / starting_capital - 1) * 100,
        buy_hold_return_pct=(bh_equity / starting_capital - 1) * 100,
        max_drawdown_pct=max_dd * 100,
        sharpe=sharpe,
        n_buys=sum(1 for t in trades if t["side"] == "buy"),
        n_sells=len(sells),
        win_rate_pct=win_rate,
        trades=trades,
        equity_curve=eq,
    )


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------

SCENARIOS = [
    # (name, days, mu, sigma, start, seed)
    ("bull_btc_like", 800, 0.60, 0.70, 30_000, 7),
    ("bear_btc_like", 800, -0.40, 0.80, 60_000, 11),
    ("sideways_choppy", 800, 0.00, 0.65, 40_000, 23),
    ("high_vol_alt", 800, 0.30, 1.10, 100, 42),
]


def make_sentiment_stream(prices: pd.DataFrame, seed: int) -> pd.Series:
    """Sentiment that loosely tracks 7-day momentum plus noise, clipped to [-1, 1].

    Real sentiment-from-social tends to lag price (mood follows recent moves)
    with substantial noise. This is a rough but defensible proxy.
    """
    rng = np.random.default_rng(seed + 1000)
    momentum = prices["price"].pct_change().rolling(7).mean().fillna(0)
    momentum = momentum / (momentum.std() + 1e-9)  # standardize
    noise = rng.standard_normal(len(prices)) * 0.4
    raw = momentum.values * 0.6 + noise
    return pd.Series(np.clip(raw, -1, 1), index=prices.index)


def diag_summary(d: dict) -> dict:
    bars = max(d["bars"], 1)
    scores = np.asarray(d["scores"])
    return {
        "ma_buy_days": d["ma_buy"],
        "ma_sell_days": d["ma_sell"],
        "rsi_oversold_days": d["rsi_oversold"],
        "rsi_overbought_days": d["rsi_overbought"],
        "ai_up_days": d["ai_up"],
        "ai_down_days": d["ai_down"],
        "score_mean": round(float(scores.mean()), 4),
        "score_std": round(float(scores.std()), 4),
        "score_max": round(float(scores.max()), 4),
        "score_min": round(float(scores.min()), 4),
        "pct_score_gt_0.5": round(float((scores > 0.5).mean() * 100), 2),
        "pct_score_lt_-0.5": round(float((scores < -0.5).mean() * 100), 2),
    }


def main() -> None:
    out_dir = Path(__file__).parent / "backtest_output"
    out_dir.mkdir(exist_ok=True)

    summary_rows = []
    diagnostics_all = {}

    print("=" * 95)
    print("RUN A — sentiment & on-chain neutralized (offline backtest baseline)")
    print("=" * 95)
    for name, n, mu, sigma, start, seed in SCENARIOS:
        prices = generate_prices(n, mu, sigma, start, seed)
        prices.to_csv(out_dir / f"prices_{name}.csv")

        diag: dict = {}
        result = run_backtest(name, prices, diagnostics=diag)
        result.equity_curve.to_csv(out_dir / f"equity_{name}.csv")
        with (out_dir / f"trades_{name}.json").open("w") as f:
            json.dump(result.trades, f, indent=2)

        ds = diag_summary(diag)
        diagnostics_all[name] = ds

        row = {
            "scenario": name,
            "mode": "neutral_sentiment",
            "buy_hold_%": round(result.buy_hold_return_pct, 2),
            "bot_%": round(result.total_return_pct, 2),
            "alpha_%": round(result.total_return_pct - result.buy_hold_return_pct, 2),
            "max_dd_%": round(result.max_drawdown_pct, 2),
            "sharpe": round(result.sharpe, 2),
            "buys": result.n_buys,
            "sells": result.n_sells,
            "win_rate_%": round(result.win_rate_pct, 1),
        }
        summary_rows.append(row)
        print(f"{name:18s}  bot={row['bot_%']:7.2f}%  bh={row['buy_hold_%']:7.2f}%  "
              f"alpha={row['alpha_%']:7.2f}%  dd={row['max_dd_%']:6.2f}%  "
              f"sharpe={row['sharpe']:5.2f}  trades={row['buys']}/{row['sells']}  "
              f"wr={row['win_rate_%']:5.1f}%  | score>0.5: {ds['pct_score_gt_0.5']:5.2f}%  "
              f"score<-0.5: {ds['pct_score_lt_-0.5']:5.2f}%")

    print()
    print("=" * 95)
    print("RUN B — synthetic sentiment stream (momentum-correlated, noisy)")
    print("=" * 95)
    for name, n, mu, sigma, start, seed in SCENARIOS:
        prices = generate_prices(n, mu, sigma, start, seed)
        sent = make_sentiment_stream(prices, seed)
        diag: dict = {}
        result = run_backtest(f"{name}_sent", prices, sentiment_stream=sent, diagnostics=diag)

        ds = diag_summary(diag)
        diagnostics_all[f"{name}_sent"] = ds

        row = {
            "scenario": name,
            "mode": "with_sentiment",
            "buy_hold_%": round(result.buy_hold_return_pct, 2),
            "bot_%": round(result.total_return_pct, 2),
            "alpha_%": round(result.total_return_pct - result.buy_hold_return_pct, 2),
            "max_dd_%": round(result.max_drawdown_pct, 2),
            "sharpe": round(result.sharpe, 2),
            "buys": result.n_buys,
            "sells": result.n_sells,
            "win_rate_%": round(result.win_rate_pct, 1),
        }
        summary_rows.append(row)
        print(f"{name:18s}  bot={row['bot_%']:7.2f}%  bh={row['buy_hold_%']:7.2f}%  "
              f"alpha={row['alpha_%']:7.2f}%  dd={row['max_dd_%']:6.2f}%  "
              f"sharpe={row['sharpe']:5.2f}  trades={row['buys']}/{row['sells']}  "
              f"wr={row['win_rate_%']:5.1f}%  | score>0.5: {ds['pct_score_gt_0.5']:5.2f}%  "
              f"score<-0.5: {ds['pct_score_lt_-0.5']:5.2f}%")

    print()
    print("=" * 95)
    print("RUN C — threshold sensitivity sweep on bull_btc_like (with sentiment)")
    print("=" * 95)
    sweep_rows = []
    name, n, mu, sigma, start, seed = SCENARIOS[0]
    prices = generate_prices(n, mu, sigma, start, seed)
    sent = make_sentiment_stream(prices, seed)
    for thr in [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]:
        result = run_backtest(
            f"thr_{thr}", prices, sentiment_stream=sent,
            buy_threshold=thr, sell_threshold=-thr,
        )
        sweep_rows.append({
            "buy_thr": thr,
            "buy_hold_%": round(result.buy_hold_return_pct, 2),
            "bot_%": round(result.total_return_pct, 2),
            "alpha_%": round(result.total_return_pct - result.buy_hold_return_pct, 2),
            "max_dd_%": round(result.max_drawdown_pct, 2),
            "sharpe": round(result.sharpe, 2),
            "buys": result.n_buys,
            "sells": result.n_sells,
            "win_rate_%": round(result.win_rate_pct, 1),
        })
        print(f"  threshold ±{thr:0.2f}  bot={sweep_rows[-1]['bot_%']:7.2f}%  "
              f"alpha={sweep_rows[-1]['alpha_%']:7.2f}%  dd={sweep_rows[-1]['max_dd_%']:6.2f}%  "
              f"sharpe={sweep_rows[-1]['sharpe']:5.2f}  trades={result.n_buys}/{result.n_sells}  "
              f"wr={sweep_rows[-1]['win_rate_%']:5.1f}%")

    print()
    print("=" * 95)
    print("Signal-firing diagnostics (out of ~600 evaluated bars per scenario)")
    print("=" * 95)
    for k, v in diagnostics_all.items():
        print(f"  {k:30s}  ma_buy={v['ma_buy_days']:3d}  ma_sell={v['ma_sell_days']:3d}  "
              f"rsi_os={v['rsi_oversold_days']:3d}  rsi_ob={v['rsi_overbought_days']:3d}  "
              f"score(min/mean/max)={v['score_min']:+.2f}/{v['score_mean']:+.2f}/{v['score_max']:+.2f}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(sweep_rows).to_csv(out_dir / "threshold_sweep.csv", index=False)
    with (out_dir / "diagnostics.json").open("w") as f:
        json.dump(diagnostics_all, f, indent=2)

    print()
    print("Summary table written to backtest_output/summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
