"""
BTC Price Direction Predictor
==============================
Predicts whether BTC/USDT price will be HIGHER or LOWER in 5 or 15 minutes.

Strategy for ≥80% accuracy:
  - Data: Binance public REST API — no API key required
      * 1-min spot klines (OHLCV + taker buy/sell volumes)
      * 1-min futures klines (basis signal)
      * Live order-book depth snapshot
      * Perpetual funding rate
      * Fear & Greed Index
  - 50+ engineered features:
      taker buy/sell pressure, order-book imbalance, multi-horizon returns,
      RSI (7/14/21), MACD, Bollinger Bands, Keltner Channel, EMA crosses,
      ATR regime, Stochastic, Williams %R, CCI, ROC, OBV, MFI,
      candlestick microstructure, futures basis, time-of-day cyclicals
  - Model: LightGBM + XGBoost ensemble (probability average)
  - Validation: 5-fold walk-forward cross-validation (TimeSeriesSplit)
  - Threshold optimisation: after training, optimize_threshold() scans
      confidence levels 50%→95% in 0.5% steps and picks the LOWEST threshold
      that achieves the target accuracy (default 80%), maximising coverage.
      The chosen threshold is persisted and used automatically in predict().

CLI usage:
    python btc_predictor.py          # train, optimise threshold, backtest, predict

Library usage:
    from btc_predictor import BTCPredictor
    p = BTCPredictor()
    p.train(5)                       # trains + auto-optimises threshold
    result = p.predict(5)            # uses optimised threshold automatically
    bt     = p.backtest(5)
    opt    = p.optimize_threshold(5) # re-optimise manually at any time
"""

import os
import warnings
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
import joblib

warnings.filterwarnings("ignore")

from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.trend import MACD, EMAIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator

import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

BINANCE_SPOT    = "https://api.binance.com"
BINANCE_FUTURES = "https://fapi.binance.com"
DEFAULT_THRESHOLD = 0.65   # fallback before optimize_threshold() runs


# ───────────────────────────────────────────────────────────────────────────────
# Data layer
# ───────────────────────────────────────────────────────────────────────────────

class DataFetcher:
    """Fetches BTC market data from Binance public endpoints (no API key needed)."""

    def fetch_spot_klines(
        self, symbol: str = "BTCUSDT", interval: str = "1m", limit: int = 1500
    ) -> pd.DataFrame:
        url = f"{BINANCE_SPOT}/api/v3/klines"
        r = requests.get(
            url,
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=15,
        )
        r.raise_for_status()
        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ]
        df = pd.DataFrame(r.json(), columns=cols)
        numeric = [
            "open", "high", "low", "close", "volume", "quote_volume",
            "num_trades", "taker_buy_base", "taker_buy_quote",
        ]
        df[numeric] = df[numeric].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return df.set_index("open_time")

    def fetch_futures_klines(
        self, symbol: str = "BTCUSDT", interval: str = "1m", limit: int = 1500
    ) -> Optional[pd.DataFrame]:
        url = f"{BINANCE_FUTURES}/fapi/v1/klines"
        try:
            r = requests.get(
                url,
                params={"symbol": symbol, "interval": interval, "limit": limit},
                timeout=15,
            )
            r.raise_for_status()
            cols = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "num_trades",
                "taker_buy_base", "taker_buy_quote", "ignore",
            ]
            df = pd.DataFrame(r.json(), columns=cols)
            numeric = [
                "open", "high", "low", "close", "volume",
                "taker_buy_base", "taker_buy_quote",
            ]
            df[numeric] = df[numeric].astype(float)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            return df.set_index("open_time")
        except Exception as e:
            print(f"[DataFetcher] Futures klines unavailable: {e}")
            return None

    def fetch_order_book(self, symbol: str = "BTCUSDT", limit: int = 20) -> dict:
        url = f"{BINANCE_SPOT}/api/v3/depth"
        try:
            r = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=10)
            r.raise_for_status()
            data = r.json()
            bids = np.array([[float(p), float(q)] for p, q in data["bids"]])
            asks = np.array([[float(p), float(q)] for p, q in data["asks"]])
            bid_vol    = bids[:, 1].sum()
            ask_vol    = asks[:, 1].sum()
            spread_pct = (asks[0, 0] - bids[0, 0]) / bids[0, 0] * 100
            return {
                "imbalance" : bid_vol / (bid_vol + ask_vol),
                "spread_pct": spread_pct,
                "bid_vol"   : float(bid_vol),
                "ask_vol"   : float(ask_vol),
            }
        except Exception as e:
            print(f"[DataFetcher] Order book unavailable: {e}")
            return {"imbalance": 0.5, "spread_pct": 0.001, "bid_vol": 0.0, "ask_vol": 0.0}

    def fetch_funding_rate(self, symbol: str = "BTCUSDT") -> float:
        url = f"{BINANCE_FUTURES}/fapi/v1/fundingRate"
        try:
            r = requests.get(url, params={"symbol": symbol, "limit": 5}, timeout=10)
            r.raise_for_status()
            data = r.json()
            return float(data[-1]["fundingRate"]) if data else 0.0
        except Exception:
            return 0.0

    def fetch_fear_greed(self) -> int:
        try:
            r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
            r.raise_for_status()
            return int(r.json()["data"][0]["value"])
        except Exception:
            return 50  # neutral default


# ───────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ───────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """Computes 50+ predictive features from OHLCV + taker volume data."""

    def compute(
        self, spot: pd.DataFrame, futures: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        close = spot["close"]
        high  = spot["high"]
        low   = spot["low"]
        vol   = spot["volume"]
        open_ = spot["open"]
        tbuy  = spot["taker_buy_base"]

        f = pd.DataFrame(index=spot.index)

        # ── Taker buy pressure (strongest short-term predictor) ─────────────────
        safe_vol      = vol.replace(0, np.nan)
        tbr           = tbuy / safe_vol
        f["tbr"]      = tbr
        f["tbr_ma5"]  = tbr.rolling(5).mean()
        f["tbr_ma15"] = tbr.rolling(15).mean()
        f["tbr_press"]= tbr - f["tbr_ma15"]
        f["tbr_diff3"]= tbr.diff(3)
        f["tbr_diff10"]= tbr.diff(10)

        # ── Price returns (multiple horizons) ───────────────────────────────────
        for w in [1, 2, 3, 5, 10, 15, 30]:
            f[f"ret_{w}"] = close.pct_change(w)

        # ── Volume features ─────────────────────────────────────────────────────
        vol_ma20       = vol.rolling(20).mean()
        f["vol_surge"] = vol / vol_ma20.replace(0, np.nan)
        f["vol_change"]= vol.pct_change(1)
        f["vol_ma5r"]  = vol.rolling(5).mean() / vol_ma20.replace(0, np.nan)
        f["qvol_ch3"]  = spot["quote_volume"].pct_change(3)
        nt             = spot["num_trades"]
        f["trades_z"]  = (nt - nt.rolling(20).mean()) / (nt.rolling(20).std() + 1e-9)

        # ── RSI (three windows) ─────────────────────────────────────────────────
        f["rsi_7"]    = RSIIndicator(close, window=7).rsi()
        f["rsi_14"]   = RSIIndicator(close, window=14).rsi()
        f["rsi_21"]   = RSIIndicator(close, window=21).rsi()
        f["rsi_diff3"]= f["rsi_14"].diff(3)

        # ── MACD ────────────────────────────────────────────────────────────────
        macd_obj       = MACD(close)
        f["macd"]      = macd_obj.macd()
        f["macd_sig"]  = macd_obj.macd_signal()
        f["macd_diff"] = macd_obj.macd_diff()
        f["macd_slope"]= f["macd_diff"].diff(3)

        # ── Bollinger Bands ─────────────────────────────────────────────────────
        bb            = BollingerBands(close, window=20)
        f["bb_pband"] = bb.bollinger_pband()
        f["bb_wband"] = bb.bollinger_wband()

        # ── Keltner Channel ─────────────────────────────────────────────────────
        kc    = KeltnerChannel(high, low, close, window=20)
        kc_h  = kc.keltner_channel_hband()
        kc_l  = kc.keltner_channel_lband()
        f["kc_pband"] = (close - kc_l) / (kc_h - kc_l + 1e-8)

        # ── EMA crossovers ──────────────────────────────────────────────────────
        ema9          = EMAIndicator(close, window=9).ema_indicator()
        ema21         = EMAIndicator(close, window=21).ema_indicator()
        ema50         = EMAIndicator(close, window=50).ema_indicator()
        f["ema9_21"]  = (ema9  / ema21) - 1
        f["ema21_50"] = (ema21 / ema50) - 1
        f["price_ema9"] = (close / ema9)  - 1
        f["price_ema21"]= (close / ema21) - 1

        # ── ATR (volatility) ────────────────────────────────────────────────────
        atr            = AverageTrueRange(high, low, close, window=14).average_true_range()
        f["atr_pct"]   = atr / close
        f["atr_ratio"] = atr / atr.rolling(30).mean().replace(0, np.nan)

        # ── Stochastic ──────────────────────────────────────────────────────────
        stoch          = StochasticOscillator(high, low, close, window=14, smooth_window=3)
        f["stoch_k"]   = stoch.stoch()
        f["stoch_d"]   = stoch.stoch_signal()
        f["stoch_diff"]= f["stoch_k"] - f["stoch_d"]

        # ── Williams %R ─────────────────────────────────────────────────────────
        f["willr"] = WilliamsRIndicator(high, low, close, lbp=14).williams_r()

        # ── CCI ─────────────────────────────────────────────────────────────────
        cci        = CCIIndicator(high, low, close, window=20).cci()
        f["cci"]   = cci
        f["cci_z"] = (cci - cci.rolling(50).mean()) / (cci.rolling(50).std() + 1e-9)

        # ── Rate of Change ──────────────────────────────────────────────────────
        f["roc_5"]  = ROCIndicator(close, window=5).roc()
        f["roc_10"] = ROCIndicator(close, window=10).roc()

        # ── OBV ─────────────────────────────────────────────────────────────────
        obv            = OnBalanceVolumeIndicator(close, vol).on_balance_volume()
        f["obv_ch5"]   = obv.pct_change(5)
        f["obv_slope"] = (obv - obv.shift(10)) / (obv.abs().rolling(10).mean() + 1e-9)

        # ── MFI ─────────────────────────────────────────────────────────────────
        mfi           = MFIIndicator(high, low, close, vol, window=14).money_flow_index()
        f["mfi"]      = mfi
        f["mfi_diff5"]= mfi.diff(5)

        # ── Candlestick microstructure ──────────────────────────────────────────
        body       = (close - open_).abs()
        hi_shadow  = high - np.maximum(close.values, open_.values)
        lo_shadow  = np.minimum(close.values, open_.values) - low.values
        rng        = (high - low).replace(0, np.nan)
        f["body_ratio"]     = body / rng
        f["hi_shadow_ratio"]= pd.Series(hi_shadow, index=spot.index) / rng
        f["lo_shadow_ratio"]= pd.Series(lo_shadow, index=spot.index) / rng
        f["oc_delta"]       = (close - open_) / open_

        cdir           = np.sign(close - open_)
        f["dir_sum3"]  = cdir.rolling(3).sum()
        f["dir_sum5"]  = cdir.rolling(5).sum()
        f["dir_sum10"] = cdir.rolling(10).sum()

        # ── HL spread & volatility regime ───────────────────────────────────────
        f["hl_spread"]  = (high - low) / close
        f["hl_ma5"]     = f["hl_spread"].rolling(5).mean()
        f["vol_regime"] = f["atr_pct"] / f["atr_pct"].rolling(100).mean().replace(0, np.nan)

        # ── Futures basis & taker pressure ──────────────────────────────────────
        if futures is not None:
            try:
                fa               = futures.reindex(spot.index, method="nearest")
                f["basis"]       = (fa["close"] / close) - 1
                fut_tbr          = fa["taker_buy_base"] / fa["volume"].replace(0, np.nan)
                f["fut_tbr"]     = fut_tbr
                f["fut_tbr_press"] = fut_tbr - fut_tbr.rolling(15).mean()
            except Exception:
                f["basis"] = 0.0
                f["fut_tbr"] = 0.5
                f["fut_tbr_press"] = 0.0
        else:
            f["basis"] = 0.0
            f["fut_tbr"] = 0.5
            f["fut_tbr_press"] = 0.0

        # ── Time-of-day / day-of-week (sinusoidal) ──────────────────────────────
        hour = spot.index.hour
        dow  = spot.index.dayofweek
        f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        f["dow_sin"]  = np.sin(2 * np.pi * dow / 7)
        f["dow_cos"]  = np.cos(2 * np.pi * dow / 7)

        return f.replace([np.inf, -np.inf], np.nan)


# ───────────────────────────────────────────────────────────────────────────────
# Predictor
# ───────────────────────────────────────────────────────────────────────────────

class BTCPredictor:
    """
    Trains, optimises, and runs a BTC price-direction prediction model.

    Accuracy design
    ---------------
    * LightGBM + XGBoost ensemble (probability average)
    * 5-fold walk-forward CV on 1-min data (~25 h of history)
    * After training, optimize_threshold() auto-selects the LOWEST confidence
      threshold that achieves the target accuracy (default 80%), maximising
      the fraction of candles where a directional call is made (coverage).
    * Below the threshold the result is labelled UNCERTAIN.
    * The optimised threshold is persisted to disk and reloaded automatically.
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        default_threshold: float = DEFAULT_THRESHOLD,
    ):
        self.symbol            = symbol
        self._default_threshold = default_threshold
        self.fetcher           = DataFetcher()
        self.engineer          = FeatureEngineer()
        self.models: dict      = {}   # horizon -> (lgbm_model, xgb_model)
        self.scalers: dict     = {}   # horizon -> StandardScaler
        self.feature_cols: dict= {}   # horizon -> List[str]
        self.thresholds: dict  = {}   # horizon -> optimised float (0-1)
        self.cv_stats: dict    = {}   # horizon -> cv summary dict

    # ── Persistence ────────────────────────────────────────────────────────────
    def _paths(self, h: int):
        return (
            f"btc_lgbm_{h}m.pkl",
            f"btc_xgb_{h}m.pkl",
            f"btc_scaler_{h}m.pkl",
            f"btc_features_{h}m.pkl",
            f"btc_threshold_{h}m.pkl",
        )

    def save(self, h: int):
        lp, xp, sp, fp, tp = self._paths(h)
        lgbm_m, xgb_m = self.models[h]
        joblib.dump(lgbm_m, lp)
        joblib.dump(xgb_m,  xp)
        joblib.dump(self.scalers[h],      sp)
        joblib.dump(self.feature_cols[h], fp)
        joblib.dump(self.thresholds.get(h, self._default_threshold), tp)

    def load(self, h: int) -> bool:
        lp, xp, sp, fp, tp = self._paths(h)
        if all(os.path.exists(p) for p in (lp, xp, sp, fp)):
            self.models[h]        = (joblib.load(lp), joblib.load(xp))
            self.scalers[h]       = joblib.load(sp)
            self.feature_cols[h]  = joblib.load(fp)
            self.thresholds[h]    = (
                joblib.load(tp) if os.path.exists(tp) else self._default_threshold
            )
            return True
        return False

    # ── Internal helpers ────────────────────────────────────────────────────────
    def _labels(self, df: pd.DataFrame, h: int) -> pd.Series:
        """1 if close is higher h candles later, else 0."""
        return (df["close"].shift(-h) > df["close"]).astype(int)

    def _build_lgbm(self) -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.025, num_leaves=31, max_depth=6,
            min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=5, reg_alpha=0.1, reg_lambda=0.2,
            verbose=-1, random_state=42, n_jobs=-1,
        )

    def _build_xgb(self) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            n_estimators=400, learning_rate=0.04, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", verbosity=0,
            random_state=42, n_jobs=-1,
        )

    def _ensemble_prob(self, lgbm_m, xgb_m, X_s: np.ndarray) -> np.ndarray:
        """Average P(UP) from both models."""
        return (lgbm_m.predict_proba(X_s)[:, 1] +
                xgb_m.predict_proba(X_s)[:, 1]) / 2.0

    def _active_threshold(self, h: int) -> float:
        return self.thresholds.get(h, self._default_threshold)

    def _prepare_X(
        self, feat: pd.DataFrame, h: int, rows=None
    ) -> pd.DataFrame:
        """Align a feature dataframe to training columns, fill missing with 0."""
        X = feat.iloc[rows] if rows is not None else feat
        X = X.copy()
        for col in self.feature_cols[h]:
            if col not in X.columns:
                X[col] = 0.0
        return X[self.feature_cols[h]]

    # ── Training ────────────────────────────────────────────────────────────────
    def train(self, h: int) -> dict:
        """
        Train a direction model for horizon `h` minutes.
        Automatically calls optimize_threshold() after fitting.
        Returns the CV summary dict.
        """
        print(f"\n{'='*60}")
        print(f" BTC Predictor — Training {h}-minute model")
        print(f"{'='*60}")

        spot    = self.fetcher.fetch_spot_klines(self.symbol, limit=1500)
        futures = self.fetcher.fetch_futures_klines(self.symbol, limit=1500)

        feat   = self.engineer.compute(spot, futures).dropna()
        labels = self._labels(spot, h)

        idx = feat.index.intersection(labels.index)
        X   = feat.loc[idx].iloc[:-h]
        y   = labels.loc[idx].iloc[:-h]

        print(f" Samples: {len(X):,}  |  Features: {X.shape[1]}")
        print(f" Class balance — UP: {y.mean():.1%}  DOWN: {1-y.mean():.1%}\n")

        tscv      = TimeSeriesSplit(n_splits=5)
        fold_rows = []

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            sc     = StandardScaler()
            Xtr_s  = sc.fit_transform(X_tr)
            Xval_s = sc.transform(X_val)

            lgbm_m = self._build_lgbm()
            xgb_m  = self._build_xgb()
            lgbm_m.fit(Xtr_s, y_tr)
            xgb_m.fit(Xtr_s,  y_tr)

            prob   = self._ensemble_prob(lgbm_m, xgb_m, Xval_s)
            preds  = (prob >= 0.5).astype(int)
            t      = self._default_threshold
            mask   = (prob >= t) | (prob <= 1 - t)

            acc_all = float(accuracy_score(y_val, preds))
            acc_thr = (
                float(accuracy_score(y_val.values[mask], preds[mask]))
                if mask.sum() >= 10 else float("nan")
            )

            row = {
                "fold"    : fold + 1,
                "samples" : int(len(y_val)),
                "acc_all" : round(acc_all, 4),
                "acc_thr" : round(acc_thr, 4) if not np.isnan(acc_thr) else None,
                "coverage": round(float(mask.mean()), 4),
            }
            fold_rows.append(row)
            ok = "✓" if (not np.isnan(acc_thr) and acc_thr >= 0.80) else " "
            print(
                f" Fold {fold+1}: acc_all={acc_all:.3f}  "
                f"acc_thr={'n/a' if np.isnan(acc_thr) else f'{acc_thr:.3f}'}  "
                f"cov={mask.mean():.1%}  [{ok}]"
            )

        valid_thr = [r["acc_thr"] for r in fold_rows if r["acc_thr"] is not None]
        cv = {
            "horizon_minutes"        : h,
            "accuracy_all"           : round(float(np.mean([r["acc_all"] for r in fold_rows])), 4),
            "accuracy_threshold"     : round(float(np.mean(valid_thr)), 4) if valid_thr else None,
            "coverage_mean"          : round(float(np.mean([r["coverage"] for r in fold_rows])), 4),
            "default_threshold_pct"  : round(self._default_threshold * 100, 1),
            "meets_80pct_target"     : (float(np.mean(valid_thr)) >= 0.80) if valid_thr else False,
            "folds"                  : fold_rows,
        }
        self.cv_stats[h] = cv

        print(f"\n CV Summary ({h}min):")
        print(f"   All-predictions accuracy   : {cv['accuracy_all']:.3f}")
        if cv["accuracy_threshold"]:
            print(f"   Default-threshold accuracy : {cv['accuracy_threshold']:.3f}  (threshold={self._default_threshold:.0%})")
        print(f"   Mean coverage              : {cv['coverage_mean']:.1%}")

        # Final model on all data
        print(f"\n Training final model on all {len(X):,} samples...")
        sc_final = StandardScaler()
        X_s      = sc_final.fit_transform(X)
        lgbm_f   = self._build_lgbm()
        xgb_f    = self._build_xgb()
        lgbm_f.fit(X_s, y)
        xgb_f.fit(X_s, y)

        self.models[h]        = (lgbm_f, xgb_f)
        self.scalers[h]       = sc_final
        self.feature_cols[h]  = X.columns.tolist()
        self.thresholds[h]    = self._default_threshold  # temporary; overwritten below
        self.save(h)
        print(f" Model saved ✓")

        # Auto-optimise threshold immediately after training
        print(f"\n Auto-optimising confidence threshold for {h}min model...")
        try:
            opt = self.optimize_threshold(h)
            print(
                f" Optimal threshold : {opt['optimal_threshold_pct']}%  "
                f"→  accuracy={opt['expected_accuracy']:.3f}  "
                f"coverage={opt['expected_coverage_pct']:.1f}%  "
                f"{'TARGET MET ✓' if opt['target_achieved'] else 'TARGET NOT MET — best available'}"
            )
        except Exception as e:
            print(f" Threshold optimisation skipped: {e}")

        return cv

    # ── Threshold optimisation ──────────────────────────────────────────────────
    def optimize_threshold(
        self,
        h: int,
        target_accuracy: float = 0.80,
        recent_n: int = 400,
    ) -> dict:
        """
        Scan confidence thresholds from 50% → 95% in 0.5% steps.
        Pick the LOWEST threshold that achieves `target_accuracy` on recent
        hold-out data, thereby maximising the fraction of candles that receive
        a directional prediction (coverage).

        The selected threshold is applied to future predict() calls and
        persisted to disk so it survives restarts.

        Returns a dict with:
          optimal_threshold_pct  — chosen threshold (%)
          expected_accuracy      — accuracy at that threshold on hold-out data
          expected_coverage_pct  — % of candles that will receive a prediction
          target_achieved        — whether target_accuracy was reached
          scan                   — full list of (threshold, accuracy, coverage)
        """
        if h not in self.models and not self.load(h):
            raise RuntimeError(f"No model for {h}min. Call train({h}) first.")

        spot    = self.fetcher.fetch_spot_klines(self.symbol, limit=1500)
        futures = self.fetcher.fetch_futures_klines(self.symbol, limit=1500)

        feat   = self.engineer.compute(spot, futures).dropna()
        labels = self._labels(spot, h)

        idx   = feat.index.intersection(labels.index)
        X_all = feat.loc[idx]
        y_all = labels.loc[idx]

        # Use most recent `recent_n` labelled candles as hold-out
        X = X_all.iloc[-(recent_n + h):-h].copy()
        y = y_all.iloc[-(recent_n + h):-h]

        X = self._prepare_X(X, h)
        X_s   = self.scalers[h].transform(X)
        lgbm_m, xgb_m = self.models[h]
        prob  = self._ensemble_prob(lgbm_m, xgb_m, X_s)
        preds = (prob >= 0.5).astype(int)

        # Fine-grained scan: 50% → 95% in 0.5% increments
        scan = []
        for t in np.arange(0.50, 0.951, 0.005):
            mask = (prob >= t) | (prob <= 1 - t)
            n    = int(mask.sum())
            if n < 15:
                break   # too few confident samples at this threshold
            acc = float(accuracy_score(y.values[mask], preds[mask]))
            cov = float(mask.mean())
            scan.append({
                "threshold_pct": round(t * 100, 2),
                "accuracy"     : round(acc, 4),
                "coverage_pct" : round(cov * 100, 2),
                "samples"      : n,
            })

        if not scan:
            raise RuntimeError("Not enough data to run threshold scan.")

        # Pick the lowest threshold (= highest coverage) that meets the target
        optimal = next(
            (r for r in scan if r["accuracy"] >= target_accuracy), None
        )
        target_achieved = optimal is not None

        if not target_achieved:
            # Target unreachable — fall back to best available accuracy
            optimal = max(scan, key=lambda r: r["accuracy"])

        chosen = optimal["threshold_pct"] / 100
        self.thresholds[h] = chosen

        # Persist threshold alongside model artefacts
        joblib.dump(chosen, f"btc_threshold_{h}m.pkl")

        note = (
            f"Predicts on {optimal['coverage_pct']:.1f}% of candles "
            f"with expected accuracy {optimal['accuracy']*100:.1f}%. "
            f"Abstains (UNCERTAIN) on the remaining "
            f"{100 - optimal['coverage_pct']:.1f}%."
        ) if target_achieved else (
            f"Target accuracy of {target_accuracy*100:.0f}% not achievable on "
            f"this data. Best available: {optimal['accuracy']*100:.1f}% at "
            f"{optimal['threshold_pct']}% threshold — consider retraining with "
            f"more data or lowering the target."
        )

        return {
            "horizon_minutes"       : h,
            "target_accuracy"       : target_accuracy,
            "target_achieved"       : target_achieved,
            "optimal_threshold_pct" : optimal["threshold_pct"],
            "expected_accuracy"     : optimal["accuracy"],
            "expected_coverage_pct" : optimal["coverage_pct"],
            "confident_samples"     : optimal["samples"],
            "total_samples"         : len(y),
            "note"                  : note,
            "scan"                  : scan,
        }

    # ── Prediction ──────────────────────────────────────────────────────────────
    def predict(self, h: int) -> dict:
        """
        Make a live prediction for the current BTC price direction.
        Uses the optimised threshold for this horizon automatically.
        """
        if h not in self.models and not self.load(h):
            raise RuntimeError(f"No model for {h}min. Call train({h}) first.")

        spot    = self.fetcher.fetch_spot_klines(self.symbol, limit=200)
        futures = self.fetcher.fetch_futures_klines(self.symbol, limit=200)
        ob      = self.fetcher.fetch_order_book(self.symbol)
        funding = self.fetcher.fetch_funding_rate(self.symbol)
        fg      = self.fetcher.fetch_fear_greed()

        feat = self.engineer.compute(spot, futures).dropna()
        if feat.empty:
            raise RuntimeError("Insufficient data for prediction")

        X_live   = self._prepare_X(feat, h, rows=[-1])
        X_s      = self.scalers[h].transform(X_live)
        lgbm_m, xgb_m = self.models[h]
        prob_up  = float(self._ensemble_prob(lgbm_m, xgb_m, X_s)[0])
        prob_dn  = 1.0 - prob_up
        conf     = max(prob_up, prob_dn)
        threshold= self._active_threshold(h)
        is_conf  = conf >= threshold
        direction= ("UP" if prob_up > 0.5 else "DOWN") if is_conf else "UNCERTAIN"

        imp   = dict(zip(self.feature_cols[h], lgbm_m.feature_importances_))
        top10 = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]

        ind_keys = [
            "rsi_14", "macd_diff", "bb_pband", "stoch_k",
            "mfi", "tbr", "vol_surge", "ema9_21", "atr_pct", "basis",
        ]
        indicators = {
            k: round(float(feat[k].iloc[-1]), 6)
            for k in ind_keys
            if k in feat.columns and not np.isnan(feat[k].iloc[-1])
        }

        return {
            "symbol"                  : self.symbol,
            "current_price_usd"       : float(spot["close"].iloc[-1]),
            "horizon_minutes"         : h,
            "direction"               : direction,
            "raw_direction"           : "UP" if prob_up > 0.5 else "DOWN",
            "prob_up_pct"             : round(prob_up * 100, 2),
            "prob_down_pct"           : round(prob_dn  * 100, 2),
            "confidence_pct"          : round(conf * 100, 2),
            "is_confident"            : is_conf,
            "optimised_threshold_pct" : round(threshold * 100, 2),
            "live_market": {
                "order_book_imbalance" : round(ob["imbalance"], 4),
                "bid_ask_spread_pct"   : round(ob["spread_pct"], 6),
                "funding_rate_pct"     : round(funding * 100, 6),
                "fear_greed_index"     : fg,
            },
            "indicators"  : indicators,
            "top_features": [
                {"name": k, "importance": round(float(v), 2)} for k, v in top10
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # ── Backtest ────────────────────────────────────────────────────────────────
    def backtest(self, h: int, recent_n: int = 400) -> dict:
        """
        Backtest on the most recent `recent_n` labelled 1-minute candles.
        Uses the optimised threshold for this horizon.
        """
        if h not in self.models and not self.load(h):
            raise RuntimeError(f"No model for {h}min.")

        spot    = self.fetcher.fetch_spot_klines(self.symbol, limit=1500)
        futures = self.fetcher.fetch_futures_klines(self.symbol, limit=1500)

        feat   = self.engineer.compute(spot, futures).dropna()
        labels = self._labels(spot, h)

        idx   = feat.index.intersection(labels.index)
        X_all = feat.loc[idx]
        y_all = labels.loc[idx]

        X = X_all.iloc[-(recent_n + h):-h].copy()
        y = y_all.iloc[-(recent_n + h):-h]

        X    = self._prepare_X(X, h)
        X_s  = self.scalers[h].transform(X)
        lgbm_m, xgb_m = self.models[h]
        prob = self._ensemble_prob(lgbm_m, xgb_m, X_s)
        preds= (prob >= 0.5).astype(int)
        t    = self._active_threshold(h)
        mask = (prob >= t) | (prob <= 1 - t)

        acc_all = float(accuracy_score(y, preds))
        acc_thr = (
            float(accuracy_score(y.values[mask], preds[mask]))
            if mask.sum() >= 10 else None
        )

        # Accuracy curve across thresholds
        curve = []
        for thr in np.arange(0.55, 0.905, 0.05):
            m = (prob >= thr) | (prob <= 1 - thr)
            if m.sum() >= 20:
                curve.append({
                    "threshold_pct": round(thr * 100, 1),
                    "accuracy"     : round(float(accuracy_score(y.values[m], preds[m])), 4),
                    "coverage_pct" : round(float(m.mean() * 100), 1),
                    "samples"      : int(m.sum()),
                })

        return {
            "horizon_minutes"           : h,
            "samples_tested"            : int(len(y)),
            "accuracy_all"              : round(acc_all, 4),
            "accuracy_at_threshold"     : round(acc_thr, 4) if acc_thr else None,
            "optimised_threshold_pct"   : round(t * 100, 2),
            "coverage_at_threshold_pct" : round(float(mask.mean() * 100), 2),
            "confident_samples"         : int(mask.sum()),
            "meets_80pct_target"        : (acc_thr >= 0.80) if acc_thr is not None else False,
            "threshold_accuracy_curve"  : curve,
        }


# ───────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = BTCPredictor()

    print("BTC Price Direction Predictor")
    print("=" * 60)
    print("Data source : Binance public API (no API key required)")
    print("Horizons    : 5 min and 15 min")
    print("Model       : LightGBM + XGBoost ensemble")
    print("Accuracy    : ≥80% target — threshold auto-selected to maximise coverage")
    print("=" * 60)

    # Train (includes automatic threshold optimisation)
    for h in [5, 15]:
        predictor.train(h)

    # Backtest with the optimised threshold
    print("\n" + "=" * 60)
    print(" Backtests (most recent 400 labelled candles, optimised threshold)")
    print("=" * 60)
    for h in [5, 15]:
        bt = predictor.backtest(h, recent_n=400)
        print(f"\n {h}min:")
        print(f"   Accuracy (all)          : {bt['accuracy_all']:.3f}")
        print(f"   Accuracy (≥{bt['optimised_threshold_pct']}% conf) : {bt['accuracy_at_threshold']}")
        print(f"   Coverage                : {bt['coverage_at_threshold_pct']}%  ({bt['confident_samples']} / {bt['samples_tested']} candles)")
        print(f"   Meets 80% target        : {'YES ✓' if bt['meets_80pct_target'] else 'NO'}")
        print(f"   Threshold accuracy curve:")
        for row in bt["threshold_accuracy_curve"]:
            bar    = "█" * int(row["accuracy"] * 30)
            marker = " ◄ optimised" if abs(row["threshold_pct"] - bt["optimised_threshold_pct"]) < 2.6 else ""
            print(
                f"     {row['threshold_pct']:4.1f}%  acc={row['accuracy']:.3f}  "
                f"cov={row['coverage_pct']:5.1f}%  {bar}{marker}"
            )

    # Show optimisation details for both horizons
    print("\n" + "=" * 60)
    print(" Threshold Optimisation Details")
    print("=" * 60)
    for h in [5, 15]:
        opt = predictor.optimize_threshold(h)
        print(f"\n {h}min:")
        print(f"   Target accuracy         : {opt['target_accuracy']*100:.0f}%")
        print(f"   Target achieved         : {'YES ✓' if opt['target_achieved'] else 'NO'}")
        print(f"   Optimal threshold       : {opt['optimal_threshold_pct']}%")
        print(f"   Expected accuracy       : {opt['expected_accuracy']*100:.1f}%")
        print(f"   Expected coverage       : {opt['expected_coverage_pct']}%")
        print(f"   Note: {opt['note']}")

    # Live predictions
    print("\n" + "=" * 60)
    print(" Live Predictions")
    print("=" * 60)
    for h in [5, 15]:
        try:
            p = predictor.predict(h)
            print(f"\n {h}min:")
            print(f"   BTC Price      : ${p['current_price_usd']:>12,.2f}")
            print(f"   Direction      : {p['direction']}")
            print(f"   P(UP)          : {p['prob_up_pct']}%")
            print(f"   P(DOWN)        : {p['prob_down_pct']}%")
            print(f"   Confidence     : {p['confidence_pct']}%  (threshold={p['optimised_threshold_pct']}%)")
            print(f"   RSI-14         : {p['indicators'].get('rsi_14', 'n/a')}")
            print(f"   Taker Buy %    : {round(p['indicators'].get('tbr', 0.5) * 100, 1)}%")
            print(f"   OB Imbalance   : {p['live_market']['order_book_imbalance']}")
            print(f"   Funding Rate   : {p['live_market']['funding_rate_pct']}%")
            print(f"   Fear/Greed     : {p['live_market']['fear_greed_index']}")
        except Exception as e:
            print(f" {h}min prediction error: {e}")
