"""
BTC Price Predictor — FastAPI server
=====================================
Exposes live BTC directional predictions as a REST API.

Run:
    uvicorn api_server:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health                            — server health + model status + CV stats
    GET  /predict/both                      — predictions for 5 min AND 15 min
    GET  /predict/{5|15}                    — prediction for one horizon
    GET  /backtest/{5|15}                   — backtest with optimised threshold
    GET  /optimize-threshold/{5|15}         — find & apply optimal threshold
    POST /retrain/{5|15}                    — retrain + re-optimise threshold
"""

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from btc_predictor import BTCPredictor

RETRAIN_INTERVAL_SECONDS = 3600  # auto-retrain + re-optimise every hour

predictor   = BTCPredictor()
_ready      = {5: False, 15: False}
_trained_at = {5: None,  15: None}


def _train(h: int) -> None:
    """Train model for horizon h (includes automatic threshold optimisation)."""
    predictor.train(h)
    _ready[h]      = True
    _trained_at[h] = datetime.now(timezone.utc).isoformat()
    print(f"[Server] {h}min model ready — {_trained_at[h]}")


def _background_retrain() -> None:
    while True:
        time.sleep(RETRAIN_INTERVAL_SECONDS)
        for h in [5, 15]:
            try:
                _train(h)
            except Exception as e:
                print(f"[Server] Retrain failed ({h}min): {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    for h in [5, 15]:
        await loop.run_in_executor(None, _train, h)
    threading.Thread(target=_background_retrain, daemon=True).start()
    yield


app = FastAPI(
    title="BTC Price Predictor API",
    description=(
        "Predicts BTC/USDT price direction (UP / DOWN / UNCERTAIN) for 5-min and "
        "15-min horizons using a LightGBM + XGBoost ensemble. "
        "The confidence threshold is auto-optimised after each training run to "
        "achieve ≥80% accuracy while maximising the fraction of candles that "
        "receive a directional call (coverage). "
        "Data source: Binance public API — no API key required."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Server health, model readiness and CV stats")
async def health():
    return {
        "status"            : "ok",
        "models_ready"      : _ready,
        "trained_at"        : _trained_at,
        "active_thresholds" : {
            str(h): round(predictor.thresholds.get(h, predictor._default_threshold) * 100, 2)
            for h in [5, 15]
        },
        "cv_stats": {str(h): predictor.cv_stats.get(h) for h in [5, 15]},
    }


@app.get("/predict/both", summary="Predictions for both 5min and 15min horizons")
async def predict_both():
    if not all(_ready.values()):
        raise HTTPException(503, detail="Models are still loading. Retry in a moment.")
    try:
        p5  = predictor.predict(5)
        p15 = predictor.predict(15)
        return {
            "5min"    : p5,
            "15min"   : p15,
            "cv_stats": {str(h): predictor.cv_stats.get(h) for h in [5, 15]},
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/predict/{horizon}", summary="Prediction for a specific horizon (5 or 15 minutes)")
async def predict(horizon: int):
    if horizon not in (5, 15):
        raise HTTPException(400, detail="horizon must be 5 or 15")
    if not _ready[horizon]:
        raise HTTPException(503, detail=f"{horizon}min model is not ready yet.")
    try:
        result = predictor.predict(horizon)
        result["cv_stats"]   = predictor.cv_stats.get(horizon)
        result["trained_at"] = _trained_at[horizon]
        return result
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/backtest/{horizon}", summary="Backtest accuracy and threshold curve")
async def backtest(
    horizon: int,
    recent_n: int = Query(
        default=400, ge=50, le=1000,
        description="Number of recent 1-min candles to evaluate on",
    ),
):
    if horizon not in (5, 15):
        raise HTTPException(400, detail="horizon must be 5 or 15")
    if not _ready[horizon]:
        raise HTTPException(503, detail=f"{horizon}min model is not ready yet.")
    try:
        return predictor.backtest(horizon, recent_n=recent_n)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get(
    "/optimize-threshold/{horizon}",
    summary="Find the lowest threshold that hits target accuracy, maximising coverage",
)
async def optimize_threshold(
    horizon: int,
    target_accuracy: float = Query(
        default=0.80, ge=0.50, le=0.99,
        description="Minimum acceptable accuracy (0.80 = 80%)",
    ),
    recent_n: int = Query(
        default=400, ge=50, le=1000,
        description="Number of recent candles to calibrate on",
    ),
):
    """
    Scans confidence thresholds from 50% to 95% in 0.5% steps.
    Picks the lowest threshold that achieves `target_accuracy` on recent hold-out
    data — maximising the fraction of candles that get a directional prediction.
    The selected threshold is applied immediately and persisted to disk.
    """
    if horizon not in (5, 15):
        raise HTTPException(400, detail="horizon must be 5 or 15")
    if not _ready[horizon]:
        raise HTTPException(503, detail=f"{horizon}min model is not ready yet.")
    try:
        return predictor.optimize_threshold(
            horizon,
            target_accuracy=target_accuracy,
            recent_n=recent_n,
        )
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.post("/retrain/{horizon}", summary="Retrain model and re-optimise threshold")
async def retrain(horizon: int):
    if horizon not in (5, 15):
        raise HTTPException(400, detail="horizon must be 5 or 15")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _train, horizon)
    return {
        "status"            : "success",
        "horizon"           : horizon,
        "trained_at"        : _trained_at[horizon],
        "optimised_threshold_pct": round(
            predictor.thresholds.get(horizon, predictor._default_threshold) * 100, 2
        ),
        "cv_stats": predictor.cv_stats.get(horizon),
    }
