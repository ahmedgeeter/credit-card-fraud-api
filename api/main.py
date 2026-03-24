"""FastAPI app entrypoint for fraud inference.

Example (after training and starting the server on port 8000)::

    curl -X POST "http://localhost:8000/predict" \\
      -H "Content-Type: application/json" \\
      -d '{\"Time\": 0.0, \"V1\": -1.359807, \"V2\": -0.072781, \"V3\": 2.536347,
            \"V4\": 1.378155, \"V5\": -0.338321, \"V6\": 0.462388, \"V7\": 0.239599,
            \"V8\": 0.098698, \"V9\": 0.363787, \"V10\": 0.090794, \"V11\": -0.551600,
            \"V12\": -0.617801, \"V13\": -0.991390, \"V14\": -0.311169, \"V15\": 1.468177,
            \"V16\": -0.470401, \"V17\": 0.207971, \"V18\": 0.025791, \"V19\": 0.403993,
            \"V20\": 0.251412, \"V21\": -0.018307, \"V22\": 0.277838, \"V23\": -0.110474,
            \"V24\": 0.066929, \"V25\": 0.128539, \"V26\": -0.189115, \"V27\": 0.133558,
            \"V28\": -0.021053, \"Amount\": 149.62}'
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from starlette.middleware.base import BaseHTTPMiddleware

from api.schemas import (
    FraudPredictResponse,
    TransactionFeatures,
    transaction_features_to_matrix,
)

logger = logging.getLogger("api.main")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

MODELS_DIR = Path("models")
PREPROCESSING_PATH = MODELS_DIR / "preprocessing_pipeline.pkl"
CLASSIFIER_PATH = MODELS_DIR / "best_model.pkl"

_preprocess: Pipeline | None = None
_classifier: BaseEstimator | None = None


def _load_artifacts() -> None:
    """Load preprocessing pipeline and classifier from disk."""
    global _preprocess, _classifier
    _preprocess = None
    _classifier = None
    if not PREPROCESSING_PATH.is_file():
        logger.warning("Missing preprocessing artifact: %s", PREPROCESSING_PATH)
        return
    if not CLASSIFIER_PATH.is_file():
        logger.warning("Missing classifier artifact: %s", CLASSIFIER_PATH)
        return
    try:
        _preprocess = joblib.load(PREPROCESSING_PATH)
        _classifier = joblib.load(CLASSIFIER_PATH)
    except Exception:
        logger.exception("Failed to load model artifacts with joblib")
        _preprocess = None
        _classifier = None


class ProcessTimeMiddleware(BaseHTTPMiddleware):
    """Add ``X-Process-Time-Ms`` header for request-level latency visibility."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
        return response


app = FastAPI(title="Credit Card Fraud Detection API", version="0.2.0")
app.add_middleware(ProcessTimeMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return 422 with validation detail for bad JSON or field types."""
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.on_event("startup")
def on_startup() -> None:
    """Load preprocessing + classifier once; API stays up if files are missing."""
    _load_artifacts()
    if _classifier is not None and _preprocess is not None:
        logger.info("Loaded preprocessing: %s, classifier: %s", PREPROCESSING_PATH, CLASSIFIER_PATH)
    else:
        logger.warning("Model artifacts not loaded; /predict will return 503 until training is run.")


@app.get("/health")
def health() -> dict[str, str | bool]:
    """Readiness-style health: service up and whether inference artifacts are present."""
    loaded = _preprocess is not None and _classifier is not None
    return {"status": "healthy", "model_loaded": loaded}


@app.post("/predict", response_model=FraudPredictResponse)
def predict(body: TransactionFeatures) -> FraudPredictResponse:
    """Score a single transaction; applies saved preprocessing then ``predict_proba``."""
    if _preprocess is None or _classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training and ensure preprocessing_pipeline.pkl and best_model.pkl exist.",
        )

    t0 = time.perf_counter()
    try:
        x_raw = transaction_features_to_matrix(body)
        x_transformed = _preprocess.transform(x_raw)
        proba_positive = _classifier.predict_proba(x_transformed)[0, 1]
    except Exception as exc:
        logger.exception("Prediction failed (internal model error)")
        raise HTTPException(status_code=500, detail="Model inference failed") from exc

    latency_ms = (time.perf_counter() - t0) * 1000
    risk_score = float(proba_positive)
    fraud = risk_score >= 0.5

    ts = datetime.now(timezone.utc).isoformat()
    logger.info(
        "prediction ts=%s risk_score=%.6f latency_ms=%.2f",
        ts,
        risk_score,
        latency_ms,
    )

    return FraudPredictResponse(fraud=fraud, risk_score=risk_score)
