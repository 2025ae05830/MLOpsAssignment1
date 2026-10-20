"""FastAPI model serving API.

Endpoints:
    GET  /health  -> liveness check
    POST /predict -> JSON features -> prediction + confidence

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""
import logging
import time

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])

logger = logging.getLogger("heart_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="Heart Disease Prediction API")

MODEL_PATH = "models/best_model"
_model = None


def get_model():
    global _model
    if _model is None:
        logger.info("Loading model from %s", MODEL_PATH)
        _model = mlflow.sklearn.load_model(MODEL_PATH)
    return _model


class Features(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


class Prediction(BaseModel):
    prediction: int = Field(description="0 = no disease, 1 = disease")
    confidence: float


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info("%s %s -> %s (%.3fs)", request.method, request.url.path,
                response.status_code, duration)
    REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.url.path).observe(duration)
    return response


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(features: Features):
    model = get_model()
    row = pd.DataFrame([features.model_dump()])
    prediction = int(model.predict(row)[0])
    confidence = float(model.predict_proba(row)[0, 1])
    return Prediction(prediction=prediction, confidence=confidence)
