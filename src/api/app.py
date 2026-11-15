"""FastAPI Application for Cross-Border Payment Routing ML System.
------------------------------------------------------------------
Exposes standardized REST endpoints for health checks, model prediction,
route recommendation, and model metadata.
"""

import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure src directory is in sys.path regardless of current working directory
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

try:
    from .schemas import (
        HealthResponse,
        ModelInfoResponse,
        PredictRequest,
        PredictResponse,
        RecommendRouteRequest,
        RecommendRouteResponse,
    )
except (ImportError, ValueError):
    try:
        from api.schemas import (
            HealthResponse,
            ModelInfoResponse,
            PredictRequest,
            PredictResponse,
            RecommendRouteRequest,
            RecommendRouteResponse,
        )
    except ImportError:
        from src.api.schemas import (
            HealthResponse,
            ModelInfoResponse,
            PredictRequest,
            PredictResponse,
            RecommendRouteRequest,
            RecommendRouteResponse,
        )
from config import DEFAULT_WEIGHTS
from decision_engine import Constraints, recommend
from feature_pipeline import TemporalEnricher
from inference import PaymentPredictor
from logger import setup_logger

logger = setup_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes models on startup."""
    logger.info("Starting Payment Routing REST API. Pre-loading ML models...")
    try:
        predictor = PaymentPredictor.get_instance()
        logger.info(
            "ML models ready: loaded=%s, version=%s",
            predictor.is_loaded,
            predictor.metrics.get("model_version"),
        )
    except Exception as e:
        logger.error("Failed to initialize ML models on startup: %s", str(e), exc_info=True)
    yield
    logger.info("Shutting down Payment Routing REST API...")


app = FastAPI(
    title="Cross-Border Payment Routing ML API",
    description=(
        "Production-grade REST API providing ML-driven payment success classification, "
        "settlement time regression, and multi-objective corridor recommendation."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom handler for request validation failures (HTTP 422)."""
    errors = []
    for err in exc.errors():
        loc = " -> ".join(str(x) for x in err.get("loc", []))
        errors.append(f"{loc}: {err.get('msg')}")
    logger.warning("Validation error on %s %s: %s", request.method, request.url.path, errors)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Invalid request payload", "details": errors},
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Custom handler for domain validation errors (HTTP 400)."""
    logger.warning("Bad request on %s: %s", request.url.path, str(exc))
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Bad request", "message": str(exc)},
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    """Custom handler for model unavailable / service errors (HTTP 503)."""
    logger.error("Service unavailable on %s: %s", request.url.path, str(exc))
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"error": "Service unavailable", "message": str(exc)},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Safety net avoiding tracebacks to clients (HTTP 500)."""
    logger.error("Unhandled internal error on %s: %s", request.url.path, str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred processing your request.",
        },
    )


@app.get("/health", response_model=HealthResponse, tags=["Observability"])
async def health_check() -> HealthResponse:
    """Service health and model load readiness check."""
    predictor = PaymentPredictor.get_instance()
    is_healthy = predictor.is_loaded
    return HealthResponse(
        status="ok" if is_healthy else "degraded",
        model_loaded=is_healthy,
        service="payment-routing-api",
        timestamp=time.time(),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Metadata"])
async def get_model_info() -> ModelInfoResponse:
    """Returns safe model metadata and performance metrics."""
    predictor = PaymentPredictor.get_instance()
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not currently loaded",
        )
    info = predictor.get_model_info()
    return ModelInfoResponse(**info)


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_payment(req: PredictRequest) -> PredictResponse:
    """Generates payment success probability and settlement time for candidate routes."""
    predictor = PaymentPredictor.get_instance()
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction models are currently unavailable",
        )

    # Enrich payment with temporal fields
    payment_dict = req.payment.model_dump()
    enriched = TemporalEnricher.enrich(payment_dict, payment_dict.get("timestamp"))

    routes_list = [r.model_dump() for r in req.routes] if req.routes else None
    result = predictor.predict(enriched, routes_list)

    return PredictResponse(**result)


@app.post("/recommend-route", response_model=RecommendRouteResponse, tags=["Prescription"])
async def recommend_route(req: RecommendRouteRequest) -> RecommendRouteResponse:
    """Applies constraints and multi-objective scoring to recommend an optimal route."""
    predictor = PaymentPredictor.get_instance()
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction models are currently unavailable",
        )

    t0 = time.perf_counter()
    payment_dict = req.payment.model_dump()
    txn_id = payment_dict.get("transaction_id") or f"TXN_{uuid.uuid4().hex[:8].upper()}"
    enriched = TemporalEnricher.enrich(payment_dict, payment_dict.get("timestamp"))

    # Convert constraints
    cons_dict = req.constraints.model_dump() if req.constraints else {}
    constraints = Constraints(**cons_dict)

    # Filter candidate routes if excluded_routes specified
    candidates = [
        r for r in predictor.route_catalog if r["route_id"] not in constraints.excluded_routes
    ] or predictor.route_catalog

    # 1. Run predictions
    pred_res = predictor.predict(enriched, candidates)
    predictions = pred_res["predictions"]

    # 2. Run decision engine
    weights = req.weights or DEFAULT_WEIGHTS
    cands_for_decision = [
        {
            "route_id": p["route_id"],
            "sending_institution": p["sending_institution"],
            "receiving_institution": p["receiving_institution"],
            "route_cost": p["route_cost"],
            "predicted_success_probability": p["predicted_success_probability"],
            "predicted_settlement_seconds": p["predicted_settlement_seconds"],
        }
        for p in predictions
    ]

    rec_result = recommend(
        candidates=cands_for_decision,
        mode=req.mode,
        weights=weights,
        constraints=constraints,
    )

    inference_ms = (time.perf_counter() - t0) * 1000.0

    return RecommendRouteResponse(
        transaction_id=txn_id,
        recommended_route=rec_result["recommended_route"],
        mode=rec_result["mode"],
        recommendation=rec_result["recommendation"],
        ranked_routes=rec_result["ranked_routes"],
        infeasible_routes=rec_result["infeasible_routes"],
        explanation=rec_result["explanation"],
        constraints_relaxed=rec_result["constraints_relaxed"],
        inference_ms=inference_ms,
    )


# ---------------------------------------------------------------------------
# Interactive Web Interfaces (Dashboard & Pipeline Walkthrough)
# ---------------------------------------------------------------------------
UI_DIR = Path(__file__).resolve().parent.parent.parent / "ui"
if UI_DIR.exists():

    @app.get("/", include_in_schema=False)
    async def serve_dashboard():
        """Serves the Operational Payment Routing Dashboard."""
        index_path = UI_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return JSONResponse({"status": "ok", "service": "payment-routing-api"})

    @app.get("/demo", include_in_schema=False)
    @app.get("/demo/", include_in_schema=False)
    async def serve_demo():
        """Serves the 5-Stage Interactive Pipeline & Corridor Walkthrough."""
        demo_path = UI_DIR / "demo.html"
        if demo_path.exists():
            return FileResponse(demo_path)
        return JSONResponse({"status": "error", "message": "Demo interface not found"})
