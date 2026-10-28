"""Pydantic schemas and contracts for the Payment Routing REST API."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class PaymentInput(BaseModel):
    """Transaction characteristics provided at decision time."""

    transaction_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the payment transaction",
        examples=["TXN_12345678"],
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="ISO 8601 transaction timestamp",
        examples=["2026-10-28T10:30:00"],
    )
    source_country: str = Field(..., min_length=2, max_length=50, examples=["US"])
    destination_country: str = Field(..., min_length=2, max_length=50, examples=["India"])
    source_currency: str = Field(..., min_length=3, max_length=10, examples=["USD"])
    destination_currency: str = Field(..., min_length=3, max_length=10, examples=["INR"])
    amount: float = Field(
        ..., gt=0.0, description="Payment amount in source currency", examples=[1500.0]
    )
    payment_type: Literal["P2P", "B2B", "C2B", "B2C"] = Field(
        ..., description="Category of cross-border transfer", examples=["P2P"]
    )


class CandidateRouteInput(BaseModel):
    """Specification of a candidate banking corridor / route."""

    route_id: str = Field(..., examples=["ROUTE_A"])
    sending_institution: str = Field(..., examples=["Bank_A"])
    receiving_institution: str = Field(..., examples=["Bank_X"])
    route_cost: float = Field(..., ge=0.0, examples=[2.50])
    historical_route_success_rate: float = Field(..., ge=0.0, le=1.0, examples=[0.95])
    historical_avg_latency_seconds: float = Field(..., gt=0.0, examples=[1800.0])


class ConstraintsInput(BaseModel):
    """Operational constraints for route decisioning."""

    max_cost: Optional[float] = Field(default=None, ge=0.0, examples=[3.0])
    max_latency_seconds: Optional[float] = Field(default=None, gt=0.0, examples=[7200.0])
    min_success_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0, examples=[0.85])
    excluded_routes: List[str] = Field(default_factory=list, examples=[["ROUTE_F"]])


class RoutePrediction(BaseModel):
    """Model predictions for a single candidate route."""

    route_id: str
    sending_institution: str
    receiving_institution: str
    route_cost: float
    predicted_success_probability: float = Field(..., ge=0.0, le=1.0)
    predicted_settlement_seconds: float = Field(..., ge=0.0)
    predicted_class: int = Field(..., ge=0, le=1)


class PredictRequest(BaseModel):
    """Request payload for model inference."""

    payment: PaymentInput
    routes: Optional[List[CandidateRouteInput]] = Field(
        default=None,
        description="Optional candidate routes. If omitted, uses active route catalog.",
    )


class PredictResponse(BaseModel):
    """Inference predictions response."""

    predictions: List[RoutePrediction]
    model_version: str
    inference_ms: float


class RecommendRouteRequest(BaseModel):
    """Request payload for route recommendation."""

    payment: PaymentInput
    mode: Literal["optimisation", "prediction"] = Field(
        default="optimisation",
        description="'optimisation' scores success, latency, and cost; 'prediction' maximizes success rate.",
    )
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Weights for multi-objective optimization (must sum to ~1.0)",
        examples=[{"success": 0.6, "latency": 0.25, "cost": 0.15}],
    )
    constraints: Optional[ConstraintsInput] = None

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if v is not None:
            required_keys = {"success", "latency", "cost"}
            if not required_keys.issubset(v.keys()):
                raise ValueError(f"Weights dictionary must contain keys: {required_keys}")
            for k, val in v.items():
                if val < 0:
                    raise ValueError(f"Weight '{k}' cannot be negative")
        return v


class RecommendRouteResponse(BaseModel):
    """Route recommendation response."""

    transaction_id: str
    recommended_route: str
    mode: str
    recommendation: Dict[str, Any]
    ranked_routes: List[Dict[str, Any]]
    infeasible_routes: List[Dict[str, Any]]
    explanation: str
    constraints_relaxed: bool
    inference_ms: float


class HealthResponse(BaseModel):
    """Service health response."""

    status: str
    model_loaded: bool
    service: str
    timestamp: float


class ModelInfoResponse(BaseModel):
    """Metadata response regarding loaded models."""

    model_version: str
    model_type: str
    decision_threshold: float
    classifier_metrics: Dict[str, Any]
    regressor_metrics: Dict[str, Any]
    supported_features: List[str]
    candidate_routes_count: int
