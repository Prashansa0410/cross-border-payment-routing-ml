"""decision_engine.py  -  Prescription Goal: "Recommend an appropriate payment route"
----------------------------------------------------------------------------------
Non-ML component. Takes the ML predictions for every candidate route and applies
business constraints + a weighted multi-objective score.

Two modes map directly to the two requirements in the problem statement:
  mode = "prediction"   -> pick the route with the highest predicted success probability
  mode = "optimisation" -> pick the route with the best combined score of
                           success probability, settlement latency and route cost
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from logger import setup_logger

logger = setup_logger("decision_engine")

DEFAULT_WEIGHTS = {"success": 0.6, "latency": 0.25, "cost": 0.15}


@dataclass
class Constraints:
    max_cost: Optional[float] = None  # currency units
    max_latency_seconds: Optional[float] = None
    min_success_probability: Optional[float] = None
    excluded_routes: List[str] = field(default_factory=list)


@dataclass
class RouteScore:
    route_id: str
    sending_institution: str
    receiving_institution: str
    route_cost: float
    predicted_success_probability: float
    predicted_settlement_seconds: float
    score: float = 0.0
    feasible: bool = True
    violated: List[str] = field(default_factory=list)


def _minmax(values: List[float]) -> List[float]:
    lo, hi = min(values), max(values)
    return [(v - lo) / (hi - lo) if hi > lo else 1.0 for v in values]


def recommend(
    candidates: List[Dict],
    mode: str = "optimisation",
    weights: Optional[Dict[str, float]] = None,
    constraints: Optional[Constraints] = None,
) -> Dict:
    """Recommends an optimal payment route from a list of candidate routes with predictions.

    Args:
        candidates: list of dicts with keys route_id, sending_institution, receiving_institution,
                    route_cost, predicted_success_probability, predicted_settlement_seconds
        mode: "optimisation" or "prediction"
        weights: Dictionary of multi-objective weights (success, latency, cost)
        constraints: Hard operational business constraints
    """
    if not candidates:
        logger.error("Empty candidate list passed to decision engine.")
        raise ValueError("Candidate route list cannot be empty.")

    weights = weights or DEFAULT_WEIGHTS
    constraints = constraints or Constraints()
    scores = [RouteScore(**c) for c in candidates]

    logger.debug(
        "Evaluating %d candidate routes under mode='%s' with constraints=%s",
        len(scores),
        mode,
        constraints,
    )

    # 1. Hard business constraints
    for s in scores:
        if s.route_id in constraints.excluded_routes:
            s.violated.append("excluded_route")
        if constraints.max_cost is not None and s.route_cost > constraints.max_cost:
            s.violated.append("max_cost")
        if (
            constraints.max_latency_seconds is not None
            and s.predicted_settlement_seconds > constraints.max_latency_seconds
        ):
            s.violated.append("max_latency")
        if (
            constraints.min_success_probability is not None
            and s.predicted_success_probability < constraints.min_success_probability
        ):
            s.violated.append("min_success_probability")
        s.feasible = len(s.violated) == 0

    feasible = [s for s in scores if s.feasible]
    constraints_relaxed = False
    if not feasible:  # graceful degradation: never return "no route"
        logger.warning(
            "No route satisfied all hard constraints. Relaxing constraints to ensure operational continuity."
        )
        feasible = scores
        constraints_relaxed = True

    # 2. Scoring
    if mode == "prediction":
        for s in feasible:
            s.score = s.predicted_success_probability
    else:
        succ = [s.predicted_success_probability for s in feasible]
        lat = _minmax([s.predicted_settlement_seconds for s in feasible])
        cost = _minmax([s.route_cost for s in feasible])
        succ_n = _minmax(succ)
        for s, a, b, c in zip(feasible, succ_n, lat, cost):
            s.score = (
                weights["success"] * a + weights["latency"] * (1 - b) + weights["cost"] * (1 - c)
            )

    ranked = sorted(feasible, key=lambda s: s.score, reverse=True)
    best = ranked[0]

    logger.info(
        "Route recommendation determined: %s (score=%.3f, relaxed=%s)",
        best.route_id,
        best.score,
        constraints_relaxed,
    )

    return {
        "mode": mode,
        "weights": weights if mode != "prediction" else None,
        "recommended_route": best.route_id,
        "recommendation": asdict(best),
        "constraints_relaxed": constraints_relaxed,
        "ranked_routes": [asdict(s) for s in ranked],
        "infeasible_routes": [asdict(s) for s in scores if not s.feasible],
        "explanation": _explain(best, mode, constraints_relaxed),
    }


def _explain(best: RouteScore, mode: str, relaxed: bool) -> str:
    hrs = best.predicted_settlement_seconds / 3600
    base = (
        f"{best.route_id} via {best.sending_institution}->{best.receiving_institution}: "
        f"P(success)={best.predicted_success_probability:.1%}, expected settlement {hrs:.2f} h, cost {best.route_cost:.2f}."
    )
    if mode == "prediction":
        base += " Selected because it has the highest predicted success probability."
    else:
        base += " Selected because it has the best weighted trade-off of success, latency and cost."
    if relaxed:
        base += " WARNING: no route satisfied all constraints; constraints were relaxed."
    return base
