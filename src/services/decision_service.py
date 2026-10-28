"""
Route Decision Service (non-ML microservice)  - port 8002
----------------------------------------------------------
Owns the route catalogue and business rules. Calls the Prediction Service
for ML insights and applies the prescription logic (decision_engine.py).
  POST /recommend  {"payment": {...}, "mode": "optimisation"|"prediction",
                    "weights": {...}, "constraints": {...}}
  GET  /routes
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from decision_engine import DEFAULT_WEIGHTS, Constraints, recommend
from services.common import post_json, serve

ART = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")
ROUTE_CATALOG = json.load(open(f"{ART}/route_catalog.json"))
PREDICTION_URL = os.environ.get("PREDICTION_URL", "http://127.0.0.1:8001")


def routes():
    return {"routes": ROUTE_CATALOG}


def do_recommend(body):
    payment = body.get("payment")
    if not payment:
        return 400, {"error": "payment is required"}
    mode = body.get("mode", "optimisation")
    weights = {**DEFAULT_WEIGHTS, **body.get("weights", {})}
    cons = Constraints(**body.get("constraints", {}))
    candidates = [
        r for r in ROUTE_CATALOG if r["route_id"] not in cons.excluded_routes
    ] or ROUTE_CATALOG
    try:
        pred = post_json(f"{PREDICTION_URL}/predict", {"payment": payment, "routes": candidates})
    except Exception as e:
        return 502, {"error": "prediction service unavailable", "detail": str(e)}
    if "error" in pred:
        return 502, {"error": "prediction service failed", "detail": pred}
    cands = [
        {
            k: p[k]
            for k in [
                "route_id",
                "sending_institution",
                "receiving_institution",
                "route_cost",
                "predicted_success_probability",
                "predicted_settlement_seconds",
            ]
        }
        for p in pred["predictions"]
    ]
    result = recommend(cands, mode=mode, weights=weights, constraints=cons)
    result["model_version"] = pred["model_version"]
    result["inference_ms"] = pred["inference_ms"]
    return 200, result


if __name__ == "__main__":
    serve("decision-service", 8002, {"/routes": routes}, {"/recommend": do_recommend})
