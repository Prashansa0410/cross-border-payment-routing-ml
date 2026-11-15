# RouteFlow AI — Intelligent Cross-Border Payment Routing ML Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5+-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests: pytest](https://img.shields.io/badge/tests-26%20passed-brightgreen.svg)](https://docs.pytest.org/)

**RouteFlow AI** is a production-grade machine learning platform that optimizes cross-border capital routing across multi-hop correspondent banking networks. By combining calibrated gradient boosted decision trees with a constrained multi-objective decision engine, the system predicts transaction success probabilities, forecasts settlement latency, and selects optimal settlement paths in real time.

---

## 🌍 The Core Problem: Why Global Payments Need Machine Learning

Cross-border money movement does not happen along a direct path. Because financial institutions cannot maintain bilateral accounting relationships with every bank worldwide, international transfers rely on **intermediary correspondent banking rails** and reciprocal ledgers (*Nostro/Vostro accounts*).

When capital moves between currency corridors (e.g., USD → INR, GBP → NGN, SGD → PHP), multiple intermediary paths are available. However:
* **Batch Window Cutoffs & Liquidity Bottlenecks:** Intermediary clearing systems (Fedwire, CHAPS, RTGS) have strict cutoffs. Off-hours or weekend transfers frequently stall for 48–72 hours.
* **Counterparty Rejection Risk:** Strict anti-money laundering (AML), sanctions screening, and fluctuating partner liquidity cause unpredictable transfer failures.
* **Opaque & Disconnected Costs:** Correspondent banks deduct fixed fees and FX spreads that vary wildly by corridor.
* **Failure of Static Routing:** Traditional banking engines rely on rigid, hardcoded lookup tables that cannot adapt to real-time volatility, volume surges, or regional holidays.

**RouteFlow AI** solves this by treating route selection as a **dual-task probabilistic machine learning problem coupled with constrained multi-objective utility optimization**.

---

## ⚡ The RouteFlow AI Architecture

```
Incoming Payment Request (e.g. US -> India, $5,000 USD)
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│  FastAPI Gateway Service (Pydantic v2 Contracts)       │
└──────────────────┬─────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│  Temporal Feature Extraction & Corridor Rail Discovery │
│  (hour_of_day, day_of_week, weekend, currency pair)    │
└──────────────────┬─────────────────────────────────────┘
                   │ Parallel Evaluation Across Candidate Routes
                   ▼
┌────────────────────────────────────────────────────────┐
│           Dual Machine Learning Inference              │
│                                                        │
│  ┌─────────────────────────┐ ┌──────────────────────┐  │
│  │ HistGradientBoosting    │ │ HistGradientBoosting │  │
│  │ Classifier              │ │ Regressor            │  │
│  │ P(success) = 98.1%      │ │ Latency = 26.6 mins  │  │
│  │ (Calibrated Sigmoid)    │ │ (Log-Transformed R²) │  │
│  └───────────┬─────────────┘ └──────────┬───────────┘  │
└──────────────┼──────────────────────────┼──────────────┘
               └────────────┬─────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│        Constrained Route Decision Engine               │
│                                                        │
│  1. Hard Constraint Filter: P(success) >= 85.0%        │
│  2. Multi-Objective Scoring:                           │
│     Score = w_success·P - w_latency·T_hours - w_cost·C │
└──────────────────┬─────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│  Recommended Route + Ranked Matrix + Telemetry Audit   │
└────────────────────────────────────────────────────────┘
```

### 1. Dual-Task Machine Learning Pipeline
* **Payment Success Classifier (`HistGradientBoostingClassifier`):** Predicts the probability that a transaction completes without counterparty rejection or compliance timeout. Calibrated with Platt sigmoid scaling to guarantee reliable probabilities (Expected Calibration Error: `0.66%`).
* **Settlement Latency Regressor (`HistGradientBoostingRegressor`):** Fitted on log-transformed settlement duration ($\log(1 + y)$), capturing non-linear weekend delays and banking cutoffs ($R^2 = 0.942$, reducing MAE by 38.5% compared to historical baselines).

### 2. Multi-Objective Prescriptive Decision Engine
Machine learning models predict outcomes, but **business governance enforces decisions**. The decision engine:
1. **Applies Hard SLA Constraints:** Immediately eliminates routes that fail minimum success thresholds (e.g. $P(\text{success}) < 0.85$), maximum cost, or excluded corridors.
2. **Evaluates Utility Scoring:** Computes a composite objective score:
   $$\text{Utility} = w_{\text{success}} \cdot P(\text{success}) - w_{\text{latency}} \cdot \frac{T_{\text{settlement}}}{3600} - w_{\text{cost}} \cdot \text{Cost}$$
3. **Supports Configurable Operational Policies:**
   * **Balanced Policy:** Standard multi-objective optimization across all dimensions.
   * **Ultra-Fast Policy (Speed Priority):** Heavily penalizes latency hours for time-sensitive commercial wires.
   * **Lowest Cost Policy:** Prioritizes minimal clearing fees while maintaining SLA reliability.

---

## 🖥️ Interactive Web Interfaces

RouteFlow AI includes two interactive web applications embedded directly into the application server:

1. **The 5-Stage Interactive Corridor Pipeline (`/demo`):**
   * Designed for stakeholders, engineers, and non-domain audiences.
   * Explains Nostro/Vostro ledgers and walks through the 5 architectural stages in plain English.
   * Features interactive scenario triggers (Normal Daytime B2B, Weekend High-Value Transfer, Low-Cost Retail Remittance).
   * Includes live API telemetry, animated loading buffers, and a collapsible comparison matrix across all candidate routes.
2. **The Operational Routing Dashboard (`/`):**
   * Operational console for payment operators.
   * One-click corridor presets, interactive policy toggles (Balanced, Ultra-Fast, Lowest Cost), and live candidate scoring tables.
   * Integrated dark/light mode toggle with preference persistence.

---

## 📊 Benchmark & Quality Metrics

| Component | Metric | Score | Industry Comparison / Baseline |
|---|---|---|---|
| **Success Classifier** | ROC-AUC | `0.6980` | Near optimal for Bernoulli noise ceiling |
| **Success Classifier** | F1 Score | `0.7361` | Tuned on validation split |
| **Success Classifier** | Brier Score | `0.0761` | Low probabilistic deviation |
| **Success Classifier** | Expected Calibration Error (ECE) | `0.66%` | **Highly calibrated probabilities** |
| **Latency Regressor** | $R^2$ Score | `0.942` | Explains 94.2% of settlement variance |
| **Latency Regressor** | Mean Absolute Error (MAE) | `69.7 min` | **38.5% reduction** vs. baseline (113.4 min) |
| **Data Quality** | Schema Conformance | `100.0%` | Zero schema discrepancies across 50,000 records |
| **Data Quality** | Duplicate Rate | `0.00%` | Fully deduplicated transaction IDs |
| **Data Quality** | Covariate Drift (PSI) | `0.0000` | Population Stability Index indicating zero drift |

---

## 📁 Repository Structure

```
├── main.py                                    # Application entrypoint (launches REST API & UI)
├── pyproject.toml                             # Black, Ruff, and Pytest configuration
├── requirements.txt                           # Pinned production dependencies
├── data.csv                                   # Canonical historical dataset (50,000 transactions)
├── artifacts/                                 # Versioned model artifacts and catalogs
│   ├── success_classifier.joblib              # Fitted HistGradientBoosting classifier
│   ├── settlement_regressor.joblib            # Fitted HistGradientBoosting regressor
│   ├── threshold.json                         # Tuned decision threshold
│   ├── route_catalog.json                     # Active correspondent corridor topology
│   └── metrics.json                           # Persisted model and data quality telemetry
├── figures/                                   # Evaluation and calibration plots
├── notebooks/                                 # Exploratory and interactive walkthroughs
│   └── pipeline_walkthrough.ipynb             # Interactive pipeline execution notebook
├── src/                                       # Production application modules
│   ├── api/                                   # FastAPI REST endpoints & Pydantic schemas
│   │   ├── app.py                             # API router & static UI mounters
│   │   └── schemas.py                         # Pydantic v2 data contracts
│   ├── config.py                              # Central configuration & feature schemas
│   ├── logger.py                              # Structured logging configuration
│   ├── data_ingestion.py                      # Data validation & stratified splitting
│   ├── feature_pipeline.py                    # Preprocessing & temporal enrichers
│   ├── model_trainer.py                       # Offline training pipeline & calibration
│   ├── inference.py                           # Thread-safe Singleton prediction service
│   ├── decision_engine.py                     # Multi-objective route decision engine
│   ├── metrics_evaluator.py                   # Data quality & PSI drift evaluators
│   └── event_bus.py                           # Asynchronous pub/sub event broker
├── tests/                                     # Automated test suite (26 tests)
│   ├── test_api.py                            # REST API contracts & UI endpoint tests
│   ├── test_data_validation.py                # Schema, null rates, and drift tests
│   ├── test_model_training.py                 # Small-batch overfit & convergence tests
│   ├── test_model_inference.py                # Range, directional & invariance tests
│   ├── test_decision_engine.py                # Hard constraints & utility weighting tests
│   └── test_integration.py                    # End-to-end integration tests
└── ui/                                        # Embedded Web Interfaces
    ├── index.html                             # Operational Routing Dashboard
    ├── demo.html                              # 5-Stage Interactive Corridor Story Walkthrough
    └── server.py                              # Standalone static showcase server (optional)
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Prashansa0410/cross-border-payment-routing-ml.git
cd cross-border-payment-routing-ml

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated QA Tests

Execute the 26-test automated suite (including unit, integration, directional ML, and invariant tests):
```bash
pytest -v
```

### 3. Run Code Quality Checks

```bash
# Verify formatting with Black
black --check src tests

# Run linter with Ruff
ruff check src tests
```

### 4. Launch the Application

Launch the unified server:
```bash
python main.py
```

Open your browser to explore:
* ⚡ **Operational Dashboard:** [http://localhost:8000/](http://localhost:8000/)
* 📖 **5-Stage Pipeline Walkthrough:** [http://localhost:8000/demo](http://localhost:8000/demo)
* 📜 **Interactive Swagger REST Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📡 REST API Reference

### Health Check
```bash
curl -X GET http://localhost:8000/health
```
```json
{
  "status": "ok",
  "model_loaded": true,
  "service": "payment-routing-api",
  "timestamp": 1794717800.5
}
```

### Model Metadata & Quality Metrics
```bash
curl -X GET http://localhost:8000/model-info
```

### Optimal Route Recommendation
```bash
curl -X POST http://localhost:8000/recommend-route \
  -H "Content-Type: application/json" \
  -d '{
    "payment": {
      "transaction_id": "TXN_LIVE_101",
      "timestamp": "2026-11-15T10:30:00",
      "source_country": "US",
      "destination_country": "India",
      "source_currency": "USD",
      "destination_currency": "INR",
      "amount": 5000.0,
      "payment_type": "B2B"
    },
    "mode": "optimisation",
    "weights": {"success": 0.60, "latency": 0.25, "cost": 0.15},
    "constraints": {"min_success_probability": 0.85, "max_cost": 5.0}
  }'
```
**Response:**
```json
{
  "transaction_id": "TXN_LIVE_101",
  "recommended_route": "ROUTE_E",
  "mode": "optimisation",
  "recommendation": {
    "route_id": "ROUTE_E",
    "sending_institution": "Bank_B",
    "receiving_institution": "Bank_Z",
    "route_cost": 3.5,
    "predicted_success_probability": 0.9811,
    "predicted_settlement_seconds": 1595.0,
    "score": 0.8643,
    "feasible": true
  },
  "ranked_routes": [...],
  "explanation": "ROUTE_E via Bank_B->Bank_Z: P(success)=98.1%, expected settlement 0.44 h, cost 3.50. Selected because it has the best weighted trade-off of success, latency and cost.",
  "constraints_relaxed": false,
  "inference_ms": 12.4
}
```

---

## 📖 Technical Whitepapers & Engineering Deep Dives

* 📘 [Architectural Evolution: From Research Prototype to Enterprise ML Serving](RESEARCH_VS_PRODUCTION.md) — Comprehensive analysis of Singleton patterns, memory caching, defensive Pydantic contracts, and hierarchical error handling.
* 🛡️ [Production Experimentation, Shadow Deployment & ML Security Architecture](PRODUCTION_EXPERIMENTATION_AND_SECURITY.md) — Production shadow deployment (dark launching), canary rollouts, drift gating (PSI), adversarial perturbation defense, and data poisoning mitigation.

---

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.
