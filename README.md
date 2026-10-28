# Cross-Border Payment Routing ML System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests: pytest](https://img.shields.io/badge/tests-25%20passed-brightgreen.svg)](https://docs.pytest.org/)

An intelligent, ML-enabled decision system for cross-border payment routing. The platform evaluates candidate partner-bank corridors and predicts:
1. **Payment Success Probability ($P(\text{success})$):** Calibrated `HistGradientBoostingClassifier` (ROC-AUC 0.6980, Expected Calibration Error 0.66%).
2. **Expected Settlement Latency:** Log-target `HistGradientBoostingRegressor` ($R^2$ 0.942, MAE 69.7 min).
3. **Multi-Objective Routing Engine:** Combines predicted success rate, settlement latency, route cost, and operational constraints (max cost, max latency, min success, excluded corridors).

Built for **AIMLZG546 — Software Engineering for Machine Learning (Group 65)**.

---

## Architecture Overview

The system is structured according to clean Object-Oriented and functional principles with a high-performance serving layer:

* **Data Ingestion & Quality Validation (`src/data_ingestion.py`):** Schema validation, missing value checks, duplicate detection, and stratified train/val/test splitting.
* **Feature Engineering Pipeline (`src/feature_pipeline.py`):** `ColumnTransformer` with imputation, one-hot encoding, standard scaling, and temporal enrichment.
* **Offline Training Pipeline (`src/model_trainer.py`):** Model training, validation-based decision threshold tuning, and artifact generation.
* **Production Inference Service (`src/inference.py`):** Thread-safe Singleton `PaymentPredictor` with in-memory model caching, domain range clipping, and latency tracking.
* **Decision Optimization Engine (`src/decision_engine.py`):** Hard business constraints enforcement and multi-objective weighted utility scoring.
* **Quality & Drift Evaluator (`src/metrics_evaluator.py`):** Model quality metrics and Population Stability Index (PSI) covariate drift detection.
* **FastAPI REST API (`src/api/`):** Pydantic v2 validation contracts, OpenAPI documentation, and structured error responses.

---

## Project Structure

```
├── 65.ipynb                                 # Executed Jupyter notebook demonstrating the entire pipeline
├── 65.docx                                  # Comprehensive Assignment Report
├── ASSIGNMENT2_IMPLEMENTATION_PLAN.md       # Requirements and verification mapping
├── RESEARCH_VS_PRODUCTION.md                # Research prototype vs Production architecture analysis
├── PRODUCTION_EXPERIMENTATION_AND_SECURITY.md # Shadow deployment architecture & ML security
├── pyproject.toml                           # Black, Ruff, and Pytest configuration
├── requirements.txt                         # Pinned dependencies
├── data.csv                                 # Canonical dataset (50,000 transactions)
├── artifacts/                               # Versioned model artifacts, threshold, and route catalog
│   ├── success_classifier.joblib
│   ├── settlement_regressor.joblib
│   ├── threshold.json
│   ├── route_catalog.json
│   └── metrics.json
├── figures/                                 # Calibration & evaluation visualizations
├── src/                                     # Production application modules
│   ├── api/                                 # FastAPI application & Pydantic schemas
│   ├── config.py                            # Central configuration & feature schemas
│   ├── logger.py                            # Structured Python logging
│   ├── data_ingestion.py                    # Ingestion & validation
│   ├── feature_pipeline.py                  # Preprocessing & temporal enricher
│   ├── model_trainer.py                     # Training & evaluation
│   ├── inference.py                         # In-memory prediction engine
│   ├── decision_engine.py                   # Multi-objective route selection
│   ├── metrics_evaluator.py                 # Data & Model quality metrics
│   ├── event_bus.py                         # Event broker (pub/sub)
│   └── services/                            # Microservices tier
└── tests/                                   # Automated test suite (25 tests)
    ├── test_data_validation.py              # Schema, missing values, drift
    ├── test_model_training.py               # Small batch overfit & convergence tests
    ├── test_model_inference.py              # Range, directional & invariance tests
    ├── test_decision_engine.py              # Business rules & constraints
    ├── test_api.py                          # FastAPI endpoint tests
    └── test_integration.py                  # End-to-end integration tests
```

---

## Quickstart Guide

### 1. Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Automated Tests

Execute the 25-test suite:
```bash
pytest -v
```

### 3. Run Code Quality Checks

```bash
# Check formatting with Black
black --check src tests

# Run linter with Ruff
ruff check src tests
```

### 4. Start the REST API

Launch the FastAPI application:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation is available at: **`http://127.0.0.1:8000/docs`**.

---

## REST API Endpoints

### 1. Health Check
```bash
curl -X GET http://127.0.0.1:8000/health
```

### 2. Model Metadata & Quality Metrics
```bash
curl -X GET http://127.0.0.1:8000/model-info
```

### 3. Payment Route Recommendation
```bash
curl -X POST http://127.0.0.1:8000/recommend-route \
  -H "Content-Type: application/json" \
  -d '{
    "payment": {
      "transaction_id": "TXN_PROD_001",
      "timestamp": "2026-10-28T14:30:00",
      "source_country": "US",
      "destination_country": "India",
      "source_currency": "USD",
      "destination_currency": "INR",
      "amount": 1500.0,
      "payment_type": "P2P"
    },
    "mode": "optimisation",
    "weights": {"success": 0.6, "latency": 0.25, "cost": 0.15},
    "constraints": {"max_cost": 3.0}
  }'
```

---

## Quality Metrics Summary

* **Payment Success Classifier:**
  * ROC-AUC: `0.6980`
  * F1 Score: `0.7361`
  * Brier Score: `0.0761`
  * Expected Calibration Error (ECE): `0.66%`
  * Decision Threshold: `0.910` (tuned on validation split)
* **Settlement Latency Regressor:**
  * $R^2$ Score: `0.942`
  * Mean Absolute Error (MAE): `69.7 minutes` (vs. `113.4 minutes` historical baseline)
* **Data Quality:**
  * Schema Conformance: `100%`
  * Duplicate Transactions: `0.0%`
  * Population Stability Index (PSI): `0.0000` (stable distribution)


