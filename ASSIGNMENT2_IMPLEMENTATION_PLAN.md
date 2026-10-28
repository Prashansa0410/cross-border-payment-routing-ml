# Assignment 2 Implementation Plan: Cross-Border Payment Routing ML System

**Course:** AIMLZG546 — Software Engineering for Machine Learning  
**Assignment:** Assignment II (Extending Assignment I)  
**Group:** 65  
**System:** Intelligent Cross-Border Payment Route Selection System  
**Status:** Completed & Validated (25/25 Tests Passing, Clean Black & Ruff Reports)

---

## 1. Executive Summary and Problem Context

The project implements an ML-enabled cross-border payment routing system. For every incoming transaction, the system evaluates candidate processing routes (banking corridors) to predict:
1. **Prediction Goal 1 (Classification):** Probability of payment success ($P(\text{success})$) using a calibrated `HistGradientBoostingClassifier`.
2. **Prediction Goal 2 (Regression):** Expected settlement latency in seconds using a log-target `HistGradientBoostingRegressor`.
3. **Prescription Goal (Decision Logic):** A multi-objective optimization engine combining predicted success probability, predicted settlement latency, route cost, and operational constraints (max cost, max latency, min success, excluded routes).

Assignment 2 builds upon Assignment 1, emphasizing production-grade software engineering for ML: refactoring into clean OOP/functional modules, building a standardized FastAPI REST API, introducing robust logging and error handling, comprehensive QA with `pytest` (including ML-specific training and directional inference tests), measurable data and model quality metrics, production experimentation design, security considerations, and rigorous code formatting and linting.

---

## 2. Comprehensive Requirements Analysis & Verification Mapping

| Requirement ID | Assignment 2 Requirement | Implementation File | Verification Evidence & Test Status | Status |
|---|---|---|---|---|
| **REQ-1** | Refactor/design ML application code using OOP/functional principles (data ingestion, feature engineering, model training, inference). | `src/data_ingestion.py`<br>`src/feature_pipeline.py`<br>`src/model_trainer.py`<br>`src/inference.py`<br>`src/decision_engine.py` | Single-responsibility classes (`DataIngestion`, `DataValidator`, `FeaturePipeline`, `ModelTrainer`, `PaymentPredictor`). Full type hints and docstrings. | **COMPLETED** |
| **REQ-2** | Clearly distinguish and demonstrate research code vs. production code for at least one component. | `RESEARCH_VS_PRODUCTION.md`<br>`65.ipynb`<br>`src/inference.py` | `RESEARCH_VS_PRODUCTION.md` details side-by-side comparison across 5 dimensions: lifecycle, input validation, error handling, logging, and testability. | **COMPLETED** |
| **REQ-3** | Implement proper error handling and logging across at least 3 critical functions/modules (INFO, WARNING, ERROR). | `src/logger.py`<br>`src/data_ingestion.py`<br>`src/inference.py`<br>`src/decision_engine.py`<br>`src/api/app.py` | Standard `logging` configured across 5 critical modules. Masked logs, hierarchical exception handling, safe HTTP 400/422/503 responses without tracebacks. | **COMPLETED** |
| **REQ-4** | Apply code formatting and linting tools (`black`, `ruff`); include before/after reports. | `pyproject.toml`<br>`src/`<br>`tests/` | 32 files formatted with `black` (`line-length=100`). `ruff check src tests` passed with 0 errors. | **COMPLETED** |
| **REQ-5** | Design and implement REST API (FastAPI) for model inference following good API practices. | `src/api/schemas.py`<br>`src/api/app.py` | FastAPI application exposing `/health`, `/predict`, `/recommend-route`, `/model-info`. Pydantic v2 validation contracts, OpenAPI docs at `/docs`. | **COMPLETED** |
| **REQ-6** | Implement at least 2 different types of tests using `pytest` (unit, integration, data validation). | `tests/test_data_validation.py`<br>`tests/test_decision_engine.py`<br>`tests/test_api.py`<br>`tests/test_integration.py` | 25 automated tests across 6 test suites covering data validation, decision engine, FastAPI client, and end-to-end integration. | **COMPLETED** |
| **REQ-7a** | Specific ML training tests (overfitting on a small batch, checking loss decreases). | `tests/test_model_training.py` | `test_overfitting_on_small_batch`: 30-sample batch achieves $\ge 95\%$ accuracy and log loss $<0.25$. `test_model_training_pipeline_execution`: verifies artifact output. | **COMPLETED** |
| **REQ-7b** | Specific ML inference tests (output shape/range, invariance/directional tests). | `tests/test_model_inference.py` | `test_directional_test_historical_success_rate` (monotonicity), `test_directional_test_temporal_latency_multiplier` (weekend multiplier), `test_invariance_test_transaction_id`. | **COMPLETED** |
| **REQ-8a** | Define and measure at least 2 Model Quality metrics (ROC-AUC, F1, calibration / ECE, MAE, R²). | `src/metrics_evaluator.py`<br>`src/model_trainer.py`<br>`artifacts/metrics.json` | ROC-AUC: 0.6980, F1: 0.7361, Brier: 0.0761, ECE: 0.0066 (0.66%), MAE: 69.7 min, $R^2$: 0.942. Evaluator in `ModelQualityEvaluator`. | **COMPLETED** |
| **REQ-8b** | Define and measure at least 2 Data Quality metrics (schema validation, missing value rate, drift detection). | `src/metrics_evaluator.py`<br>`src/data_ingestion.py`<br>`tests/test_data_validation.py` | Missing value rate (0.0031), duplicate rate (0.0), range violations (0.0), covariate drift via Population Stability Index (PSI). | **COMPLETED** |
| **REQ-9** | Production experimentation approach (shadow deployment, canary) & ML security considerations. | `PRODUCTION_EXPERIMENTATION_AND_SECURITY.md` | Comprehensive architectural guide detailing asynchronous traffic mirroring for Shadow Deployment, gated Canary progression, Pydantic defense against adversarial inputs, and model integrity. | **COMPLETED** |

---

## 3. Test Suite Summary

Executed command:
```bash
pytest -v
```

**Results:**
```
tests/test_api.py::test_health_check_endpoint PASSED                     [  4%]
tests/test_api.py::test_model_info_endpoint PASSED                       [  8%]
tests/test_api.py::test_predict_endpoint_valid_request PASSED            [ 12%]
tests/test_api.py::test_predict_endpoint_custom_routes PASSED            [ 16%]
tests/test_api.py::test_recommend_route_endpoint PASSED                  [ 20%]
tests/test_api.py::test_recommend_route_with_constraints PASSED          [ 24%]
tests/test_api.py::test_api_validation_errors PASSED                     [ 28%]
tests/test_api.py::test_openapi_schema_availability PASSED               [ 32%]
tests/test_data_validation.py::test_data_ingestion_and_schema PASSED     [ 36%]
tests/test_data_validation.py::test_missing_and_corrupt_data_detection PASSED [ 40%]
tests/test_data_validation.py::test_stratified_splits PASSED             [ 44%]
tests/test_data_validation.py::test_data_quality_evaluator_and_drift PASSED [ 48%]
tests/test_decision_engine.py::test_recommend_prediction_mode PASSED     [ 52%]
tests/test_decision_engine.py::test_recommend_optimisation_mode_with_constraints PASSED [ 56%]
tests/test_decision_engine.py::test_constraints_graceful_relaxation PASSED [ 60%]
tests/test_decision_engine.py::test_excluded_routes_constraint PASSED    [ 64%]
tests/test_decision_engine.py::test_empty_candidates_error PASSED        [ 68%]
tests/test_integration.py::test_end_to_end_pipeline_integration PASSED   [ 72%]
tests/test_model_inference.py::test_inference_output_shape_and_range PASSED [ 76%]
tests/test_model_inference.py::test_directional_test_historical_success_rate PASSED [ 80%]
tests/test_model_inference.py::test_directional_test_temporal_latency_multiplier PASSED [ 84%]
tests/test_model_inference.py::test_invariance_test_transaction_id PASSED [ 88%]
tests/test_model_inference.py::test_invalid_input_rejection PASSED       [ 92%]
tests/test_model_training.py::test_overfitting_on_small_batch PASSED     [ 96%]
tests/test_model_training.py::test_model_training_pipeline_execution PASSED [100%]

======================== 25 passed in 2.90s ========================
```
