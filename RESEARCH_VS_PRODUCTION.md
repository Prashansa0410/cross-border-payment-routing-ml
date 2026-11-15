# Architectural Evolution: From Research Prototype to Enterprise ML Serving

**System:** RouteFlow AI — Cross-Border Payment Routing ML Engine  
**Domain:** Global Correspondent Banking & Intermediary Settlement Optimization  
**Core Components:** High-Throughput Inference Service, Data Preprocessing Pipeline, and Prescriptive Route Decision Engine  

---

## 1. Overview and Rationale

In financial Machine Learning Engineering, transitioning from research/experimentation to production requires addressing critical software qualities: reliability, maintainability, testability, observability, and defensive error handling.

In this project:
- **Research / Experimentation Environment:** Interactive Jupyter exploratory pipeline (`notebooks/pipeline_walkthrough.ipynb`).
- **Production-Oriented Codebase:** Modular Python packages in `src/` (`src/data_ingestion.py`, `src/feature_pipeline.py`, `src/inference.py`, `src/decision_engine.py`, `src/api/app.py`).

We analyze the concrete architectural and operational differences across five dimensions below.

---

## 2. Side-by-Side Architectural Comparison

| Dimension | Research Prototype (`notebooks/`) | Production Implementation (`src/`) |
|---|---|---|
| **Execution Paradigm** | Sequential, interactive cell execution with global state. | Stateless, thread-safe, modular OOP services. |
| **Data Handling** | Ad-hoc `pd.read_csv("../data.csv")` without type enforcement. | OOP `DataIngestion` with path resolution, file existence checks, and schema validation. |
| **Model Lifecycle** | Models loaded repeatedly or kept in notebook memory cells (`clf = joblib.load(...)`). | Thread-safe Singleton `PaymentPredictor.get_instance()` with cached in-memory pipelines. |
| **Input Validation** | Implicit assumptions; uncaught `KeyError` or missing feature errors if dict keys are missing. | Strict Pydantic v2 schemas (`PaymentInput`, `CandidateRouteInput`), validating data types, numeric ranges ($>0$), and allowable categories. |
| **Error Handling** | Raw Python exceptions causing cell halts or HTTP 500 crashes. | Hierarchical exception handlers converting domain issues into standard HTTP 400/422 responses without leaking internal stack traces. |
| **Logging & Telemetry**| Unstructured `print()` statements mixed with notebook outputs. | Standard Python `logging` with levels (`INFO`, `WARNING`, `ERROR`, `DEBUG`) and structured context. |
| **Testability** | Visual inspection of printed tables and manual cell runs. | Automated `pytest` test suite covering unit tests, property/directional tests, and integration tests. |

---

## 3. Deep-Dive: Specific Component Transitions

### A. Data Ingestion and Validation
* **Notebook (`notebooks/pipeline_walkthrough.ipynb`):**  
  Directly executes:
  ```python
  df = load_raw("../data.csv")
  report = validate(df)
  ```
  If `data.csv` is missing or corrupted, the notebook halts with an uncaught `FileNotFoundError` or pandas parse traceback.
* **Production (`src/data_ingestion.py`):**  
  Encapsulated inside `DataIngestion` and `DataValidator` classes. Verifies path existence defensively, logs row counts and execution metrics via `logger.info`, computes schema conformance, and generates a structured `DataQualityReport` dataclass.

### B. Feature Transformation & Temporal Enrichment
* **Notebook (`notebooks/pipeline_walkthrough.ipynb`):**  
  Manual dictionary concatenation in a list comprehension:
  ```python
  X = pd.DataFrame([{**payment, **r} for r in routes])[FEATURES]
  ```
  If `payment` lacks a required feature, or a route dict has mismatched keys, pandas produces unaligned columns with silent NaNs or crashes downstream during `predict_proba`.
* **Production (`src/feature_pipeline.py` & `src/api/schemas.py`):**  
  `TemporalEnricher.enrich()` extracts `hour_of_day`, `day_of_week`, and `is_weekend` safely with fallback defaults. `FeaturePipeline.prepare_inference_frame()` explicitly checks all required columns, validates data types, and raises descriptive `ValueError` exceptions before any ML model is invoked.

### C. Model Loading & Inference Engine
* **Notebook (`notebooks/pipeline_walkthrough.ipynb`):**  
  Direct script execution:
  ```python
  clf = joblib.load("../artifacts/success_classifier.joblib")
  reg = joblib.load("../artifacts/settlement_regressor.joblib")
  p_success = clf.predict_proba(X)[:, 1]
  t_settle = np.expm1(reg.predict(X))
  ```
  If this code were executed inside a concurrent web server, loading the artifact from disk on every HTTP call would introduce extreme disk I/O bottlenecks and race conditions.
* **Production (`src/inference.py`):**  
  `PaymentPredictor` implements the Singleton pattern with `threading.Lock`. The GBDT classifier and regressor are loaded once at application startup into memory. Inference applies domain range protections (`np.clip(p_success, 0.0, 1.0)` and `np.clip(t_settle, 0.0, None)`), measures inference duration via high-resolution monotonic clocks (`time.perf_counter()`), and returns versioned metadata.

### D. Route Recommendation & Decision Logic
* **Notebook (`notebooks/pipeline_walkthrough.ipynb`):**  
  Calls `recommend(cands, mode=mode, constraints=cons)` and prints results to stdout.
* **Production (`src/decision_engine.py` & `src/api/app.py`):**  
  Exposed via a REST endpoint `POST /recommend-route`. Input constraints and optimization weights are verified by Pydantic validators (preventing negative weights or conflicting bounds). When hard constraints cannot be satisfied, the engine gracefully relaxes constraints to avoid dropped payments and logs a structured warning.

---

## 4. Input Validation and Defensive Error Handling

In production, client requests cannot be trusted:
1. **Adversarial / Malformed Inputs:** The REST API rejects strings in numeric fields, negative amounts (`amount <= 0`), invalid categories (`payment_type="INVALID"`), and corrupt ISO timestamps with HTTP 422 Unprocessable Entity.
2. **Graceful Upstream Degradation:** If the model artifacts cannot be loaded, the system does not crash or display a 500 Internal Server Error traceback to the caller; instead, it returns an HTTP 503 Service Unavailable with a clean JSON payload:
   ```json
   {
     "error": "Service unavailable",
     "message": "Prediction models are currently unavailable"
   }
   ```
3. **Data Protection:** Payment IDs are tokenized/masked in logs, and raw customer PII is never logged at `INFO` or `ERROR` levels.

---

## 5. Consistency Between Research and Production

To prevent model drift between research and production:
1. **Shared Preprocessor Definition:** Both the training pipeline (`src/model_trainer.py`) and the production feature builder (`src/feature_pipeline.py`) use the exact same `ColumnTransformer` specification, categorical features (`CAT_FEATURES`), and numeric scaling parameters.
2. **Unified Versioned Artifacts:** Production loads the serialized artifacts produced by training (`artifacts/success_classifier.joblib`, `artifacts/settlement_regressor.joblib`, `artifacts/threshold.json`, `artifacts/route_catalog.json`).
3. **Reproducible Split Logic:** The stratified splitting logic (70/15/15) in `DataValidator.create_splits()` is fixed by seed (`random_state=42`), ensuring that validation thresholds and evaluation metrics match across both environments.
