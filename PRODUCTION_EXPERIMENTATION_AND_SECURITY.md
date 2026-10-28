# Production Experimentation & Security Architecture

**Course:** AIMLZG546 — Software Engineering for Machine Learning  
**Assignment:** Assignment II (Objective 2, Requirement 9)  
**Group:** 65  
**System:** Cross-Border Payment Routing ML System  

---

## 1. Production Testing & Experimentation: Shadow Deployment Architecture

In mission-critical financial applications such as cross-border payment routing, deploying an untested or retrained ML model directly to live traffic poses significant operational and financial risks (e.g., misrouting high-value transactions, corridor timeouts, SLA penalties).

### A. Recommended Approach: Shadow Deployment (Dark Launching)

To safely evaluate new candidate models in production without risking real customer funds, we employ **Shadow Deployment** leveraging our event-driven backbone:

1. **Traffic Mirroring at the API Gateway:**
   - When a client issues a payment request to the API Gateway (`POST /route-payment`), the production Champion Model ($v_1$) synchronously evaluates candidate corridors and returns the authoritative recommendation to the customer within the 100 ms SLA.
   - Concurrently, the Gateway asynchronously publishes the transaction payload to the event broker topic `payment.requested`.

2. **Shadow Service Execution:**
   - A parallel container running the Challenger/Shadow Model ($v_2$) subscribes to `payment.requested`.
   - The shadow service executes feature preprocessing, inference, and corridor scoring against live production data.
   - The shadow model's decisions are **not** returned to the user or downstream payment rails. Instead, they are published to `shadow.route.recommended`.

3. **Offline Comparison & Safety Gating:**
   - The `audit_consumer` aggregates telemetry from both champion and shadow topics, comparing:
     - Inference latency distributions ($p_{50}, p_{95}, p_{99}$).
     - Predicted success rate uplift and settlement latency differences.
     - Agreement rate (percentage of transactions where shadow and champion select the identical route).
     - Covariate and prediction drift (via Population Stability Index, PSI).

4. **Canary Rollout Progression:**
   - Once the shadow model demonstrates statistical parity or superiority over a 7-day shadow window without latency regressions, traffic is migrated incrementally via a Canary Deployment:
     $$\text{Shadow (0\% live)} \longrightarrow \text{Canary (5\%)} \longrightarrow \text{Canary (25\%)} \longrightarrow \text{Champion (100\%)}$$
   - Automated rollback triggers if payment failure rates spike by $>0.5\%$ or latency exceeds 80 ms.

---

## 2. Security Considerations & Adversarial Defense

### A. Threat Model & Risks
Cross-border payment ML systems face distinct attack surfaces:
1. **Adversarial Input Manipulation:** Manipulating amounts, timestamps, or route parameters to game the decision scoring engine into routing traffic through compromised or higher-fee corridors.
2. **Denial-of-Service (DoS) via Payload Inflation:** Sending oversized strings, extreme numeric magnitudes, or malicious JSON bodies to trigger memory exhaustion during feature scaling.
3. **Model Artifact Tampering / Deserialization Exploits:** Compromising `.joblib` model files to execute arbitrary code upon unpickling.
4. **Data Leakage in Telemetry:** Unintentionally logging customer PII, account identifiers, or transaction values in server logs.

### B. Implemented Safeguards and Defenses

1. **Strict Input Schema Validation (Pydantic v2 Defense):**
   - All REST API entry points (`/predict`, `/recommend-route`) enforce strict typing, boundary limits, and literal category whitelisting:
     - `amount`: Must be strictly positive ($>0.0$) and bounded to realistic financial ranges.
     - `payment_type`: Restricted to `Literal["P2P", "B2B", "C2B", "B2C"]`.
     - Numeric ranges: Historical success rates are validated in $[0.0, 1.0]$.
   - Malformed or out-of-boundary payloads are rejected immediately at the gateway with HTTP 422, shielding ML preprocessing from corrupt inputs.

2. **Domain-Constrained Prediction Clamping:**
   - Even in the event of unexpected covariate shifts, the `PaymentPredictor` enforces mathematical domain clipping:
     ```python
     p_success = np.clip(p_success, 0.0, 1.0)
     t_settle = np.clip(t_settle, 0.0, None)
     ```
   - Prevents negative latencies or nonsensical probabilities from corrupting the multi-objective scoring formula.

3. **Safe Deserialization & Artifact Integrity:**
   - Model loading is restricted to trusted local filesystem paths configured via environment variables.
   - Model versions and parameter schemas are validated against `threshold.json` and `metrics.json` prior to serving traffic.

4. **Information Sanitization & Least-Privilege Telemetry:**
   - The logging configuration (`src/logger.py`) suppresses sensitive transaction amounts, recipient account details, and full customer payloads at `INFO` and `ERROR` levels.
   - Client responses for internal errors return generalized error codes (`{"error": "Internal server error"}`) without exposing internal Python stack traces, file paths, or package versions to external callers.
