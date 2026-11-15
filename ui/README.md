# 🚀 RouteFlow AI — Interactive Web Interfaces & Visual Pipeline

This directory contains the interactive presentation interface and operational dashboard for the **RouteFlow AI Cross-Border Payment Routing ML Engine**.

---

## 🌟 Available Interfaces

### 1. 📖 Pipeline & Corridor Flow: [`/demo`](http://localhost:8000/demo) (or `http://localhost:3000/demo`)
- **Technical & Accessible Architecture Walkthrough**: Explains Nostro/Vostro correspondent banking ledgers, why multi-hop payment routing exists, and how bilateral clearing networks operate.
- **Live API Integration**: Dispatches live HTTP requests to `/recommend-route` and displays exact server inference and network round-trip latency.
- **Realistic Loading State**: Visual buffer and shimmer animation while awaiting model inference.
- **Stage 5 Collapsible Route Comparison**: Winning route highlighted alongside an expandable accordion comparing all evaluated candidate routes.
- **Dark / Light Theme Toggle**: Seamless switch between dark mode and a crisp white theme.

### 2. ⚡ Operational Routing Dashboard: [`/`](http://localhost:8000/) (or `http://localhost:3000/`)
- **Live ML Inference Telemetry**: Displays exact inference duration in milliseconds (`inference_ms` + roundtrip).
- **One-Click Presets**: Test US → India ($5k B2B), UK → Nigeria (£12.5k P2P), US → Mexico ($25k high-value), etc.
- **Multi-Objective Policy Selector**: Switch between Balanced, Ultra-Fast (Speed), and Lowest Cost optimization.
- **Unified Pipeline Stages**: Unified status indicators across all 5 architecture stages (`✓ 01.` to `✓ 05.`).
- **Dark / Light Theme Toggle**: Integrated theme switch on the top bar.

---

## 🖥️ Running the Application

### Option A: Direct FastAPI Integration (Recommended)
When starting the FastAPI backend, the web interfaces are mounted automatically:
```bash
python main.py
```
Visit:
- **Corridor Walkthrough**: [http://localhost:8000/demo](http://localhost:8000/demo)
- **Operational Dashboard**: [http://localhost:8000/](http://localhost:8000/)
- **Swagger REST Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Option B: Dedicated Showcase Server
Alternatively, serve the static UI independently on port 3000:
```bash
python ui/server.py
```
