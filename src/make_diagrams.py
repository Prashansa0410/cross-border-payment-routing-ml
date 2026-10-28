"""Generates all diagrams for the report with Graphviz (system architecture, pattern views, GR4ML views)."""

import os

from graphviz import Digraph

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
ML = {
    "style": "filled,rounded",
    "fillcolor": "#d7ecf6",
    "color": "#2a6f97",
    "shape": "box",
    "penwidth": "1.6",
}
NONML = {
    "style": "filled,rounded",
    "fillcolor": "#fdf1d6",
    "color": "#b8860b",
    "shape": "box",
    "penwidth": "1.6",
}
DATA = {"style": "filled", "fillcolor": "#e8e8e8", "color": "#555555", "shape": "cylinder"}
EXT = {"style": "filled", "fillcolor": "#ffffff", "color": "#333333", "shape": "box"}
F = {"fontname": "Helvetica", "fontsize": "10"}

# ------------------------------------------------------------------ 1. System architecture
g = Digraph("arch", format="png")
g.attr(rankdir="LR", nodesep="0.35", ranksep="0.6", **F, bgcolor="white")
g.attr("node", **F)
g.attr("edge", fontname="Helvetica", fontsize="8")
g.node("client", "Payment Ops Team /\nClient Application", **EXT)
with g.subgraph(name="cluster_online") as c:
    c.attr(
        label="ONLINE SERVING  (runtime - one OS process per service)",
        style="dashed",
        color="#888888",
        fontsize="11",
        fontname="Helvetica-Bold",
    )
    c.node(
        "gw",
        "API Gateway /\nPayment Orchestrator\n:8000\n(validate, enrich temporal\nfeatures, publish events)",
        **NONML,
    )
    c.node(
        "dec",
        "Route Decision Service\n:8002\nroute catalogue +\nconstraints + weighted\nmulti-objective scoring",
        **NONML,
    )
    c.node(
        "pred",
        "Prediction Service\n:8001\nGBDT success classifier\nGBDT settlement regressor\nshared preprocessing pipeline",
        **ML,
    )
    c.node("models", "Model artifacts\n(.joblib, threshold,\nmetrics, version)", **DATA)
    c.node("catalog", "Route catalogue\n(route_catalog.json)", **DATA)
with g.subgraph(name="cluster_events") as c:
    c.attr(
        label="EVENT BACKBONE",
        style="dashed",
        color="#888888",
        fontsize="11",
        fontname="Helvetica-Bold",
    )
    c.node(
        "broker",
        "Event Broker :8010\ntopics: payment.requested\nprediction.completed\nroute.recommended\nservice.error",
        shape="box",
        style="filled",
        fillcolor="#e6e0f8",
        color="#6a4c93",
        penwidth="1.6",
    )
    c.node(
        "audit", "Audit & Monitoring\nConsumer :8003\nKPIs, route distribution,\nevent log", **NONML
    )
    c.node("log", "Durable event log\n(event_log.jsonl)", **DATA)
with g.subgraph(name="cluster_offline") as c:
    c.attr(
        label="OFFLINE  ML TRAINING PIPELINE  (batch)",
        style="dashed",
        color="#888888",
        fontsize="11",
        fontname="Helvetica-Bold",
    )
    c.node("raw", "Synthetic transaction\ndataset (50k rows CSV)", **DATA)
    c.node("prep", "Data Preparation\nvalidation, imputation,\nencoding, scaling, split", **ML)
    c.node("train", "Model Training &\nEvaluation\n(threshold tuning,\ncalibration, metrics)", **ML)
g.edge("client", "gw", label="POST /route-payment")
g.edge("gw", "dec", label="POST /recommend")
g.edge("dec", "pred", label="POST /predict\n(6 candidate routes)")
g.edge("pred", "models", dir="back", style="dashed", label="load")
g.edge("dec", "catalog", dir="back", style="dashed")
g.edge("gw", "broker", label="publish events", color="#6a4c93")
g.edge("broker", "audit", label="consume (group=audit)", color="#6a4c93")
g.edge("audit", "log", style="dashed")
g.edge("raw", "prep")
g.edge("prep", "train")
g.edge("train", "models", label="publish model version", style="dashed")
g.edge("train", "catalog", style="dashed")
# legend
with g.subgraph(name="cluster_legend") as c:
    c.attr(label="Legend", fontsize="9", color="#aaaaaa")
    c.node("l1", "ML component", **ML)
    c.node("l2", "Non-ML component", **NONML)
    c.node("l3", "Data / artifact store", **DATA)
g.render(f"{FIG}/system_architecture", cleanup=True)

# ------------------------------------------------------------------ 2. Microservices pattern view
g = Digraph("ms", format="png")
g.attr(rankdir="LR", **F, bgcolor="white")
g.attr("node", **F)
g.attr("edge", fontname="Helvetica", fontsize="8")
g.node("client", "Client", **EXT)
for n, lbl, st in [
    ("gw", "api-gateway\n:8000\nown process\nown config (env URLs)", NONML),
    ("dec", "decision-service\n:8002\nowns: catalogue,\nbusiness rules", NONML),
    ("pred", "prediction-service\n:8001\nowns: models,\nfeature schema", ML),
    ("audit", "audit-consumer\n:8003\nowns: KPIs,\nevent log", NONML),
]:
    g.node(n, lbl, **st)
g.edge("client", "gw", label="HTTP/JSON")
g.edge("gw", "dec", label="HTTP/JSON")
g.edge("dec", "pred", label="HTTP/JSON")
g.edge("gw", "pred", label="/health probe", style="dotted")
g.edge("gw", "audit", label="no direct call\n(decoupled via events)", style="invis")
g.attr(
    label="Microservices pattern: each capability = independently deployable process with its own data, API contract and health endpoint",
    fontsize="10",
    labelloc="b",
)
g.render(f"{FIG}/pattern_microservices", cleanup=True)

# ------------------------------------------------------------------ 3. Event-driven pattern view
g = Digraph("ev", format="png")
g.attr(rankdir="LR", **F, bgcolor="white")
g.attr("node", **F)
g.attr("edge", fontname="Helvetica", fontsize="8")
g.node("gw", "Producer:\napi-gateway", **NONML)
for t in ["payment.requested", "prediction.completed", "route.recommended", "service.error"]:
    g.node(t, f"topic\n{t}", shape="cds", style="filled", fillcolor="#e6e0f8", color="#6a4c93")
    g.edge("gw", t, label="publish")
g.node("audit", "Consumer group 'audit':\naudit-consumer\n(KPIs, event log)", **NONML)
g.node(
    "future",
    "Future consumers\n(fraud screening, billing,\nmodel-drift monitor...)\nadded with no producer change",
    shape="box",
    style="dashed",
)
for t in ["payment.requested", "prediction.completed", "route.recommended", "service.error"]:
    g.edge(t, "audit", label="consume\n(offset per group)", color="#6a4c93")
    g.edge(t, "future", style="dashed", color="#999999")
g.attr(
    label="Event-driven pattern: producers emit immutable domain events to topics; consumers subscribe asynchronously with their own offsets; the log supports replay",
    fontsize="10",
    labelloc="b",
)
g.render(f"{FIG}/pattern_event_driven", cleanup=True)

# ------------------------------------------------------------------ 4. GR4ML Business View
ACTOR = {"shape": "circle", "style": "filled", "fillcolor": "#f8f8f8", "width": "0.9"}
GOAL = {"shape": "ellipse", "style": "filled", "fillcolor": "#fff3b0"}  # strategic / decision goals
QGOAL = {"shape": "ellipse", "style": "filled", "fillcolor": "#ffe8cc"}  # question goals
INSIGHT = {"shape": "note", "style": "filled", "fillcolor": "#d7ecf6"}
INDIC = {"shape": "hexagon", "style": "filled", "fillcolor": "#d8f3dc"}
SOFT = {"shape": "egg", "style": "filled,dashed", "fillcolor": "#f1e4f3"}
g = Digraph("bv", format="png")
g.attr(rankdir="TB", **F, bgcolor="white", nodesep="0.3", ranksep="0.45")
g.attr("node", **F)
g.attr("edge", fontname="Helvetica", fontsize="8")
g.node("actor", "Payment\nOperations\nTeam", **ACTOR)
g.node("sg", "Strategic Goal\nTransfer money efficiently\nand reliably across borders", **GOAL)
g.node("dg", "Decision Goal\nSelect the most appropriate\nprocessing route per payment", **GOAL)
g.node("q1", "Question Goal 1\nWhat is the probability the\npayment succeeds on route r?", **QGOAL)
g.node("q2", "Question Goal 2\nHow long will the payment\ntake to settle on route r?", **QGOAL)
g.node(
    "q3",
    "Question Goal 3\nWhich route gives the best\nsuccess / latency / cost trade-off?",
    **QGOAL,
)
g.node("i1", "Insight\nPredicted success\nprobability per route", **INSIGHT)
g.node("i2", "Insight\nPredicted settlement\ntime per route", **INSIGHT)
g.node("i3", "Insight\nRanked routes with\nweighted score & constraints", **INSIGHT)
g.node("k1", "Indicator\nPayment success rate\n(target >= 95% expected)", **INDIC)
g.node("k2", "Indicator\nAverage settlement time\n(target <= 2 h expected)", **INDIC)
g.node("k3", "Indicator\nAverage route cost\nper transaction", **INDIC)
g.node("s1", "Softgoal\nCompliance / reliability", **SOFT)
g.node("s2", "Softgoal\nCost efficiency", **SOFT)
g.edge("actor", "sg", label="wants")
g.edge("sg", "dg", label="refined by")
for q in ["q1", "q2", "q3"]:
    g.edge("dg", q, label="requires answer to")
g.edge("q1", "i1")
g.edge("q2", "i2")
g.edge("q3", "i3")
g.edge("i1", "i3", style="dashed")
g.edge("i2", "i3", style="dashed")
g.edge("i1", "k1", label="measured by")
g.edge("i2", "k2", label="measured by")
g.edge("i3", "k3", label="measured by")
g.edge("sg", "s1", style="dashed", label="+")
g.edge("sg", "s2", style="dashed", label="+")
g.render(f"{FIG}/gr4ml_business_view", cleanup=True)

# ------------------------------------------------------------------ 5. GR4ML Analytics Design View
PRED = {"shape": "box", "style": "filled,rounded", "fillcolor": "#d7ecf6"}
PRESC = {"shape": "box", "style": "filled,rounded", "fillcolor": "#fdf1d6"}
ALG = {"shape": "component", "style": "filled", "fillcolor": "#ffffff"}
g = Digraph("adv", format="png")
g.attr(rankdir="TB", **F, bgcolor="white", nodesep="0.3", ranksep="0.45")
g.attr("node", **F)
g.attr("edge", fontname="Helvetica", fontsize="8")
g.node("dg", "Decision Goal\nSelect the most appropriate route", **GOAL)
g.node(
    "pg1",
    "Prediction Goal 1\nPredict P(payment_success = 1 | payment, route)\nType: binary classification (probabilistic)",
    **PRED,
)
g.node(
    "pg2",
    "Prediction Goal 2\nPredict settlement_time_seconds | success\nType: regression (log-target)",
    **PRED,
)
g.node(
    "prg",
    "Prescription Goal\nRecommend route: argmax weighted score\nsubject to cost / latency / P(success) constraints\nModes: prediction | optimisation",
    **PRESC,
)
g.node("a1", "Algorithm\nHistGradientBoostingClassifier\n(+ validation-tuned threshold)", **ALG)
g.node("a2", "Algorithm\nHistGradientBoostingRegressor\n(log1p target)", **ALG)
g.node("a3", "Algorithm\nRule-based multi-objective\nscoring engine (non-ML)", **ALG)
g.node(
    "m1",
    "Quality indicators\nROC-AUC, Brier, log-loss,\ncalibration error, failure recall",
    **INDIC,
)
g.node("m2", "Quality indicators\nMAE, RMSE, R² vs\nhistorical-latency baseline", **INDIC)
g.node(
    "m3", "Quality indicators\nexpected success uplift,\nsettlement-time reduction, cost", **INDIC
)
g.node("s1", "Softgoal\nAccuracy / calibration", **SOFT)
g.node("s2", "Softgoal\nPrediction latency\n(<= 100 ms per request)", **SOFT)
g.node("s3", "Softgoal\nExplainability of\nrecommendation", **SOFT)
g.edge("dg", "prg", label="achieved by")
g.edge("prg", "pg1", label="needs")
g.edge("prg", "pg2", label="needs")
g.edge("pg1", "a1")
g.edge("pg2", "a2")
g.edge("prg", "a3")
g.edge("pg1", "m1", style="dashed")
g.edge("pg2", "m2", style="dashed")
g.edge("prg", "m3", style="dashed")
g.edge("a1", "s1", style="dashed", label="+")
g.edge("a2", "s1", style="dashed", label="+")
g.edge("a1", "s2", style="dashed", label="+")
g.edge("a3", "s3", style="dashed", label="+")
g.render(f"{FIG}/gr4ml_analytics_design_view", cleanup=True)

# ------------------------------------------------------------------ 6. GR4ML Data Preparation View
ENT = {"shape": "box3d", "style": "filled", "fillcolor": "#e8e8e8"}
OP = {"shape": "cds", "style": "filled", "fillcolor": "#d8f3dc"}
g = Digraph("dpv", format="png")
g.attr(rankdir="TB", **F, bgcolor="white", nodesep="0.25", ranksep="0.35")
g.attr("node", **F)
g.attr("edge", fontname="Helvetica", fontsize="8")
with g.subgraph(name="cluster_src") as c:
    c.attr(label="Data sources / entities", style="dashed", color="#888888", rank="same")
    c.node(
        "e1",
        "Payment Transaction\ntransaction_id, timestamp, source/destination\ncountry & currency, amount, payment_type",
        **ENT,
    )
    c.node("e2", "Candidate Route\nroute_id, sending/receiving institution,\nroute_cost", **ENT)
    c.node(
        "e3",
        "Historical Route Performance\nhistorical_route_success_rate,\nhistorical_route_failure_rate,\nhistorical_avg_latency_seconds",
        **ENT,
    )
    c.node("e4", "Outcome (targets)\npayment_success,\nsettlement_time_seconds", **ENT)
g.node("o1", "Validate\nschema, types, duplicates,\ntemporal consistency", **OP)
g.node(
    "o2",
    "Drop identifiers &\nredundant columns\n(transaction_id, timestamp,\nfailure_rate = 1 - success_rate)",
    **OP,
)
g.node("o3", "Impute\ncategorical -> 'MISSING'\nnumeric -> median", **OP)
g.node("o4", "Encode & scale\nOne-hot (8 categorical)\nStandardScaler (7 numeric)", **OP)
g.node(
    "o5", "Derive temporal features\nhour_of_day, day_of_week,\nis_weekend (from timestamp)", **OP
)
g.node(
    "o6",
    "Separate features / targets\n15 features; y_clf, y_reg\nlog1p(y_reg), success-only rows",
    **OP,
)
g.node("o7", "Stratified split\n70% train / 15% val / 15% test", **OP)
g.node(
    "out",
    "Prepared datasets\nfor Prediction Goals 1 & 2",
    shape="folder",
    style="filled",
    fillcolor="#d7ecf6",
)
for e in ["e1", "e2", "e3", "e4"]:
    g.edge(e, "o1")
g.edge("o1", "o2")
g.edge("o2", "o5")
g.edge("o5", "o3")
g.edge("o3", "o4")
g.edge("o4", "o6")
g.edge("o6", "o7")
g.edge("o7", "out")
g.render(f"{FIG}/gr4ml_data_preparation_view", cleanup=True)
print("diagrams written:", sorted(f for f in os.listdir(FIG)))
