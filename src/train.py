"""
train.py  -  ML Training Pipeline (offline component of the architecture)
-------------------------------------------------------------------------
Trains the two prediction models from the GR4ML Analytics Design View:
  Prediction Goal 1: payment_success          -> HistGradientBoostingClassifier
  Prediction Goal 2: settlement_time_seconds   -> HistGradientBoostingRegressor (log-target)
Also trains a baseline for each so improvement is measurable, writes
metrics to artifacts/metrics.json and figures to figures/.
"""

import json
import os
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline

from data_prep import (
    FEATURES,
    build_preprocessor,
    load_raw,
    make_splits,
    split_features_targets,
    validate,
)

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(ART, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def clf_metrics(y, p, proba):
    y = np.asarray(y)
    p = np.asarray(p)
    fail_recall = ((p == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "brier": brier_score_loss(y, proba),
        "log_loss": log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6)),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "failure_recall": fail_recall,
        "accuracy": accuracy_score(y, p),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p),
        "f1": f1_score(y, p),
    }


def reg_metrics(y, pred):
    return {
        "mae_seconds": mean_absolute_error(y, pred),
        "rmse_seconds": root_mean_squared_error(y, pred),
        "r2": r2_score(y, pred),
        "mae_minutes": mean_absolute_error(y, pred) / 60.0,
    }


def main(data_path="../data.csv"):
    t0 = time.time()
    df = load_raw(data_path)
    quality = validate(df)
    train_df, val_df, test_df = make_splits(df)
    Xtr, ytr_c, ytr_r = split_features_targets(train_df)
    Xva, yva_c, yva_r = split_features_targets(val_df)
    Xte, yte_c, yte_r = split_features_targets(test_df)

    # ---------------- Prediction Goal 1: payment success -----------------
    clf = Pipeline(
        [
            ("prep", build_preprocessor()),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.03,
                    max_iter=500,
                    max_leaf_nodes=7,
                    min_samples_leaf=200,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=42,
                ),
            ),
        ]
    )
    t = time.time()
    clf.fit(Xtr, ytr_c)
    clf_train_time = time.time() - t
    base_clf = Pipeline(
        [("prep", build_preprocessor()), ("model", LogisticRegression(max_iter=2000))]
    ).fit(Xtr, ytr_c)
    majority = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr_c)

    # Decision threshold tuned on the VALIDATION split (maximise balanced accuracy);
    # 0.5 is meaningless with a 91% success base-rate.
    proba_va = clf.predict_proba(Xva)[:, 1]
    ths = np.arange(0.60, 0.98, 0.005)
    bas = [balanced_accuracy_score(yva_c, (proba_va >= t).astype(int)) for t in ths]
    threshold = float(ths[int(np.argmax(bas))])

    res_clf = {}
    for name, m in [
        ("gbdt_classifier", clf),
        ("logistic_baseline", base_clf),
        ("majority_baseline", majority),
    ]:
        proba = m.predict_proba(Xte)[:, 1]
        res_clf[name] = clf_metrics(yte_c, (proba >= threshold).astype(int), proba)
    # heuristic baseline: rank by historical_route_success_rate only
    hist_sr = Xte["historical_route_success_rate"].fillna(
        Xtr["historical_route_success_rate"].median()
    )
    res_clf["historical_success_rate_baseline"] = {
        "roc_auc": roc_auc_score(yte_c, hist_sr),
        "brier": brier_score_loss(yte_c, hist_sr),
    }
    proba_te = clf.predict_proba(Xte)[:, 1]
    cm = confusion_matrix(yte_c, (proba_te >= threshold).astype(int))

    # ---------------- Prediction Goal 2: settlement time -----------------
    # Expected settlement time is modelled for completed payments (the decision
    # engine already receives the failure probability separately, so mixing
    # 48-hour time-out rows into the regression would corrupt the signal).
    ok_tr = ytr_c == 1
    ok_te = yte_c == 1
    reg = Pipeline(
        [
            ("prep", build_preprocessor()),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=600,
                    max_leaf_nodes=31,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=42,
                ),
            ),
        ]
    )
    t = time.time()
    reg.fit(Xtr[ok_tr], np.log1p(ytr_r[ok_tr]))
    reg_train_time = time.time() - t
    base_reg = DummyRegressor(strategy="median").fit(Xtr[ok_tr], ytr_r[ok_tr])
    # "historical latency" heuristic baseline = just use historical_avg_latency_seconds
    hist_pred = Xte.loc[ok_te, "historical_avg_latency_seconds"].fillna(
        Xtr["historical_avg_latency_seconds"].median()
    )

    pred_te = np.expm1(reg.predict(Xte[ok_te]))
    res_reg = {
        "gbdt_regressor": reg_metrics(yte_r[ok_te], pred_te),
        "historical_latency_baseline": reg_metrics(yte_r[ok_te], hist_pred),
        "median_baseline": reg_metrics(yte_r[ok_te], base_reg.predict(Xte[ok_te])),
    }

    # ---------------- Latency measurement (quality requirement #2) -------
    sample = Xte.iloc[:1]
    times = []
    for _ in range(200):
        s = time.perf_counter()
        clf.predict_proba(sample)
        reg.predict(sample)
        times.append((time.perf_counter() - s) * 1000)
    latency = {
        "single_request_p50_ms": float(np.percentile(times, 50)),
        "single_request_p95_ms": float(np.percentile(times, 95)),
        "single_request_p99_ms": float(np.percentile(times, 99)),
    }
    s = time.perf_counter()
    clf.predict_proba(Xte.iloc[:6])
    reg.predict(Xte.iloc[:6])
    latency["six_route_batch_ms"] = (time.perf_counter() - s) * 1000

    # ---------------- Feature importance ----------------------------------
    pi = permutation_importance(
        clf, Xte.iloc[:3000], yte_c.iloc[:3000], n_repeats=5, random_state=42, scoring="roc_auc"
    )
    imp_clf = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)
    pir = permutation_importance(
        reg,
        Xte[ok_te].iloc[:3000],
        np.log1p(yte_r[ok_te].iloc[:3000]),
        n_repeats=5,
        random_state=42,
        scoring="r2",
    )
    imp_reg = pd.Series(pir.importances_mean, index=FEATURES).sort_values(ascending=False)

    # ---------------- Figures ---------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    imp_clf.plot.barh(ax=ax[0], color="#2a6f97")
    ax[0].invert_yaxis()
    ax[0].set_title("Permutation importance - success classifier (ROC AUC drop)")
    imp_reg.plot.barh(ax=ax[1], color="#e76f51")
    ax[1].invert_yaxis()
    ax[1].set_title("Permutation importance - settlement-time regressor (R² drop)")
    plt.tight_layout()
    plt.savefig(f"{FIG}/feature_importance.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    fpr, tpr, _ = roc_curve(yte_c, proba_te)
    ax[0].plot(fpr, tpr, label=f"GBDT AUC={res_clf['gbdt_classifier']['roc_auc']:.3f}")
    fpr2, tpr2, _ = roc_curve(yte_c, base_clf.predict_proba(Xte)[:, 1])
    ax[0].plot(fpr2, tpr2, "--", label=f"LogReg AUC={res_clf['logistic_baseline']['roc_auc']:.3f}")
    ax[0].plot([0, 1], [0, 1], "k:")
    ax[0].set_xlabel("FPR")
    ax[0].set_ylabel("TPR")
    ax[0].set_title("ROC - payment_success")
    ax[0].legend()
    ax[1].imshow(cm, cmap="Blues")
    ax[1].set_title(f"Confusion matrix (test, tuned threshold {threshold:.3f})")
    for i in range(2):
        for j in range(2):
            ax[1].text(j, i, cm[i, j], ha="center", va="center", color="red", fontsize=13)
    ax[1].set_xticks([0, 1])
    ax[1].set_yticks([0, 1])
    ax[1].set_xticklabels(["Pred fail", "Pred success"])
    ax[1].set_yticklabels(["Actual fail", "Actual success"])
    plt.tight_layout()
    plt.savefig(f"{FIG}/classification_eval.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(yte_r[ok_te] / 3600, pred_te / 3600, s=4, alpha=0.3)
    lim = [0, max(yte_r[ok_te].max(), pred_te.max()) / 3600]
    ax.plot(lim, lim, "r--")
    ax.set_xlabel("Actual settlement time (hours)")
    ax.set_ylabel("Predicted (hours)")
    ax.set_title("Settlement-time regressor: actual vs predicted (test)")
    plt.tight_layout()
    plt.savefig(f"{FIG}/regression_eval.png", dpi=150)
    plt.close()

    # Calibration: reliability diagram + predicted-vs-actual success rate per route
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    frac, mean_pred = calibration_curve(yte_c, proba_te, n_bins=10, strategy="quantile")
    ax[0].plot(mean_pred, frac, "o-", label="GBDT classifier")
    ax[0].plot([0.7, 1], [0.7, 1], "k--", label="perfect calibration")
    ax[0].set_xlabel("Mean predicted success probability")
    ax[0].set_ylabel("Observed success rate")
    ax[0].set_title("Reliability diagram (test)")
    ax[0].legend()
    tmp = (
        Xte.assign(proba=proba_te, actual=yte_c.values)
        .groupby("route_id")[["proba", "actual"]]
        .mean()
    )
    tmp.plot.bar(ax=ax[1], color=["#2a6f97", "#90be6d"])
    ax[1].set_ylim(0.7, 1.0)
    ax[1].set_title("Mean predicted vs actual success rate per route (test)")
    ax[1].set_ylabel("rate")
    ax[1].tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(f"{FIG}/route_calibration.png", dpi=150)
    plt.close()
    calibration_error = float(np.mean(np.abs(frac - mean_pred)))

    # ---------------- Persist artifacts ----------------------------------
    joblib.dump(clf, f"{ART}/success_classifier.joblib")
    json.dump({"decision_threshold": threshold}, open(f"{ART}/threshold.json", "w"))
    joblib.dump(reg, f"{ART}/settlement_regressor.joblib")
    route_catalog = (
        df[
            [
                "route_id",
                "sending_institution",
                "receiving_institution",
                "route_cost",
                "historical_route_success_rate",
                "historical_avg_latency_seconds",
            ]
        ]
        .dropna()
        .groupby("route_id")
        .first()
        .reset_index()
    )
    route_catalog.to_json(f"{ART}/route_catalog.json", orient="records", indent=2)
    metrics = {
        "data_quality": quality,
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "classification": res_clf,
        "confusion_matrix": cm.tolist(),
        "decision_threshold": threshold,
        "expected_calibration_error": calibration_error,
        "regression_success_only": res_reg,
        "latency": latency,
        "training_time_seconds": {"classifier": clf_train_time, "regressor": reg_train_time},
        "feature_importance_classifier": imp_clf.round(5).to_dict(),
        "feature_importance_regressor": imp_reg.round(5).to_dict(),
        "model_version": time.strftime("v%Y%m%d-%H%M%S"),
        "total_pipeline_seconds": time.time() - t0,
    }
    json.dump(metrics, open(f"{ART}/metrics.json", "w"), indent=2, default=float)
    return metrics


if __name__ == "__main__":
    m = main()
    print("=== Classification (test set) ===")
    print(pd.DataFrame(m["classification"]).T.round(4).to_string())
    print(
        "Confusion matrix [[TN,FP],[FN,TP]]:",
        m["confusion_matrix"],
        "| tuned threshold:",
        round(m["decision_threshold"], 3),
        "| ECE:",
        round(m["expected_calibration_error"], 4),
    )
    print("\n=== Regression - settlement time, completed payments (test set) ===")
    print(pd.DataFrame(m["regression_success_only"]).T.round(3).to_string())
    print("\n=== Inference latency ===")
    print(json.dumps(m["latency"], indent=2))
    print("\n=== Top features (classifier) ===")
    print(json.dumps(m["feature_importance_classifier"], indent=2))
    print(
        "\nModel version:",
        m["model_version"],
        "| total pipeline time: %.1fs" % m["total_pipeline_seconds"],
    )
