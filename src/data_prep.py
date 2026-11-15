"""
data_prep.py  -  GR4ML Data Preparation View implementation
------------------------------------------------------------
Implements the data-preparation operators defined in Objective 1:
  * schema / type / missing-value validation
  * identifier handling (transaction_id excluded)
  * temporal features (hour_of_day, day_of_week, is_weekend already derived; verified against timestamp)
  * feature / target separation
  * train / validation / test split (stratified on payment_success)
  * preprocessing pipeline (imputation + encoding) shared by both models
"""

import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COLS = ["transaction_id", "timestamp"]
CAT_FEATURES = [
    "source_country",
    "destination_country",
    "source_currency",
    "destination_currency",
    "payment_type",
    "route_id",
    "sending_institution",
    "receiving_institution",
]
NUM_FEATURES = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "route_cost",
    "historical_route_success_rate",
    "historical_avg_latency_seconds",
]
# historical_route_failure_rate is exactly 1 - success_rate -> redundant, dropped
DROPPED_REDUNDANT = ["historical_route_failure_rate"]
TARGET_CLF = "payment_success"
TARGET_REG = "settlement_time_seconds"
FEATURES = CAT_FEATURES + NUM_FEATURES


def load_raw(path: str = "data.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        for candidate in [
            path,
            "data.csv",
            "../data.csv",
            os.path.join(os.path.dirname(__file__), "..", "data.csv"),
            os.path.join(os.path.dirname(__file__), "data.csv"),
        ]:
            if os.path.exists(candidate):
                path = candidate
                break
    return pd.read_csv(path)


def validate(df: pd.DataFrame) -> dict:
    """Data quality validation operator. Returns a report dict."""
    expected = set(
        ID_COLS + CAT_FEATURES + NUM_FEATURES + DROPPED_REDUNDANT + [TARGET_CLF, TARGET_REG]
    )
    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "schema_ok": expected == set(df.columns),
        "missing_per_column": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "missing_targets": int(df[[TARGET_CLF, TARGET_REG]].isna().sum().sum()),
        "duplicate_transaction_ids": int(df["transaction_id"].duplicated().sum()),
        "target_balance": {str(k): int(v) for k, v in df[TARGET_CLF].value_counts().items()},
    }
    ts = pd.to_datetime(df["timestamp"])
    report["temporal_consistency_ok"] = bool(
        (ts.dt.hour == df["hour_of_day"]).all()
        and (ts.dt.dayofweek == df["day_of_week"]).all()
        and ((ts.dt.dayofweek >= 5).astype(int) == df["is_weekend"]).all()
    )
    report["amount_range"] = [float(df["amount"].min()), float(df["amount"].max())]
    report["success_rate_range_ok"] = bool(
        df["historical_route_success_rate"].dropna().between(0, 1).all()
    )
    return report


def split_features_targets(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y_clf = df[TARGET_CLF].astype(int)
    y_reg = df[TARGET_REG].astype(float)
    return X, y_clf, y_reg


def build_preprocessor() -> ColumnTransformer:
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    num_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("cat", cat_pipe, CAT_FEATURES),
            ("num", num_pipe, NUM_FEATURES),
        ]
    )


def make_splits(df: pd.DataFrame, seed: int = 42):
    """70 / 15 / 15 stratified split on payment_success."""
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df[TARGET_CLF]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df[TARGET_CLF]
    )
    return train_df, val_df, test_df


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "../data.csv"
    df = load_raw(path)
    rep = validate(df)
    print(json.dumps(rep, indent=2))
    tr, va, te = make_splits(df)
    print(f"train={len(tr)}  val={len(va)}  test={len(te)}")
