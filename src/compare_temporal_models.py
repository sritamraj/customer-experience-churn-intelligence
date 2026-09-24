from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# ================================================================
# STEP 19 — CONTROLLED TEMPORAL MODEL COMPARISON
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "model"
OUTPUT_FILE = DATA_DIR / "temporal_model_comparison.csv"


# ------------------------------------------------
# Canonical temporal development split
# ------------------------------------------------

TRAIN_SNAPSHOT = pd.Timestamp("2017-09-01")
VALIDATION_SNAPSHOT = pd.Timestamp("2017-12-01")


# ------------------------------------------------
# Canonical 16 features
# ------------------------------------------------

FEATURES = [
    "recency_days",
    "customer_age_days",
    "frequency",
    "monetary",
    "historical_avg_order_value",
    "historical_avg_review_score",
    "historical_avg_delivery_days",
    "historical_avg_delivery_delay",
    "historical_freight_value",
    "historical_item_count",
    "historical_product_count",
    "historical_seller_count",
    "non_delivered_order_count",
    "avg_freight_per_order",
    "avg_items_per_order",
    "avg_sellers_per_order",
]

TARGET = "future_purchase_flag"


# ------------------------------------------------
# Top-K ranking evaluation
# ------------------------------------------------

def top_k_metrics(y_true, scores, fraction):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    k = max(1, int(np.ceil(len(y_true) * fraction)))

    order = np.argsort(
        -scores,
        kind="mergesort"
    )

    top_y = y_true[order[:k]]

    precision = top_y.mean()

    total_buyers = y_true.sum()

    capture = (
        top_y.sum() / total_buyers
        if total_buyers > 0
        else np.nan
    )

    base_rate = y_true.mean()

    lift = (
        precision / base_rate
        if base_rate > 0
        else np.nan
    )

    return {
        "k": k,
        "buyer_rate": precision,
        "lift": lift,
        "capture": capture,
    }


# ------------------------------------------------
# Evaluate model
# ------------------------------------------------

def evaluate_model(name, y_true, probabilities):
    probabilities = np.asarray(probabilities)

    probabilities = np.clip(
        probabilities,
        1e-15,
        1 - 1e-15,
    )

    result = {
        "model": name,
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1],
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
    }

    for pct in [0.01, 0.05, 0.10, 0.20]:

        metrics = top_k_metrics(
            y_true,
            probabilities,
            pct,
        )

        label = int(pct * 100)

        result[f"top_{label}_buyer_rate"] = metrics["buyer_rate"]
        result[f"top_{label}_lift"] = metrics["lift"]
        result[f"top_{label}_capture"] = metrics["capture"]
        result[f"top_{label}_k"] = metrics["k"]

    return result


# ------------------------------------------------
# Main
# ------------------------------------------------

def main():

    print("=" * 70)
    print("STEP 19 — CONTROLLED TEMPORAL MODEL COMPARISON")
    print("=" * 70)

    print()
    print("IMPORTANT:")
    print("Only train.csv is loaded.")
    print("2018-03-01 validation holdout is NOT loaded.")
    print("2018-06-19 final test is NOT loaded.")
    print()

    # ------------------------------------------------------------
    # Load ONLY the development training table
    # ------------------------------------------------------------

    train = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    train["snapshot_date"] = pd.to_datetime(
        train["snapshot_date"]
    )

    print(
        f"Development rows loaded: {len(train):,}"
    )

    # ------------------------------------------------------------
    # Validate expected snapshots
    # ------------------------------------------------------------

    available_dates = sorted(
        train["snapshot_date"].dropna().unique()
    )

    print()
    print("Available development snapshots:")

    for date in available_dates:
        print(
            f"  {pd.Timestamp(date).date()}"
        )

    assert TRAIN_SNAPSHOT in train["snapshot_date"].values, (
        "Expected training snapshot is missing."
    )

    assert VALIDATION_SNAPSHOT in train["snapshot_date"].values, (
        "Expected temporal validation snapshot is missing."
    )

    # ------------------------------------------------------------
    # Explicit temporal split
    # ------------------------------------------------------------

    train_df = train[
        train["snapshot_date"] == TRAIN_SNAPSHOT
    ].copy()

    validation_df = train[
        train["snapshot_date"] == VALIDATION_SNAPSHOT
    ].copy()

    # ------------------------------------------------------------
    # Leakage guard
    # ------------------------------------------------------------

    assert (
        train_df["snapshot_date"].max()
        < validation_df["snapshot_date"].min()
    )

    assert (
        validation_df["snapshot_date"].min()
        == VALIDATION_SNAPSHOT
    )

    # ------------------------------------------------------------
    # Prepare X / y
    # ------------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in train.columns
    ]

    assert not missing_features, (
        f"Missing canonical features: {missing_features}"
    )

    assert TARGET in train.columns, (
        f"Missing target column: {TARGET}"
    )

    X_train = train_df[FEATURES].copy()
    y_train = train_df[TARGET].astype(int).copy()

    X_validation = validation_df[FEATURES].copy()
    y_validation = validation_df[TARGET].astype(int).copy()

    # ------------------------------------------------------------
    # Basic split audit
    # ------------------------------------------------------------

    print()
    print("TEMPORAL DEVELOPMENT SPLIT")
    print("-" * 70)

    print(
        f"Train date:       {TRAIN_SNAPSHOT.date()}"
    )

    print(
        f"Validation date:  {VALIDATION_SNAPSHOT.date()}"
    )

    print(
        f"Train rows:       {len(X_train):,}"
    )

    print(
        f"Validation rows:  {len(X_validation):,}"
    )

    print(
        f"Train buyers:     {y_train.sum():,}"
    )

    print(
        f"Validation buyers:{y_validation.sum():,}"
    )

    train_rate = y_train.mean()
    validation_rate = y_validation.mean()

    print(
        f"Train rate:       {train_rate:.6f}"
    )

    print(
        f"Validation rate:  {validation_rate:.6f}"
    )

    # ------------------------------------------------------------
    # Model definitions
    # ------------------------------------------------------------

    models = {

        "Logistic Regression": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                )
            ),
        ]),

        "Random Forest": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                )
            ),
        ]),

        "XGBoost": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=42,
                )
            ),
        ]),
    }

    results = []

    # ------------------------------------------------------------
    # Frequency baseline
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("FREQUENCY BASELINE")
    print("=" * 70)

    baseline_probability = np.full(
        len(y_validation),
        train_rate,
        dtype=float,
    )

    baseline_result = evaluate_model(
        "Frequency Baseline",
        y_validation,
        baseline_probability,
    )

    # Top-K ranking is NOT meaningful for a constant score.
    for pct in [1, 5, 10, 20]:
        baseline_result[f"top_{pct}_buyer_rate"] = np.nan
        baseline_result[f"top_{pct}_lift"] = np.nan
        baseline_result[f"top_{pct}_capture"] = np.nan
        baseline_result[f"top_{pct}_k"] = np.nan

    results.append(baseline_result)

    print(
        f"Log loss: {baseline_result['log_loss']:.6f}"
    )

    print(
        f"ROC AUC:  {baseline_result['roc_auc']:.6f}"
    )

    print(
        f"PR AUC:   {baseline_result['pr_auc']:.6f}"
    )

    # ------------------------------------------------------------
    # Train challenger models
    # ------------------------------------------------------------

    for name, model in models.items():

        print()
        print("=" * 70)
        print(f"TRAINING: {name}")
        print("=" * 70)

        model.fit(
            X_train,
            y_train,
        )

        probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        result = evaluate_model(
            name,
            y_validation,
            probabilities,
        )

        results.append(result)

        print(
            f"Log loss: {result['log_loss']:.6f}"
        )

        print(
            f"ROC AUC:  {result['roc_auc']:.6f}"
        )

        print(
            f"PR AUC:   {result['pr_auc']:.6f}"
        )

        for pct in [1, 5, 10, 20]:

            print(
                f"Top {pct:2d}% "
                f"| K={result[f'top_{pct}_k']:6,} "
                f"| BuyerRate="
                f"{result[f'top_{pct}_buyer_rate']:.4f} "
                f"| Lift="
                f"{result[f'top_{pct}_lift']:.2f}x "
                f"| Capture="
                f"{result[f'top_{pct}_capture']:.4f}"
            )

    # ------------------------------------------------------------
    # Save comparison
    # ------------------------------------------------------------

    results_df = pd.DataFrame(results)

    column_order = [
        "model",
        "log_loss",
        "roc_auc",
        "pr_auc",

        "top_1_k",
        "top_1_buyer_rate",
        "top_1_lift",
        "top_1_capture",

        "top_5_k",
        "top_5_buyer_rate",
        "top_5_lift",
        "top_5_capture",

        "top_10_k",
        "top_10_buyer_rate",
        "top_10_lift",
        "top_10_capture",

        "top_20_k",
        "top_20_buyer_rate",
        "top_20_lift",
        "top_20_capture",
    ]

    results_df = results_df[column_order]

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 19 COMPLETE")
    print("=" * 70)

    print()
    print(results_df.to_string(index=False))

    print()
    print(
        f"Saved comparison to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("MODEL-SELECTION RULE:")
    print(
        "Use this development comparison to document "
        "the trade-offs between probability quality "
        "and ranking performance."
    )

    print()
    print(
        "IMPORTANT: This is one temporal development fold. "
        "Do not claim it proves universal model superiority."
    )


if __name__ == "__main__":
    main()
