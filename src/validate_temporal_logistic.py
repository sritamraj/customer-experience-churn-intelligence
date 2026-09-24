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


# ============================================================
# STEP 18 — TEMPORAL ROBUSTNESS VALIDATION
# ============================================================

DATA_DIR = Path("data/model")

TRAIN_PATH = DATA_DIR / "train.csv"
OUTPUT_PATH = DATA_DIR / "logistic_temporal_validation.csv"


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


def build_pipeline():
    """
    Build the same preprocessing/model structure as the
    canonical logistic model, without touching validation
    or final-test data.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate(y_true, probabilities):
    """
    Evaluate ranking/probability quality.

    Accuracy and threshold=0.5 classification metrics are
    intentionally excluded because the target is highly
    imbalanced and the project is fundamentally a ranking
    problem.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    return {
        "log_loss": log_loss(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
    }


def main():

    print("=" * 70)
    print("STEP 18 — TEMPORAL ROBUSTNESS VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load ONLY the development training data
    # --------------------------------------------------------

    print("\nLoading development training data...")

    df = pd.read_csv(TRAIN_PATH)

    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"]
    )

    print(f"Rows: {len(df):,}")

    # --------------------------------------------------------
    # Validate canonical feature contract
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing canonical features: {missing_features}"
        )

    if TARGET not in df.columns:
        raise ValueError(
            f"Missing target column: {TARGET}"
        )

    print(f"Locked feature count: {len(FEATURES)}")

    # --------------------------------------------------------
    # Discover available development snapshots
    # --------------------------------------------------------

    snapshots = sorted(
        df["snapshot_date"].dropna().unique()
    )

    print("\nAvailable development snapshots:")

    for snapshot in snapshots:
        count = int(
            (df["snapshot_date"] == snapshot).sum()
        )

        buyers = int(
            df.loc[
                df["snapshot_date"] == snapshot,
                TARGET,
            ].sum()
        )

        rate = buyers / count if count else np.nan

        print(
            f"  {snapshot.date()} | "
            f"rows={count:,} | "
            f"buyers={buyers:,} | "
            f"rate={rate:.6f}"
        )

    # --------------------------------------------------------
    # Require at least two chronological snapshots
    # --------------------------------------------------------

    if len(snapshots) < 2:
        raise ValueError(
            "Temporal validation requires at least "
            "two development snapshots."
        )

    # --------------------------------------------------------
    # Build expanding-window temporal folds
    #
    # With the current dataset:
    #
    # Fold 1:
    #   Train      2017-09-01
    #   Validate   2017-12-01
    #
    # No future data is used.
    # --------------------------------------------------------

    results = []

    for fold_number in range(1, len(snapshots)):

        train_dates = snapshots[:fold_number]
        validation_date = snapshots[fold_number]

        train_mask = df["snapshot_date"].isin(
            train_dates
        )

        validation_mask = (
            df["snapshot_date"] == validation_date
        )

        train_df = df.loc[train_mask].copy()
        validation_df = df.loc[
            validation_mask
        ].copy()

        print("\n" + "-" * 70)
        print(f"FOLD {fold_number}")
        print("-" * 70)

        print(
            "Training snapshots:",
            ", ".join(
                str(x.date())
                for x in train_dates
            ),
        )

        print(
            "Validation snapshot:",
            validation_date.date(),
        )

        print(
            f"Training rows:   {len(train_df):,}"
        )

        print(
            f"Validation rows: {len(validation_df):,}"
        )

        # ----------------------------------------------------
        # Temporal leakage assertions
        # ----------------------------------------------------

        train_max_date = train_df[
            "snapshot_date"
        ].max()

        validation_min_date = validation_df[
            "snapshot_date"
        ].min()

        if train_max_date >= validation_min_date:
            raise ValueError(
                "Temporal leakage detected: "
                "training data is not strictly earlier "
                "than validation data."
            )

        # ----------------------------------------------------
        # Target sanity checks
        # ----------------------------------------------------

        y_train = train_df[TARGET].astype(int)
        y_validation = validation_df[TARGET].astype(int)

        if y_train.nunique() < 2:
            raise ValueError(
                "Training fold contains only one target class."
            )

        if y_validation.nunique() < 2:
            raise ValueError(
                "Validation fold contains only one target class."
            )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        X_train = train_df[FEATURES]
        X_validation = validation_df[FEATURES]

        print("\nFitting logistic pipeline...")

        pipeline = build_pipeline()

        pipeline.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Predict probabilities
        # ----------------------------------------------------

        probabilities = pipeline.predict_proba(
            X_validation
        )[:, 1]

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = evaluate(
            y_validation,
            probabilities,
        )

        train_buyers = int(y_train.sum())
        validation_buyers = int(y_validation.sum())

        train_rate = (
            train_buyers / len(y_train)
        )

        validation_rate = (
            validation_buyers / len(y_validation)
        )

        result = {
            "fold": fold_number,
            "train_start": train_dates[0].date().isoformat(),
            "train_end": train_dates[-1].date().isoformat(),
            "validation_date": (
                validation_date.date().isoformat()
            ),
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "train_buyers": train_buyers,
            "validation_buyers": validation_buyers,
            "train_purchase_rate": train_rate,
            "validation_purchase_rate": validation_rate,
            "log_loss": metrics["log_loss"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
        }

        results.append(result)

        print("\nResults:")

        print(
            f"Log loss : {metrics['log_loss']:.6f}"
        )

        print(
            f"ROC AUC  : {metrics['roc_auc']:.6f}"
        )

        print(
            f"PR AUC   : {metrics['pr_auc']:.6f}"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("TEMPORAL VALIDATION SUMMARY")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print("\nOutput:")
    print(OUTPUT_PATH)

    print("\nIMPORTANT:")
    print(
        "2018-03-01 validation.csv was NOT loaded."
    )

    print(
        "2018-06-19 test.csv was NOT loaded."
    )

    print(
        "No model selection was performed using the final test."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()