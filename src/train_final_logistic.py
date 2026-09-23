from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("data/model")


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


def evaluate(y, score):

    y = np.asarray(y)
    score = np.asarray(score)

    metrics = {
        "pr_auc": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score),
    }

    base_rate = y.mean()

    order = np.argsort(-score)

    for pct in [1, 5, 10]:

        k = max(
            1,
            int(len(y) * pct / 100)
        )

        top_y = y[order[:k]]

        precision = top_y.mean()
        recall = top_y.sum() / y.sum()
        lift = precision / base_rate

        metrics[f"top{pct}_precision"] = precision
        metrics[f"top{pct}_recall"] = recall
        metrics[f"top{pct}_lift"] = lift

    return metrics


def print_metrics(name, metrics):

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(
        f"PR-AUC : {metrics['pr_auc']:.6f}"
    )

    print(
        f"ROC-AUC: {metrics['roc_auc']:.6f}"
    )

    for pct in [1, 5, 10]:

        print(
            f"Top {pct:>2}%: "
            f"precision="
            f"{metrics[f'top{pct}_precision']:.4f} "
            f"recall="
            f"{metrics[f'top{pct}_recall']:.4f} "
            f"lift="
            f"{metrics[f'top{pct}_lift']:.2f}x"
        )


def main():

    print("=" * 80)
    print("FINAL LOGISTIC REGRESSION — LOCKED 16 FEATURES")
    print("=" * 80)

    train = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    validation = pd.read_csv(
        DATA_DIR / "validation.csv"
    )

    test = pd.read_csv(
        DATA_DIR / "test.csv"
    )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    print(f"\nFeature count: {len(FEATURES)}")

    print(
        f"Train:      {len(train):,} rows | "
        f"positives={y_train.sum():,}"
    )

    print(
        f"Validation: {len(validation):,} rows | "
        f"positives={y_validation.sum():,}"
    )

    print(
        f"Test:       {len(test):,} rows | "
        f"positives={y_test.sum():,}"
    )

    model = Pipeline([
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
                class_weight="balanced",
                C=1.0,
                max_iter=2000,
                random_state=42,
            )
        ),
    ])

    # ----------------------------------------------------------
    # TRAIN
    # ----------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    train_score = model.predict_proba(
        X_train
    )[:, 1]

    validation_score = model.predict_proba(
        X_validation
    )[:, 1]

    test_score = model.predict_proba(
        X_test
    )[:, 1]

    # ----------------------------------------------------------
    # EVALUATION
    # ----------------------------------------------------------

    train_metrics = evaluate(
        y_train,
        train_score
    )

    validation_metrics = evaluate(
        y_validation,
        validation_score
    )

    test_metrics = evaluate(
        y_test,
        test_score
    )

    print_metrics(
        "TRAIN",
        train_metrics
    )

    print_metrics(
        "VALIDATION",
        validation_metrics
    )

    print_metrics(
        "TEST",
        test_metrics
    )

    # ----------------------------------------------------------
    # SAVE TEST SCORES
    # ----------------------------------------------------------

    output = test[
        [
            "snapshot_date",
            "customer_unique_id",
            TARGET,
            "future_delivered_order_count",
            "future_revenue",
            "future_item_revenue",
        ]
    ].copy()

    output["predicted_probability"] = test_score

    output = output.sort_values(
        "predicted_probability",
        ascending=False
    )

    output_path = (
        DATA_DIR /
        "final_logistic_test_scored.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    # ----------------------------------------------------------
    # SAVE METRICS
    # ----------------------------------------------------------

    metrics_rows = []

    for split_name, metrics in [
        ("train", train_metrics),
        ("validation", validation_metrics),
        ("test", test_metrics),
    ]:

        row = {
            "split": split_name,
            "feature_count": len(FEATURES),
            **metrics,
        }

        metrics_rows.append(row)

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    metrics_path = (
        DATA_DIR /
        "final_logistic_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    print("\n" + "=" * 80)
    print("SAVED")
    print("=" * 80)

    print(output_path)
    print(metrics_path)


if __name__ == "__main__":
    main()