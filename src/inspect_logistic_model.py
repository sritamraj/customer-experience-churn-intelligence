from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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


def main():

    print("=" * 70)
    print("LOGISTIC REGRESSION — FEATURE COEFFICIENT AUDIT")
    print("=" * 70)

    train = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight=None,
                random_state=42
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    classifier = model.named_steps["model"]

    coefficients = classifier.coef_[0]

    results = pd.DataFrame({
        "feature": FEATURES,
        "coefficient": coefficients,
        "odds_ratio": np.exp(coefficients),
        "abs_coefficient": np.abs(coefficients)
    })

    results = results.sort_values(
        "abs_coefficient",
        ascending=False
    )

    print("\nFeature importance by absolute coefficient:\n")

    print(
        results[
            [
                "feature",
                "coefficient",
                "odds_ratio"
            ]
        ].to_string(
            index=False
        )
    )

    print("\nTop positive signals:")

    print(
        results
        .sort_values(
            "coefficient",
            ascending=False
        )
        .head(5)
        [
            [
                "feature",
                "coefficient",
                "odds_ratio"
            ]
        ]
        .to_string(index=False)
    )

    print("\nTop negative signals:")

    print(
        results
        .sort_values(
            "coefficient",
            ascending=True
        )
        .head(5)
        [
            [
                "feature",
                "coefficient",
                "odds_ratio"
            ]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()