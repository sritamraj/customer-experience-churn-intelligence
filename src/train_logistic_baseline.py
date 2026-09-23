import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    "orders_last_30d",
    "orders_last_60d",
    "orders_last_90d",
    "spend_last_30d",
    "spend_last_60d",
    "spend_last_90d",
]

TARGET = "future_purchase_flag"


def evaluate_predictions(y_true, scores, label):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    pr_auc = average_precision_score(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)

    order = np.argsort(-scores)
    y_sorted = y_true[order]

    n = len(y_true)
    total_positives = y_true.sum()
    base_rate = total_positives / n

    print(f"\n{label}")
    print("-" * 70)
    print(f"PR-AUC : {pr_auc:.6f}")
    print(f"ROC-AUC: {roc_auc:.6f}")

    for pct in [1, 5, 10]:
        k = max(1, int(np.ceil(n * pct / 100)))
        positives = y_sorted[:k].sum()

        precision = positives / k
        recall = positives / total_positives if total_positives > 0 else 0
        lift = precision / base_rate if base_rate > 0 else 0

        print(
            f"Top {pct:>2}%: "
            f"precision={precision:.4f} "
            f"recall={recall:.4f} "
            f"lift={lift:.2f}x"
        )

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }


def main():

    print("=" * 70)
    print("LOGISTIC REGRESSION — 22 FEATURE MODEL")
    print("=" * 70)

    train = pd.read_csv("data/model/train.csv")
    validation = pd.read_csv("data/model/validation.csv")
    test = pd.read_csv("data/model/test.csv")

    print("\nFeature count:", len(FEATURES))

    missing = [
        feature
        for feature in FEATURES
        if feature not in train.columns
    ]

    if missing:
        raise ValueError(
            "Missing features from train.csv:\n"
            + "\n".join(missing)
        )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    pipeline = Pipeline(
        steps=[
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
                    class_weight="balanced",
                    C=1.0,
                    random_state=42
                )
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    train_scores = pipeline.predict_proba(X_train)[:, 1]
    validation_scores = pipeline.predict_proba(X_validation)[:, 1]
    test_scores = pipeline.predict_proba(X_test)[:, 1]

    train_metrics = evaluate_predictions(
        y_train,
        train_scores,
        "TRAIN"
    )

    validation_metrics = evaluate_predictions(
        y_validation,
        validation_scores,
        "VALIDATION"
    )

    test_metrics = evaluate_predictions(
        y_test,
        test_scores,
        "TEST"
    )

    # --------------------------------------------------------
    # Coefficient audit
    # --------------------------------------------------------

    model = pipeline.named_steps["model"]

    coefficients = pd.DataFrame({
        "feature": FEATURES,
        "coefficient": model.coef_[0],
    })

    coefficients["odds_ratio"] = np.exp(
        coefficients["coefficient"]
    )

    coefficients["abs_coefficient"] = (
        coefficients["coefficient"].abs()
    )

    coefficients = coefficients.sort_values(
        "abs_coefficient",
        ascending=False
    )

    print("\nCoefficient audit:")
    print(
        coefficients[
            ["feature", "coefficient", "odds_ratio"]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Save model scores
    # --------------------------------------------------------

    validation_scored = validation.copy()
    validation_scored["model_score"] = validation_scores

    test_scored = test.copy()
    test_scored["model_score"] = test_scores

    validation_scored.to_csv(
        "data/model/logistic_validation_scored.csv",
        index=False
    )

    test_scored.to_csv(
        "data/model/logistic_test_scored.csv",
        index=False
    )

    print("\nSaved:")
    print("data/model/logistic_validation_scored.csv")
    print("data/model/logistic_test_scored.csv")

    print("\n22-feature Logistic Regression complete.")


if __name__ == "__main__":
    main()