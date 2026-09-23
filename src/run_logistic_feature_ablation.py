from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("data/model")


FEATURES_16 = [
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


FEATURES_22 = FEATURES_16 + [
    "orders_last_30d",
    "orders_last_60d",
    "orders_last_90d",
    "spend_last_30d",
    "spend_last_60d",
    "spend_last_90d",
]


TARGET = "future_purchase_flag"


def evaluate(y, score):

    y = np.asarray(y)
    score = np.asarray(score)

    result = {
        "pr_auc": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score),
    }

    base_rate = y.mean()

    order = np.argsort(-score)

    for pct in [1, 5, 10]:

        k = max(1, int(len(y) * pct / 100))

        top_y = y[order[:k]]

        precision = top_y.mean()
        recall = top_y.sum() / y.sum()
        lift = precision / base_rate

        result[f"top{pct}_precision"] = precision
        result[f"top{pct}_recall"] = recall
        result[f"top{pct}_lift"] = lift

    return result


def train_and_evaluate(features, name):

    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")

    X_train = train[features]
    y_train = train[TARGET]

    X_val = validation[features]
    y_val = validation[TARGET]

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

    model.fit(X_train, y_train)

    train_score = model.predict_proba(X_train)[:, 1]
    val_score = model.predict_proba(X_val)[:, 1]

    train_metrics = evaluate(
        y_train,
        train_score
    )

    val_metrics = evaluate(
        y_val,
        val_score
    )

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(f"Feature count: {len(features)}")

    print("\nTRAIN")
    print(
        f"PR-AUC : {train_metrics['pr_auc']:.6f}"
    )
    print(
        f"ROC-AUC: {train_metrics['roc_auc']:.6f}"
    )

    print("\nVALIDATION")
    print(
        f"PR-AUC : {val_metrics['pr_auc']:.6f}"
    )
    print(
        f"ROC-AUC: {val_metrics['roc_auc']:.6f}"
    )

    for pct in [1, 5, 10]:
        print(
            f"Top {pct:>2}%: "
            f"precision="
            f"{val_metrics[f'top{pct}_precision']:.4f} "
            f"recall="
            f"{val_metrics[f'top{pct}_recall']:.4f} "
            f"lift="
            f"{val_metrics[f'top{pct}_lift']:.2f}x"
        )

    return {
        "model": name,
        "feature_count": len(features),
        **{
            f"validation_{k}": v
            for k, v in val_metrics.items()
        },
    }


def main():

    print("=" * 80)
    print("LOGISTIC REGRESSION — FEATURE ABLATION")
    print("=" * 80)

    results = []

    results.append(
        train_and_evaluate(
            FEATURES_16,
            "A_16_original"
        )
    )

    results.append(
        train_and_evaluate(
            FEATURES_22,
            "B_22_all_features"
        )
    )

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("VALIDATION COMPARISON")
    print("=" * 80)

    columns = [
        "model",
        "feature_count",
        "validation_pr_auc",
        "validation_roc_auc",
        "validation_top1_precision",
        "validation_top1_recall",
        "validation_top1_lift",
        "validation_top5_precision",
        "validation_top5_recall",
        "validation_top5_lift",
        "validation_top10_precision",
        "validation_top10_recall",
        "validation_top10_lift",
    ]

    print(
        results_df[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    output_path = (
        DATA_DIR /
        "logistic_feature_ablation_validation.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()