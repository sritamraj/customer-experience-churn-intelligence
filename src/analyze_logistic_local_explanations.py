from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "model" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "model" / "validation.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "model"

RANDOM_STATE = 42

TARGET = "future_purchase_flag"


# ============================================================
# LOCKED LOGISTIC 16 FEATURES
# ============================================================

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


# ============================================================
# MODEL
# ============================================================

def build_model():

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
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    C=1.0,
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ============================================================
# SIGMOID CALIBRATION
# ============================================================

def sigmoid_calibration(raw_logit):

    # Calibration parameters estimated in Step 9.
    #
    # IMPORTANT:
    # These parameters were fitted using the validation period.
    # Therefore this is an analysis/calibration demonstration,
    # not yet the final production calibration protocol.

    calibration_a = 0.0
    calibration_b = 0.0

    # Placeholder values are intentionally not hard-coded.
    # They will be estimated below from validation data.

    return calibration_a, calibration_b


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOGISTIC LOCAL EXPLAINABILITY")
print("=" * 70)

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)

X_train = train[FEATURES]
y_train = train[TARGET].astype(int)

X_validation = validation[FEATURES]
y_validation = validation[TARGET].astype(int)


print("\nTrain rows:", len(train))
print("Validation rows:", len(validation))
print("Validation buyers:", int(y_validation.sum()))


# ============================================================
# TRAIN LOCKED MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOCKED LOGISTIC 16")
print("=" * 70)

model = build_model()

model.fit(
    X_train,
    y_train,
)


# ============================================================
# EXTRACT PIPELINE COMPONENTS
# ============================================================

imputer = model.named_steps["imputer"]
scaler = model.named_steps["scaler"]
classifier = model.named_steps["classifier"]

coefficients = classifier.coef_[0]
intercept = classifier.intercept_[0]


# ============================================================
# TRANSFORM VALIDATION DATA
# ============================================================

X_validation_imputed = imputer.transform(
    X_validation
)

X_validation_scaled = scaler.transform(
    X_validation_imputed
)


# ============================================================
# LOCAL CONTRIBUTIONS
# ============================================================

contributions = (
    X_validation_scaled
    * coefficients
)


# ============================================================
# VERIFY LOGISTIC MATHEMATICS
# ============================================================

calculated_logits = (
    intercept
    + contributions.sum(axis=1)
)

model_logits = model.decision_function(
    X_validation
)

max_logit_difference = np.max(
    np.abs(
        calculated_logits
        - model_logits
    )
)


print("\n" + "=" * 70)
print("LOCAL EXPLANATION MATHEMATICAL CHECK")
print("=" * 70)

print(
    f"Maximum absolute logit difference: "
    f"{max_logit_difference:.12f}"
)

if max_logit_difference < 1e-8:
    print("PASS: contributions reconstruct the model logit.")
else:
    print("WARNING: contribution reconstruction mismatch.")


# ============================================================
# RAW MODEL SCORE
# ============================================================

raw_probabilities = model.predict_proba(
    X_validation
)[:, 1]


# ============================================================
# FIT PLATT/SIGMOID CALIBRATION
# ============================================================
#
# We use validation raw logits and validation outcomes.
#
# This intentionally reproduces the Step 9 calibration setup.
# It is useful for explainability, but the validation calibration
# is in-sample and therefore optimistic.
#
# A production calibration protocol will be addressed later.
# ============================================================

from scipy.special import expit

calibration_model = LogisticRegression(
    class_weight=None,
    C=1e6,
    max_iter=2000,
    random_state=RANDOM_STATE,
)

calibration_model.fit(
    model_logits.reshape(-1, 1),
    y_validation,
)

calibration_a = calibration_model.coef_[0][0]
calibration_b = calibration_model.intercept_[0]

calibrated_probabilities = expit(
    calibration_a * model_logits
    + calibration_b
)


print("\n" + "=" * 70)
print("SIGMOID CALIBRATION")
print("=" * 70)

print(
    f"Calibration slope: {calibration_a:.8f}"
)

print(
    f"Calibration intercept: {calibration_b:.8f}"
)


# ============================================================
# BASE CUSTOMER TABLE
# ============================================================

result = validation[
    [
        "customer_unique_id",
        "snapshot_date",
        TARGET,
    ]
].copy()

result["raw_model_probability"] = (
    raw_probabilities
)

result["raw_model_logit"] = (
    model_logits
)

result["calibrated_probability"] = (
    calibrated_probabilities
)

result["model_rank"] = (
    result["raw_model_probability"]
    .rank(
        ascending=False,
        method="first",
    )
    .astype(int)
)

result["model_percentile"] = (
    1
    - (
        result["model_rank"] - 1
    )
    / len(result)
)


# ============================================================
# TOP POSITIVE / NEGATIVE CONTRIBUTORS
# ============================================================

positive_feature_names = []
negative_feature_names = []

positive_contribution_values = []
negative_contribution_values = []

for i in range(len(validation)):

    row_contributions = contributions[i]

    positive_indices = np.argsort(
        row_contributions
    )[::-1]

    positive_indices = [
        idx
        for idx in positive_indices
        if row_contributions[idx] > 0
    ]

    negative_indices = np.argsort(
        row_contributions
    )

    negative_indices = [
        idx
        for idx in negative_indices
        if row_contributions[idx] < 0
    ]

    top_positive = positive_indices[:5]
    top_negative = negative_indices[:5]

    positive_feature_names.append(
        " | ".join(
            FEATURES[idx]
            for idx in top_positive
        )
    )

    negative_feature_names.append(
        " | ".join(
            FEATURES[idx]
            for idx in top_negative
        )
    )

    positive_contribution_values.append(
        " | ".join(
            f"{row_contributions[idx]:.6f}"
            for idx in top_positive
        )
    )

    negative_contribution_values.append(
        " | ".join(
            f"{row_contributions[idx]:.6f}"
            for idx in top_negative
        )
    )


result[
    "top_positive_features"
] = positive_feature_names

result[
    "top_negative_features"
] = negative_feature_names

result[
    "top_positive_contributions"
] = positive_contribution_values

result[
    "top_negative_contributions"
] = negative_contribution_values


# ============================================================
# LONG-FORM CONTRIBUTION TABLE
# ============================================================

long_rows = []

for i in range(len(validation)):

    customer_id = validation.iloc[i][
        "customer_unique_id"
    ]

    snapshot_date = validation.iloc[i][
        "snapshot_date"
    ]

    target = int(
        validation.iloc[i][TARGET]
    )

    for j, feature in enumerate(FEATURES):

        value = X_validation_imputed[i, j]

        standardized_value = (
            X_validation_scaled[i, j]
        )

        contribution = (
            contributions[i, j]
        )

        long_rows.append(
            {
                "customer_unique_id": customer_id,
                "snapshot_date": snapshot_date,
                TARGET: target,
                "feature": feature,
                "raw_value": value,
                "standardized_value": standardized_value,
                "coefficient": coefficients[j],
                "contribution_to_logit": contribution,
                "absolute_contribution": abs(
                    contribution
                ),
                "direction": (
                    "positive"
                    if contribution > 0
                    else "negative"
                    if contribution < 0
                    else "neutral"
                ),
            }
        )


long_contributions = pd.DataFrame(
    long_rows
)


# ============================================================
# CONTRIBUTION RANK WITHIN CUSTOMER
# ============================================================

long_contributions[
    "contribution_rank"
] = (
    long_contributions
    .groupby(
        [
            "customer_unique_id",
            "snapshot_date",
        ]
    )[
        "absolute_contribution"
    ]
    .rank(
        ascending=False,
        method="first",
    )
    .astype(int)
)


# ============================================================
# TOP-RISK CUSTOMERS
# ============================================================

top_1_percent_count = max(
    1,
    int(len(result) * 0.01)
)

top_5_percent_count = max(
    1,
    int(len(result) * 0.05)
)

top_10_percent_count = max(
    1,
    int(len(result) * 0.10)
)


top_1 = result[
    result["model_rank"]
    <= top_1_percent_count
].copy()

top_5 = result[
    result["model_rank"]
    <= top_5_percent_count
].copy()

top_10 = result[
    result["model_rank"]
    <= top_10_percent_count
].copy()


# ============================================================
# DISPLAY TOP CUSTOMERS
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 VALIDATION CUSTOMERS")
print("=" * 70)

display_columns = [
    "model_rank",
    "customer_unique_id",
    "snapshot_date",
    TARGET,
    "raw_model_probability",
    "calibrated_probability",
    "top_positive_features",
    "top_negative_features",
]

print(
    result.sort_values(
        "model_rank"
    )[
        display_columns
    ].head(10).to_string(
        index=False
    )
)


# ============================================================
# TOP CONTRIBUTION SUMMARY
# ============================================================

global_local_importance = (
    long_contributions
    .groupby("feature")
    .agg(
        mean_absolute_contribution=(
            "absolute_contribution",
            "mean",
        ),
        mean_contribution=(
            "contribution_to_logit",
            "mean",
        ),
    )
    .reset_index()
)

global_local_importance[
    "importance_rank"
] = (
    global_local_importance[
        "mean_absolute_contribution"
    ]
    .rank(
        ascending=False,
        method="first",
    )
    .astype(int)
)

global_local_importance = (
    global_local_importance
    .sort_values(
        "importance_rank"
    )
)


print("\n" + "=" * 70)
print("AVERAGE ABSOLUTE LOCAL CONTRIBUTION")
print("=" * 70)

print(
    global_local_importance.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


summary_path = (
    OUTPUT_DIR
    / "logistic_validation_local_explanations.csv"
)

long_path = (
    OUTPUT_DIR
    / "logistic_validation_local_contributions.csv"
)

importance_path = (
    OUTPUT_DIR
    / "logistic_local_contribution_importance.csv"
)

top1_path = (
    OUTPUT_DIR
    / "logistic_validation_top1_explanations.csv"
)

top5_path = (
    OUTPUT_DIR
    / "logistic_validation_top5_explanations.csv"
)

top10_path = (
    OUTPUT_DIR
    / "logistic_validation_top10_explanations.csv"
)


result.to_csv(
    summary_path,
    index=False,
)

long_contributions.to_csv(
    long_path,
    index=False,
)

global_local_importance.to_csv(
    importance_path,
    index=False,
)

top_1.to_csv(
    top1_path,
    index=False,
)

top_5.to_csv(
    top5_path,
    index=False,
)

top_10.to_csv(
    top10_path,
    index=False,
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(summary_path)
print(long_path)
print(importance_path)
print(top1_path)
print(top5_path)
print(top10_path)

print("\nLocal explainability analysis complete.")