from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "model" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "model" / "validation.csv"

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "data"
    / "model"
    / "artifacts"
)

MODEL_PATH = ARTIFACT_DIR / "logistic_model.joblib"
CALIBRATOR_PATH = ARTIFACT_DIR / "sigmoid_calibrator.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

OUTPUT_DIR = PROJECT_ROOT / "data" / "model"

TARGET = "future_purchase_flag"

EXPECTED_MODEL_VERSION = "logistic_16_sigmoid_v1"


# ============================================================
# LOAD FROZEN PRODUCTION ARTIFACTS
# ============================================================

print("=" * 70)
print("LOGISTIC LOCAL EXPLAINABILITY")
print("=" * 70)

print("\nLoading frozen production artifacts...")

model = joblib.load(MODEL_PATH)
calibrator = joblib.load(CALIBRATOR_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

model_version = metadata["model_version"]
features = metadata["features"]

print(f"Model version: {model_version}")
print(f"Locked feature count: {len(features)}")

if model_version != EXPECTED_MODEL_VERSION:
    raise ValueError(
        f"Unexpected model version: {model_version}"
    )

if len(features) != 16:
    raise ValueError(
        f"Expected 16 locked features, got {len(features)}"
    )


# ============================================================
# FROZEN PIPELINE VALIDATION
# ============================================================

expected_steps = {
    "imputer",
    "scaler",
    "model",
}

actual_steps = set(model.named_steps.keys())

if actual_steps != expected_steps:
    raise ValueError(
        "Frozen model pipeline structure mismatch.\n"
        f"Expected: {expected_steps}\n"
        f"Actual: {actual_steps}"
    )

imputer = model.named_steps["imputer"]
scaler = model.named_steps["scaler"]
classifier = model.named_steps["model"]


# ============================================================
# PROVENANCE CHECK
# ============================================================

print("\n" + "=" * 70)
print("PROVENANCE CHECK")
print("=" * 70)

print("Retrained: NO")
print("Recalibrated: NO")
print("Test data used: NO")


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)

X_train = train[features]
X_validation = validation[features]

y_validation = validation[TARGET].astype(int)

print("\nTrain rows:", len(train))
print("Validation rows:", len(validation))
print("Validation buyers:", int(y_validation.sum()))


# ============================================================
# VERIFY FEATURE CONTRACT
# ============================================================

if list(X_validation.columns) != list(features):
    raise ValueError(
        "Validation feature order does not match locked model features."
    )

if list(X_train.columns) != list(features):
    raise ValueError(
        "Training feature order does not match locked model features."
    )


# ============================================================
# EXTRACT FROZEN MODEL COMPONENTS
# ============================================================

coefficients = classifier.coef_[0]
intercept = classifier.intercept_[0]


if len(coefficients) != len(features):
    raise ValueError(
        "Coefficient count does not match feature count."
    )


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
    print(
        "PASS: frozen-model contributions reconstruct "
        "the model logit."
    )
else:
    raise ValueError(
        "Contribution reconstruction mismatch."
    )


# ============================================================
# FROZEN MODEL PROBABILITIES
# ============================================================

raw_probabilities = model.predict_proba(
    X_validation
)[:, 1]


# ============================================================
# FROZEN SIGMOID CALIBRATION
# ============================================================

calibrated_probabilities = calibrator.predict_proba(
    model_logits
)

print("\n" + "=" * 70)
print("FROZEN SIGMOID CALIBRATION")
print("=" * 70)

print("Calibration model loaded from artifact.")
print("Calibration fit performed here: NO")


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
            features[idx]
            for idx in top_positive
        )
    )

    negative_feature_names.append(
        " | ".join(
            features[idx]
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

    for j, feature in enumerate(features):

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