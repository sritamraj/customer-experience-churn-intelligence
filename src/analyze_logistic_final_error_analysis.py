from pathlib import Path
import sys
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    log_loss,
)


# ============================================================
# STEP 21 — FINAL FROZEN-MODEL ERROR ANALYSIS
# ============================================================

print("=" * 70)
print("STEP 21 — FINAL FROZEN-MODEL ERROR ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data" / "model"
ARTIFACT_DIR = DATA_DIR / "artifacts"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


TEST_PATH = DATA_DIR / "test.csv"
MODEL_PATH = ARTIFACT_DIR / "logistic_model.joblib"
CALIBRATOR_PATH = ARTIFACT_DIR / "sigmoid_calibrator.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"


# New Step-21 artifacts.
SUMMARY_PATH = DATA_DIR / "logistic_final_error_analysis_summary.csv"
TYPE_PATH = DATA_DIR / "logistic_final_error_type_summary.csv"
SEGMENT_PATH = DATA_DIR / "logistic_final_error_segment_analysis.csv"
FN_PATH = DATA_DIR / "logistic_final_high_confidence_false_negatives.csv"
FP_PATH = DATA_DIR / "logistic_final_high_confidence_false_positives.csv"


EXPECTED_MODEL_VERSION = "logistic_16_sigmoid_v1"
EXPECTED_TEST_DATE = pd.Timestamp("2018-06-19")

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


# ------------------------------------------------------------
# LOAD METADATA
# ------------------------------------------------------------

print("\nValidating canonical metadata...")

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

assert metadata["model_version"] == EXPECTED_MODEL_VERSION
assert metadata["feature_count"] == len(FEATURES)
assert metadata["features"] == FEATURES
assert metadata["target"] == TARGET
assert metadata["test_snapshot"] == "2018-06-19"
assert metadata["test_used_for_training"] is False
assert metadata["test_used_for_calibration"] is False

print("Model version:", metadata["model_version"])
print("Feature count:", metadata["feature_count"])
print("Test snapshot:", metadata["test_snapshot"])
print("Test used for training:", metadata["test_used_for_training"])
print("Test used for calibration:", metadata["test_used_for_calibration"])


# ------------------------------------------------------------
# LOAD FROZEN ARTIFACTS
# ------------------------------------------------------------

print("\nLoading frozen artifacts...")

model = joblib.load(MODEL_PATH)
calibrator = joblib.load(CALIBRATOR_PATH)

assert list(model.named_steps.keys()) == ["imputer", "scaler", "model"]

assert hasattr(calibrator, "predict_proba")
assert hasattr(calibrator, "model")

print("Frozen model:", type(model))
print("Pipeline steps:", list(model.named_steps.keys()))
print("Frozen calibrator:", type(calibrator))
print("Calibrator model:", type(calibrator.model))

print("Calibrator coefficient:", calibrator.model.coef_.ravel().tolist())
print("Calibrator intercept:", calibrator.model.intercept_.tolist())


# ------------------------------------------------------------
# LOAD FINAL TEST
# ------------------------------------------------------------

print("\nLoading final test data...")

df = pd.read_csv(TEST_PATH)

required_columns = FEATURES + [TARGET]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

assert not missing_columns, (
    f"Missing required columns: {missing_columns}"
)

assert len(FEATURES) == 16
assert TARGET not in FEATURES

X = df[FEATURES].copy()
y = df[TARGET].astype(int).copy()

if "snapshot_date" in df.columns:
    snapshot_dates = pd.to_datetime(df["snapshot_date"])
    assert snapshot_dates.nunique() == 1
    assert snapshot_dates.iloc[0] == EXPECTED_TEST_DATE
    print("Snapshot:", snapshot_dates.iloc[0].date())
else:
    print("WARNING: snapshot_date column not present in test.csv")


# ------------------------------------------------------------
# FROZEN PREDICTIONS
# ------------------------------------------------------------

print("\nGenerating predictions from frozen artifacts...")

# Raw Logistic probability.
raw_probability = model.predict_proba(X)[:, 1]

# Raw Logistic decision score.
raw_score = model.decision_function(X)

# Frozen sigmoid calibration.
calibrated_probability = calibrator.predict_proba(raw_score)

assert np.isfinite(raw_probability).all()
assert np.isfinite(raw_score).all()
assert np.isfinite(calibrated_probability).all()

assert ((calibrated_probability >= 0.0) &
        (calibrated_probability <= 1.0)).all()

# IMPORTANT:
# No fit() call exists anywhere in this script.


# ------------------------------------------------------------
# FINAL TEST METRICS
# ------------------------------------------------------------

threshold = 0.5

predicted_class = (
    calibrated_probability >= threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y,
    predicted_class,
    labels=[0, 1],
).ravel()

roc_auc = roc_auc_score(y, calibrated_probability)
pr_auc = average_precision_score(y, calibrated_probability)
calibrated_log_loss = log_loss(y, calibrated_probability)

accuracy = (tp + tn) / len(y)

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0.0
)

recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0.0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0.0
)


# ------------------------------------------------------------
# BUILD ANALYSIS TABLE
# ------------------------------------------------------------

analysis = df.copy()

analysis["raw_score"] = raw_score
analysis["raw_probability"] = raw_probability
analysis["calibrated_probability"] = calibrated_probability
analysis["predicted_class"] = predicted_class

analysis["error_type"] = np.select(
    [
        (y == 1) & (predicted_class == 1),
        (y == 0) & (predicted_class == 0),
        (y == 0) & (predicted_class == 1),
        (y == 1) & (predicted_class == 0),
    ],
    [
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
    ],
    default="unknown",
)

analysis["absolute_probability_error"] = (
    np.abs(y.to_numpy() - calibrated_probability)
)

analysis["squared_probability_error"] = (
    (y.to_numpy() - calibrated_probability) ** 2
)


# ------------------------------------------------------------
# RANKING
# ------------------------------------------------------------

analysis["score_rank"] = (
    analysis["calibrated_probability"]
    .rank(method="first", ascending=False)
    .astype(int)
)

analysis["score_percentile"] = (
    analysis["score_rank"] / len(analysis)
)


# ------------------------------------------------------------
# HIGH-CONFIDENCE ERRORS
# ------------------------------------------------------------

# False negatives:
# Actual buyers that received low model probability.
false_negatives = analysis[
    analysis["error_type"] == "false_negative"
].copy()

false_negatives = false_negatives.sort_values(
    "calibrated_probability",
    ascending=True,
)

# False positives:
# Non-buyers receiving the highest model probability.
false_positives = analysis[
    analysis["error_type"] == "false_positive"
].copy()

false_positives = false_positives.sort_values(
    "calibrated_probability",
    ascending=False,
)


# Keep useful diagnostic columns first.
diagnostic_columns = [
    "customer_id",
    "snapshot_date",
    TARGET,
    "raw_score",
    "raw_probability",
    "calibrated_probability",
    "predicted_class",
    "score_rank",
    "score_percentile",
    "absolute_probability_error",
]

available_diagnostic_columns = [
    col for col in diagnostic_columns
    if col in analysis.columns
]

false_negatives[
    available_diagnostic_columns
].head(100).to_csv(
    FN_PATH,
    index=False,
)

false_positives[
    available_diagnostic_columns
].head(100).to_csv(
    FP_PATH,
    index=False,
)


# ------------------------------------------------------------
# ERROR TYPE SUMMARY
# ------------------------------------------------------------

error_type_rows = []

for error_type in [
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
]:

    subset = analysis[
        analysis["error_type"] == error_type
    ]

    count = len(subset)

    error_type_rows.append(
        {
            "error_type": error_type,
            "count": count,
            "share_of_test_pct": (
                count / len(analysis) * 100
            ),
            "mean_calibrated_probability": (
                subset["calibrated_probability"].mean()
                if count
                else np.nan
            ),
            "median_calibrated_probability": (
                subset["calibrated_probability"].median()
                if count
                else np.nan
            ),
            "mean_raw_score": (
                subset["raw_score"].mean()
                if count
                else np.nan
            ),
        }
    )

error_type_summary = pd.DataFrame(error_type_rows)

error_type_summary.to_csv(
    TYPE_PATH,
    index=False,
)


# ------------------------------------------------------------
# BEHAVIOR SEGMENTS
# ------------------------------------------------------------

# These are descriptive diagnostic segments.
# They do not change the model.

segment_definitions = {
    "frequency": "frequency",
    "monetary": "monetary",
    "recency_days": "recency_days",
    "customer_age_days": "customer_age_days",
    "historical_avg_review_score": "historical_avg_review_score",
    "historical_avg_delivery_delay": "historical_avg_delivery_delay",
    "historical_item_count": "historical_item_count",
    "non_delivered_order_count": "non_delivered_order_count",
}

segment_rows = []

for feature_name, source_column in segment_definitions.items():

    if source_column not in analysis.columns:
        continue

    values = analysis[source_column]

    # Only meaningful for numeric variables.
    if not pd.api.types.is_numeric_dtype(values):
        continue

    try:
        quartile = pd.qcut(
            values,
            q=4,
            labels=[
                "Q1_low",
                "Q2",
                "Q3",
                "Q4_high",
            ],
            duplicates="drop",
        )
    except ValueError:
        continue

    temp = analysis.copy()
    temp["_segment"] = quartile

    for segment_name, subset in temp.groupby(
        "_segment",
        observed=False,
    ):

        subset = subset.dropna(
            subset=["_segment"]
        )

        if len(subset) == 0:
            continue

        buyers = int(subset[TARGET].sum())

        predicted_positive = int(
            subset["predicted_class"].sum()
        )

        false_negative_count = int(
            (
                (subset[TARGET] == 1)
                &
                (subset["predicted_class"] == 0)
            ).sum()
        )

        false_positive_count = int(
            (
                (subset[TARGET] == 0)
                &
                (subset["predicted_class"] == 1)
            ).sum()
        )

        segment_rows.append(
            {
                "feature": feature_name,
                "segment": str(segment_name),
                "rows": len(subset),
                "buyers": buyers,
                "buyer_rate": (
                    buyers / len(subset)
                ),
                "mean_calibrated_probability": (
                    subset["calibrated_probability"].mean()
                ),
                "predicted_positive_count": (
                    predicted_positive
                ),
                "false_negative_count": (
                    false_negative_count
                ),
                "false_positive_count": (
                    false_positive_count
                ),
                "false_negative_rate_among_buyers": (
                    false_negative_count / buyers
                    if buyers > 0
                    else np.nan
                ),
                "mean_absolute_probability_error": (
                    subset["absolute_probability_error"].mean()
                ),
            }
        )

segment_analysis = pd.DataFrame(segment_rows)

segment_analysis.to_csv(
    SEGMENT_PATH,
    index=False,
)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

base_rate = y.mean()

summary = pd.DataFrame(
    [
        {
            "model_version": EXPECTED_MODEL_VERSION,
            "test_snapshot": str(
                EXPECTED_TEST_DATE.date()
            ),
            "test_customers": len(y),
            "test_buyers": int(y.sum()),
            "base_rate": base_rate,
            "calibrated_log_loss": calibrated_log_loss,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "threshold": threshold,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": (
                fp / (fp + tn)
                if (fp + tn) > 0
                else 0.0
            ),
            "false_negative_rate": (
                fn / (fn + tp)
                if (fn + tp) > 0
                else 0.0
            ),
            "mean_calibrated_probability": (
                calibrated_probability.mean()
            ),
            "median_calibrated_probability": (
                np.median(calibrated_probability)
            ),
            "max_calibrated_probability": (
                calibrated_probability.max()
            ),
            "min_calibrated_probability": (
                calibrated_probability.min()
            ),
            "false_positive_count": fp,
            "false_negative_count": fn,
        }
    ]
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)


# ------------------------------------------------------------
# CONSOLE OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL FROZEN ERROR ANALYSIS")
print("=" * 70)

print(f"Model version: {EXPECTED_MODEL_VERSION}")
print(f"Test snapshot: {EXPECTED_TEST_DATE.date()}")
print(f"Customers: {len(y):,}")
print(f"Buyers: {int(y.sum()):,}")
print(f"Base rate: {base_rate:.6%}")

print("\nMetrics:")
print(f"Calibrated log loss: {calibrated_log_loss:.6f}")
print(f"ROC AUC:             {roc_auc:.6f}")
print(f"PR AUC:              {pr_auc:.6f}")

print("\nConfusion matrix @ threshold 0.5:")
print(f"TN: {tn:,}")
print(f"FP: {fp:,}")
print(f"FN: {fn:,}")
print(f"TP: {tp:,}")

print("\nClassification metrics:")
print(f"Accuracy:  {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1:        {f1:.6f}")

print("\nProbability distribution:")
print(
    f"Mean calibrated probability: "
    f"{calibrated_probability.mean():.6f}"
)
print(
    f"Median calibrated probability: "
    f"{np.median(calibrated_probability):.6f}"
)
print(
    f"Min calibrated probability: "
    f"{calibrated_probability.min():.6f}"
)
print(
    f"Max calibrated probability: "
    f"{calibrated_probability.max():.6f}"
)

print("\nError counts:")
print(f"False positives: {fp:,}")
print(f"False negatives: {fn:,}")

print("\nSaved artifacts:")
print(SUMMARY_PATH)
print(TYPE_PATH)
print(SEGMENT_PATH)
print(FN_PATH)
print(FP_PATH)

print("\n" + "=" * 70)
print("STEP 21 COMPLETE")
print("=" * 70)