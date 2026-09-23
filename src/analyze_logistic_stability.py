from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss, roc_auc_score


# ============================================================
# STEP 14D — LOCKED-MODEL SNAPSHOT + SEGMENT STABILITY
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "data" / "model"
ARTIFACT_DIR = MODEL_DIR / "artifacts"

VALIDATION_PATH = MODEL_DIR / "validation.csv"

MODEL_PATH = ARTIFACT_DIR / "logistic_model.joblib"
CALIBRATOR_PATH = ARTIFACT_DIR / "sigmoid_calibrator.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

SNAPSHOT_OUTPUT_PATH = (
    MODEL_DIR / "logistic_snapshot_stability.csv"
)

SEGMENT_OUTPUT_PATH = (
    MODEL_DIR / "logistic_segment_stability.csv"
)

DRIVER_OUTPUT_PATH = (
    MODEL_DIR / "logistic_driver_stability.csv"
)

SUMMARY_OUTPUT_PATH = (
    MODEL_DIR / "logistic_stability_summary.csv"
)


# ============================================================
# LOCKED FEATURES — LOGISTIC 16
# ============================================================

LOCKED_FEATURES = [
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
SNAPSHOT_COLUMN = "snapshot_date"


# ============================================================
# RANKING BANDS
#
# These are EXCLUSIVE bands.
#
# Top 1%
# Top 1–5%
# Top 5–10%
# Top 10–20%
# Bottom 80%
# ============================================================

SEGMENT_ORDER = [
    "top_1pct",
    "top_1_5pct",
    "top_5_10pct",
    "top_10_20pct",
    "bottom_80pct",
]


# ============================================================
# HELPERS
# ============================================================

def assign_exclusive_segments(scores):
    """
    Assign exclusive ranking bands using deterministic stable sorting.
    """

    scores = np.asarray(scores)
    n = len(scores)

    order = np.argsort(
        -scores,
        kind="mergesort",
    )

    segment = np.empty(
        n,
        dtype=object,
    )

    top1_n = max(
        1,
        int(np.ceil(n * 0.01)),
    )

    top5_n = max(
        1,
        int(np.ceil(n * 0.05)),
    )

    top10_n = max(
        1,
        int(np.ceil(n * 0.10)),
    )

    top20_n = max(
        1,
        int(np.ceil(n * 0.20)),
    )

    segment[order[:top1_n]] = "top_1pct"

    segment[
        order[top1_n:top5_n]
    ] = "top_1_5pct"

    segment[
        order[top5_n:top10_n]
    ] = "top_5_10pct"

    segment[
        order[top10_n:top20_n]
    ] = "top_10_20pct"

    segment[
        order[top20_n:]
    ] = "bottom_80pct"

    return segment


def cumulative_top_k_metrics(y_true, scores, fraction):
    """
    Calculate cumulative Top-K performance.

    Example:
        fraction=0.05 means cumulative Top 5%,
        not the exclusive 1–5% band.
    """

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    n = len(y_true)

    if n == 0:
        return {
            "customers": 0,
            "buyers": 0,
            "precision": np.nan,
            "capture": np.nan,
            "lift": np.nan,
        }

    k = max(
        1,
        int(np.ceil(n * fraction)),
    )

    order = np.argsort(
        -scores,
        kind="mergesort",
    )

    selected = order[:k]

    buyers_selected = int(
        y_true[selected].sum()
    )

    total_buyers = int(
        y_true.sum()
    )

    precision = (
        buyers_selected / k
    )

    capture = (
        buyers_selected / total_buyers
        if total_buyers > 0
        else np.nan
    )

    base_rate = (
        total_buyers / n
    )

    lift = (
        precision / base_rate
        if base_rate > 0
        else np.nan
    )

    return {
        "customers": k,
        "buyers": buyers_selected,
        "precision": precision,
        "capture": capture,
        "lift": lift,
    }


def safe_roc_auc(y_true, scores):
    """
    ROC AUC is undefined when a snapshot contains only one class.
    """

    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(
        y_true,
        scores,
    )


def safe_log_loss(y_true, probabilities):
    """
    Numerically safe calibrated log loss.
    """

    probabilities = np.clip(
        probabilities,
        1e-15,
        1 - 1e-15,
    )

    return log_loss(
        y_true,
        probabilities,
        labels=[0, 1],
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print("STEP 14D — LOCKED-MODEL SNAPSHOT + SEGMENT STABILITY")
print("=" * 78)


# ============================================================
# LOAD CANONICAL ARTIFACTS
# ============================================================

print("\nLoading canonical production artifacts...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Missing canonical model:\n{MODEL_PATH}"
    )

if not CALIBRATOR_PATH.exists():
    raise FileNotFoundError(
        f"Missing canonical calibrator:\n{CALIBRATOR_PATH}"
    )

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Missing model metadata:\n{METADATA_PATH}"
    )

import joblib
import json

model = joblib.load(
    MODEL_PATH
)

calibrator = joblib.load(
    CALIBRATOR_PATH
)

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as f:
    metadata = json.load(f)


# ============================================================
# VERIFY LOCKED MODEL
# ============================================================

print(
    f"Model version: "
    f"{metadata.get('model_version', 'UNKNOWN')}"
)

expected_version = "logistic_16_sigmoid_v1"

if metadata.get("model_version") != expected_version:
    raise ValueError(
        "Unexpected model version. "
        f"Expected {expected_version}, "
        f"found {metadata.get('model_version')}"
    )


if not hasattr(model, "named_steps"):
    raise ValueError(
        "Canonical model does not expose named_steps."
    )


pipeline_steps = list(
    model.named_steps.keys()
)

print(
    f"Pipeline steps: {pipeline_steps}"
)

expected_steps = [
    "imputer",
    "scaler",
    "model",
]

if pipeline_steps != expected_steps:
    raise ValueError(
        "Canonical pipeline structure mismatch.\n"
        f"Expected: {expected_steps}\n"
        f"Found:    {pipeline_steps}"
    )


classifier = model.named_steps["model"]

print(
    f"Classifier: "
    f"{classifier.__class__.__name__}"
)


# ============================================================
# VERIFY LOCKED FEATURES
# ============================================================

metadata_features = metadata.get(
    "features"
)

if metadata_features is not None:

    if metadata_features != LOCKED_FEATURES:
        raise ValueError(
            "Metadata feature list does not match "
            "the locked 16-feature specification."
        )


print(
    f"Locked feature count: "
    f"{len(LOCKED_FEATURES)}"
)


# ============================================================
# LOAD VALIDATION ONLY
# ============================================================

print("\nLoading validation data...")

if not VALIDATION_PATH.exists():
    raise FileNotFoundError(
        f"Missing validation data:\n"
        f"{VALIDATION_PATH}"
    )

validation = pd.read_csv(
    VALIDATION_PATH
)

validation[SNAPSHOT_COLUMN] = pd.to_datetime(
    validation[SNAPSHOT_COLUMN]
)


# ============================================================
# VALIDATE DATA
# ============================================================

required_columns = (
    LOCKED_FEATURES
    + [
        TARGET,
        SNAPSHOT_COLUMN,
    ]
)

missing_columns = [
    column
    for column in required_columns
    if column not in validation.columns
]

if missing_columns:
    raise ValueError(
        "Validation data is missing columns:\n"
        f"{missing_columns}"
    )


if not set(
    validation[TARGET].dropna().unique()
).issubset({0, 1}):

    raise ValueError(
        "Target must contain only binary 0/1 values."
    )


# ============================================================
# BASIC VALIDATION INFORMATION
# ============================================================

y = validation[TARGET].astype(int)

print(
    f"Validation rows: "
    f"{len(validation):,}"
)

print(
    f"Validation buyers: "
    f"{int(y.sum()):,}"
)

print(
    f"Validation base rate: "
    f"{y.mean():.6%}"
)

print(
    f"Validation snapshots: "
    f"{validation[SNAPSHOT_COLUMN].nunique():,}"
)


# ============================================================
# SCORE WITH LOCKED MODEL
#
# IMPORTANT:
# NO FIT
# NO RETRAIN
# NO RECALIBRATION
# ============================================================

print("\nScoring validation with canonical model...")

X = validation[LOCKED_FEATURES]

raw_scores = classifier.decision_function(
    model.named_steps["scaler"].transform(
        model.named_steps["imputer"].transform(X)
    )
)

calibrated_probabilities = calibrator.predict_proba(
    raw_scores
)

validation = validation.copy()

validation["raw_score"] = raw_scores
validation["calibrated_probability"] = (
    calibrated_probabilities
)


# ============================================================
# ASSIGN EXCLUSIVE RANKING SEGMENTS
# ============================================================

validation["segment"] = assign_exclusive_segments(
    validation["calibrated_probability"].to_numpy()
)


# ============================================================
# 14D-A
# SNAPSHOT PERFORMANCE
# ============================================================

snapshot_rows = []

for snapshot_date, group in (
    validation
    .sort_values(SNAPSHOT_COLUMN)
    .groupby(
        SNAPSHOT_COLUMN,
        sort=True,
    )
):

    y_snapshot = group[TARGET].astype(int).to_numpy()

    probabilities = (
        group["calibrated_probability"]
        .to_numpy()
    )

    customers = len(group)

    buyers = int(
        y_snapshot.sum()
    )

    base_rate = (
        buyers / customers
        if customers > 0
        else np.nan
    )

    calibrated_log_loss = safe_log_loss(
        y_snapshot,
        probabilities,
    )

    roc_auc = safe_roc_auc(
        y_snapshot,
        probabilities,
    )

    top1 = cumulative_top_k_metrics(
        y_snapshot,
        probabilities,
        0.01,
    )

    top5 = cumulative_top_k_metrics(
        y_snapshot,
        probabilities,
        0.05,
    )

    top10 = cumulative_top_k_metrics(
        y_snapshot,
        probabilities,
        0.10,
    )

    top20 = cumulative_top_k_metrics(
        y_snapshot,
        probabilities,
        0.20,
    )

    snapshot_rows.append(
        {
            "snapshot_date": snapshot_date,
            "customers": customers,
            "buyers": buyers,
            "base_rate": base_rate,
            "calibrated_log_loss": calibrated_log_loss,
            "roc_auc": roc_auc,
            "mean_predicted_probability": probabilities.mean(),
            "median_predicted_probability": np.median(
                probabilities
            ),

            "top1_precision": top1["precision"],
            "top1_lift": top1["lift"],
            "top1_capture": top1["capture"],

            "top5_precision": top5["precision"],
            "top5_lift": top5["lift"],
            "top5_capture": top5["capture"],

            "top10_precision": top10["precision"],
            "top10_lift": top10["lift"],
            "top10_capture": top10["capture"],

            "top20_precision": top20["precision"],
            "top20_lift": top20["lift"],
            "top20_capture": top20["capture"],
        }
    )


snapshot_results = pd.DataFrame(
    snapshot_rows
).sort_values(
    SNAPSHOT_COLUMN
).reset_index(
    drop=True
)


# ============================================================
# 14D-B
# SEGMENT STABILITY BY SNAPSHOT
# ============================================================

segment_rows = []

for snapshot_date, snapshot_group in (
    validation
    .sort_values(SNAPSHOT_COLUMN)
    .groupby(
        SNAPSHOT_COLUMN,
        sort=True,
    )
):

    base_rate = snapshot_group[TARGET].mean()

    for segment_name in SEGMENT_ORDER:

        segment = snapshot_group[
            snapshot_group["segment"]
            == segment_name
        ]

        customers = len(segment)

        buyers = int(
            segment[TARGET].sum()
        )

        buyer_rate = (
            buyers / customers
            if customers > 0
            else np.nan
        )

        lift = (
            buyer_rate / base_rate
            if base_rate > 0
            else np.nan
        )

        capture = (
            buyers
            / int(snapshot_group[TARGET].sum())
            if int(snapshot_group[TARGET].sum()) > 0
            else np.nan
        )

        segment_rows.append(
            {
                "snapshot_date": snapshot_date,
                "segment": segment_name,
                "customers": customers,
                "buyers": buyers,
                "buyer_rate": buyer_rate,
                "base_rate": base_rate,
                "lift": lift,
                "capture": capture,
                "mean_predicted_probability": (
                    segment[
                        "calibrated_probability"
                    ].mean()
                    if customers > 0
                    else np.nan
                ),
                "median_predicted_probability": (
                    segment[
                        "calibrated_probability"
                    ].median()
                    if customers > 0
                    else np.nan
                ),
            }
        )


segment_results = pd.DataFrame(
    segment_rows
)


# ============================================================
# 14D-C
# LOCKED DRIVER CONTRIBUTION STABILITY
#
# No model is fitted here.
#
# contribution = coefficient × standardized feature value
#
# This is a descriptive decomposition of the existing
# canonical logistic model.
# ============================================================

print("\nCalculating locked-model driver contributions...")

imputer = model.named_steps["imputer"]
scaler = model.named_steps["scaler"]

X_imputed = imputer.transform(
    validation[LOCKED_FEATURES]
)

X_scaled = scaler.transform(
    X_imputed
)

coefficients = classifier.coef_[0]

driver_rows = []

for feature_index, feature in enumerate(
    LOCKED_FEATURES
):

    feature_contribution = (
        X_scaled[:, feature_index]
        * coefficients[feature_index]
    )

    validation_feature_contribution = pd.DataFrame(
        {
            "snapshot_date": validation[
                SNAPSHOT_COLUMN
            ].to_numpy(),

            "segment": validation[
                "segment"
            ].to_numpy(),

            "contribution": feature_contribution,
        }
    )

    for snapshot_date, snapshot_group in (
        validation_feature_contribution
        .groupby(
            "snapshot_date",
            sort=True,
        )
    ):

        contribution_values = (
            snapshot_group["contribution"]
        )

        for segment_name in SEGMENT_ORDER:

            segment_values = contribution_values[
                snapshot_group["segment"]
                == segment_name
            ]

            if len(segment_values) == 0:
                continue

            driver_rows.append(
                {
                    "snapshot_date": snapshot_date,
                    "segment": segment_name,
                    "feature": feature,
                    "coefficient": coefficients[
                        feature_index
                    ],
                    "mean_contribution": (
                        segment_values.mean()
                    ),
                    "median_contribution": (
                        segment_values.median()
                    ),
                    "std_contribution": (
                        segment_values.std()
                    ),
                }
            )


driver_results = pd.DataFrame(
    driver_rows
)


# ============================================================
# 14D-D
# GLOBAL STABILITY SUMMARY
# ============================================================

summary_rows = []

for metric in [
    "base_rate",
    "calibrated_log_loss",
    "roc_auc",
    "top1_lift",
    "top1_capture",
    "top5_lift",
    "top5_capture",
    "top10_lift",
    "top10_capture",
    "top20_lift",
    "top20_capture",
]:

    values = snapshot_results[
        metric
    ].dropna()

    if len(values) == 0:
        continue

    summary_rows.append(
        {
            "metric": metric,
            "mean": values.mean(),
            "std": (
                values.std(ddof=1)
                if len(values) > 1
                else np.nan
            ),
            "min": values.min(),
            "max": values.max(),
            "range": values.max() - values.min(),
        }
    )


summary_results = pd.DataFrame(
    summary_rows
)


# ============================================================
# DISPLAY SNAPSHOT RESULTS
# ============================================================

print("\n" + "=" * 78)
print("14D — SNAPSHOT PERFORMANCE")
print("=" * 78)

display_columns = [
    "snapshot_date",
    "customers",
    "buyers",
    "base_rate",
    "calibrated_log_loss",
    "roc_auc",
    "top1_lift",
    "top1_capture",
    "top5_capture",
    "top10_capture",
    "top20_capture",
]

print(
    snapshot_results[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# DISPLAY SEGMENT RESULTS
# ============================================================

print("\n" + "=" * 78)
print("14D — SEGMENT STABILITY")
print("=" * 78)

print(
    segment_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# DISPLAY STABILITY SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("14D — METRIC STABILITY SUMMARY")
print("=" * 78)

print(
    summary_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# DRIVER STABILITY SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("14D — DRIVER CONTRIBUTION STABILITY")
print("=" * 78)

driver_summary = (
    driver_results
    .groupby(
        ["segment", "feature"],
        as_index=False,
    )
    .agg(
        mean_contribution=(
            "mean_contribution",
            "mean",
        ),
        std_contribution=(
            "mean_contribution",
            "std",
        ),
        min_contribution=(
            "mean_contribution",
            "min",
        ),
        max_contribution=(
            "mean_contribution",
            "max",
        ),
    )
)

print(
    driver_summary
    .sort_values(
        [
            "segment",
            "mean_contribution",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# SAVE ARTIFACTS
# ============================================================

snapshot_results.to_csv(
    SNAPSHOT_OUTPUT_PATH,
    index=False,
)

segment_results.to_csv(
    SEGMENT_OUTPUT_PATH,
    index=False,
)

driver_results.to_csv(
    DRIVER_OUTPUT_PATH,
    index=False,
)

summary_results.to_csv(
    SUMMARY_OUTPUT_PATH,
    index=False,
)


# ============================================================
# FINAL VALIDATION CHECKS
# ============================================================

if len(snapshot_results) == 0:
    raise ValueError(
        "No snapshot stability results were produced."
    )

if len(segment_results) == 0:
    raise ValueError(
        "No segment stability results were produced."
    )

if len(driver_results) == 0:
    raise ValueError(
        "No driver stability results were produced."
    )


print("\n" + "=" * 78)
print("FILES SAVED")
print("=" * 78)

print(SNAPSHOT_OUTPUT_PATH)
print(SEGMENT_OUTPUT_PATH)
print(DRIVER_OUTPUT_PATH)
print(SUMMARY_OUTPUT_PATH)

print("\n" + "=" * 78)
print("STEP 14D COMPLETE")
print("=" * 78)

print(
    "\nIMPORTANT:"
    "\n- Canonical model loaded; no retraining."
    "\n- Canonical sigmoid calibrator loaded; no recalibration."
    "\n- Validation only; test set was not used."
    "\n- Ranking bands are exclusive."
    "\n- Driver contributions use the locked model coefficients."
)