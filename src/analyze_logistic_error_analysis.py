from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_DIR = ROOT / "data" / "model" / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "logistic_model.joblib"
CALIBRATOR_PATH = ARTIFACT_DIR / "sigmoid_calibrator.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

VALIDATION_PATH = ROOT / "data" / "model" / "validation.csv"

OUTPUT_DIR = ROOT / "data" / "model"


# ============================================================
# HELPERS
# ============================================================

def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def require_columns(df, columns, name):
    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 14B — LOCKED-MODEL ERROR ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD CANONICAL ARTIFACTS
# ============================================================

print("\nLoading canonical production artifacts...")

require_file(MODEL_PATH)
require_file(CALIBRATOR_PATH)
require_file(METADATA_PATH)
require_file(VALIDATION_PATH)

model = joblib.load(MODEL_PATH)
calibrator = joblib.load(CALIBRATOR_PATH)

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as f:
    metadata = json.load(f)

print(
    f"Model version: "
    f"{metadata.get('model_version')}"
)

locked_features = metadata.get("features")

if not locked_features:
    raise ValueError(
        "No locked feature list found in model metadata."
    )

print(
    f"Locked feature count: "
    f"{len(locked_features)}"
)

print(
    f"Pipeline steps: "
    f"{list(model.named_steps.keys())}"
)


# ============================================================
# VERIFY CANONICAL PIPELINE
# ============================================================

expected_steps = [
    "imputer",
    "scaler",
    "model",
]

actual_steps = list(model.named_steps.keys())

if actual_steps != expected_steps:
    raise ValueError(
        "Canonical pipeline structure mismatch.\n"
        f"Expected: {expected_steps}\n"
        f"Actual:   {actual_steps}"
    )

classifier = model.named_steps["model"]

print(
    f"Classifier: "
    f"{classifier.__class__.__name__}"
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation data...")

validation = pd.read_csv(
    VALIDATION_PATH
)

TARGET = "future_purchase_flag"

require_columns(
    validation,
    locked_features + [TARGET],
    "Validation data",
)

X_validation = validation[
    locked_features
].copy()

y_validation = validation[
    TARGET
].astype(int)

print(
    f"Validation rows: "
    f"{len(validation):,}"
)

print(
    f"Validation buyers: "
    f"{y_validation.sum():,}"
)

print(
    f"Validation base rate: "
    f"{y_validation.mean():.6f}"
)


# ============================================================
# SCORE WITH LOCKED MODEL
# ============================================================

print(
    "\nScoring validation data with "
    "canonical production model..."
)

# IMPORTANT:
# model is loaded from the locked production artifact.
# There is NO fitting/retraining here.

raw_scores = model.decision_function(
    X_validation
)

calibrated_probability = calibrator.predict_proba(
    raw_scores
)

calibrated_probability = np.asarray(
    calibrated_probability
).reshape(-1)

if len(calibrated_probability) != len(
    validation
):
    raise ValueError(
        "Prediction count does not match validation rows."
    )

if not np.all(
    np.isfinite(calibrated_probability)
):
    raise ValueError(
        "Non-finite calibrated probabilities detected."
    )

if np.any(calibrated_probability < 0) or np.any(
    calibrated_probability > 1
):
    raise ValueError(
        "Calibrated probabilities outside [0, 1]."
    )


# ============================================================
# CORE MODEL METRICS
# ============================================================

validation_logloss = log_loss(
    y_validation,
    calibrated_probability,
)

validation_auc = roc_auc_score(
    y_validation,
    calibrated_probability,
)

print("\nCore validation metrics:")
print(
    f"  Calibrated log loss: "
    f"{validation_logloss:.6f}"
)

print(
    f"  ROC AUC: "
    f"{validation_auc:.6f}"
)


# ============================================================
# BUILD ERROR ANALYSIS TABLE
# ============================================================

analysis = pd.DataFrame(
    {
        "row_index": np.arange(
            len(validation)
        ),
        "actual": y_validation.values,
        "raw_logit_score": raw_scores,
        "predicted_probability":
            calibrated_probability,
    }
)


# Preserve useful identifiers if present.

for column in [
    "customer_id",
    "snapshot_date",
]:
    if column in validation.columns:
        analysis[column] = (
            validation[column].values
        )


# ============================================================
# STANDARD 0.50 THRESHOLD DIAGNOSTIC
# ============================================================

STANDARD_THRESHOLD = 0.50

analysis["predicted_class_0_50"] = (
    analysis["predicted_probability"]
    >= STANDARD_THRESHOLD
).astype(int)

y_pred = analysis[
    "predicted_class_0_50"
].values

tn, fp, fn, tp = confusion_matrix(
    y_validation,
    y_pred,
    labels=[0, 1],
).ravel()

accuracy = accuracy_score(
    y_validation,
    y_pred,
)

precision = precision_score(
    y_validation,
    y_pred,
    zero_division=0,
)

recall = recall_score(
    y_validation,
    y_pred,
    zero_division=0,
)

f1 = f1_score(
    y_validation,
    y_pred,
    zero_division=0,
)

print("\n0.50 threshold diagnostic:")
print(
    f"  TN: {tn:,}"
)

print(
    f"  FP: {fp:,}"
)

print(
    f"  FN: {fn:,}"
)

print(
    f"  TP: {tp:,}"
)

print(
    f"  Accuracy:  {accuracy:.6f}"
)

print(
    f"  Precision: {precision:.6f}"
)

print(
    f"  Recall:    {recall:.6f}"
)

print(
    f"  F1:        {f1:.6f}"
)


# ============================================================
# ERROR TYPE
# ============================================================

analysis["error_type"] = np.select(
    [
        (
            (analysis["actual"] == 1)
            & (analysis["predicted_class_0_50"] == 1)
        ),
        (
            (analysis["actual"] == 0)
            & (analysis["predicted_class_0_50"] == 0)
        ),
        (
            (analysis["actual"] == 0)
            & (analysis["predicted_class_0_50"] == 1)
        ),
        (
            (analysis["actual"] == 1)
            & (analysis["predicted_class_0_50"] == 0)
        ),
    ],
    [
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
    ],
    default="unknown",
)


# ============================================================
# RANK CUSTOMERS
# ============================================================

analysis = (
    analysis
    .sort_values(
        "predicted_probability",
        ascending=False,
    )
    .reset_index(drop=True)
)

n = len(analysis)

analysis["rank"] = (
    np.arange(n) + 1
)

analysis["rank_pct"] = (
    analysis["rank"] / n
)


# ============================================================
# RANKING SEGMENTS
# ============================================================

analysis["ranking_segment"] = np.select(
    [
        analysis["rank_pct"] <= 0.01,
        analysis["rank_pct"] <= 0.05,
        analysis["rank_pct"] <= 0.10,
        analysis["rank_pct"] <= 0.20,
    ],
    [
        "Top 1%",
        "Top 1-5%",
        "Top 5-10%",
        "Top 10-20%",
    ],
    default="Bottom 80%",
)


# ============================================================
# CAPTURE / PRECISION BY RANKING SEGMENT
# ============================================================

segment_order = [
    "Top 1%",
    "Top 1-5%",
    "Top 5-10%",
    "Top 10-20%",
    "Bottom 80%",
]

segment_rows = []

total_positive = int(
    y_validation.sum()
)

for segment in segment_order:

    subset = analysis[
        analysis["ranking_segment"] == segment
    ]

    positive_count = int(
        subset["actual"].sum()
    )

    row_count = len(subset)

    precision_segment = (
        positive_count / row_count
        if row_count > 0
        else 0.0
    )

    capture_rate = (
        positive_count / total_positive
        if total_positive > 0
        else 0.0
    )

    segment_rows.append(
        {
            "segment": segment,
            "row_count": row_count,
            "positive_count": positive_count,
            "positive_rate": precision_segment,
            "positive_capture_rate":
                capture_rate,
            "mean_predicted_probability":
                subset[
                    "predicted_probability"
                ].mean(),
            "min_predicted_probability":
                subset[
                    "predicted_probability"
                ].min(),
            "max_predicted_probability":
                subset[
                    "predicted_probability"
                ].max(),
        }
    )

ranking_summary = pd.DataFrame(
    segment_rows
)


# ============================================================
# CUMULATIVE TOP-K PERFORMANCE
# ============================================================

cutoffs = [
    0.01,
    0.05,
    0.10,
    0.20,
]

top_k_rows = []

for cutoff in cutoffs:

    k = max(
        1,
        int(np.ceil(n * cutoff)),
    )

    subset = analysis.iloc[:k]

    positive_count = int(
        subset["actual"].sum()
    )

    top_k_rows.append(
        {
            "top_fraction": cutoff,
            "rows": k,
            "positive_count": positive_count,
            "precision":
                positive_count / k,
            "positive_capture_rate":
                (
                    positive_count / total_positive
                    if total_positive > 0
                    else 0.0
                ),
            "mean_probability":
                subset[
                    "predicted_probability"
                ].mean(),
        }
    )

top_k_summary = pd.DataFrame(
    top_k_rows
)


# ============================================================
# FALSE POSITIVE / FALSE NEGATIVE PROFILES
# ============================================================

false_positive = analysis[
    analysis["error_type"] == "false_positive"
].copy()

false_negative = analysis[
    analysis["error_type"] == "false_negative"
].copy()

true_positive = analysis[
    analysis["error_type"] == "true_positive"
].copy()

true_negative = analysis[
    analysis["error_type"] == "true_negative"
].copy()


# ============================================================
# HIGH-CONFIDENCE ERRORS
# ============================================================

# Highest-confidence false positives:
# actual = 0 but model probability is high.

high_confidence_fp = (
    false_positive
    .sort_values(
        "predicted_probability",
        ascending=False,
    )
    .head(100)
    .copy()
)

# Highest-confidence false negatives:
# actual = 1 but model probability is low.

high_confidence_fn = (
    false_negative
    .sort_values(
        "predicted_probability",
        ascending=True,
    )
    .head(100)
    .copy()
)


# ============================================================
# ERROR SUMMARY
# ============================================================

error_summary = pd.DataFrame(
    [
        {
            "error_type": "true_positive",
            "row_count": len(true_positive),
            "mean_probability":
                true_positive[
                    "predicted_probability"
                ].mean(),
            "mean_actual":
                true_positive["actual"].mean(),
        },
        {
            "error_type": "true_negative",
            "row_count": len(true_negative),
            "mean_probability":
                true_negative[
                    "predicted_probability"
                ].mean(),
            "mean_actual":
                true_negative["actual"].mean(),
        },
        {
            "error_type": "false_positive",
            "row_count": len(false_positive),
            "mean_probability":
                false_positive[
                    "predicted_probability"
                ].mean(),
            "mean_actual":
                false_positive["actual"].mean(),
        },
        {
            "error_type": "false_negative",
            "row_count": len(false_negative),
            "mean_probability":
                false_negative[
                    "predicted_probability"
                ].mean(),
            "mean_actual":
                false_negative["actual"].mean(),
        },
    ]
)


# ============================================================
# SNAPSHOT ERROR ANALYSIS
# ============================================================

snapshot_rows = []

if "snapshot_date" in validation.columns:

    snapshot_values = (
        validation["snapshot_date"]
        .dropna()
        .unique()
    )

    snapshot_lookup = pd.DataFrame(
        {
            "row_index":
                np.arange(len(validation)),
            "snapshot_date":
                validation[
                    "snapshot_date"
                ].values,
        }
    )

    snapshot_lookup = snapshot_lookup[
        [
            "row_index",
            "snapshot_date",
        ]
    ]

    snapshot_analysis = analysis.merge(
        snapshot_lookup,
        on="row_index",
        how="left",
        suffixes=("", "_lookup"),
    )

    for snapshot, group in (
        snapshot_analysis
        .groupby("snapshot_date")
    ):

        actual = group["actual"]

        predicted = (
            group[
                "predicted_class_0_50"
            ]
        )

        snapshot_rows.append(
            {
                "snapshot_date": snapshot,
                "row_count": len(group),
                "positive_count":
                    int(actual.sum()),
                "positive_rate":
                    actual.mean(),
                "false_positive_count":
                    int(
                        (
                            (actual == 0)
                            & (predicted == 1)
                        ).sum()
                    ),
                "false_negative_count":
                    int(
                        (
                            (actual == 1)
                            & (predicted == 0)
                        ).sum()
                    ),
                "mean_probability":
                    group[
                        "predicted_probability"
                    ].mean(),
                "log_loss":
                    log_loss(
                        actual,
                        group[
                            "predicted_probability"
                        ],
                    ),
            }
        )

snapshot_summary = pd.DataFrame(
    snapshot_rows
)


# ============================================================
# WRITE OUTPUTS
# ============================================================

error_scores_path = (
    OUTPUT_DIR
    / "logistic_validation_error_scores.csv"
)

metrics_path = (
    OUTPUT_DIR
    / "logistic_error_metrics.csv"
)

ranking_path = (
    OUTPUT_DIR
    / "logistic_ranking_performance.csv"
)

topk_path = (
    OUTPUT_DIR
    / "logistic_topk_performance.csv"
)

error_summary_path = (
    OUTPUT_DIR
    / "logistic_error_type_summary.csv"
)

fp_path = (
    OUTPUT_DIR
    / "logistic_high_confidence_false_positives.csv"
)

fn_path = (
    OUTPUT_DIR
    / "logistic_high_confidence_false_negatives.csv"
)

snapshot_path = (
    OUTPUT_DIR
    / "logistic_snapshot_error_analysis.csv"
)


analysis.to_csv(
    error_scores_path,
    index=False,
)

metrics = pd.DataFrame(
    [
        {
            "model_version":
                metadata.get(
                    "model_version"
                ),
            "validation_rows":
                len(validation),
            "validation_positive_count":
                int(y_validation.sum()),
            "validation_positive_rate":
                y_validation.mean(),
            "calibrated_log_loss":
                validation_logloss,
            "roc_auc":
                validation_auc,
            "threshold":
                STANDARD_THRESHOLD,
            "true_negative":
                tn,
            "false_positive":
                fp,
            "false_negative":
                fn,
            "true_positive":
                tp,
            "accuracy":
                accuracy,
            "precision":
                precision,
            "recall":
                recall,
            "f1":
                f1,
        }
    ]
)

metrics.to_csv(
    metrics_path,
    index=False,
)

ranking_summary.to_csv(
    ranking_path,
    index=False,
)

top_k_summary.to_csv(
    topk_path,
    index=False,
)

error_summary.to_csv(
    error_summary_path,
    index=False,
)

high_confidence_fp.to_csv(
    fp_path,
    index=False,
)

high_confidence_fn.to_csv(
    fn_path,
    index=False,
)

snapshot_summary.to_csv(
    snapshot_path,
    index=False,
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("STEP 14B COMPLETE")
print("=" * 70)

print("\nProduction lock:")
print(
    f"  Model version: "
    f"{metadata.get('model_version')}"
)
print(
    f"  Classifier: "
    f"{classifier.__class__.__name__}"
)
print("  Retrained: NO")
print("  Recalibrated: NO")
print("  Test data used: NO")

print("\nCore metrics:")
print(
    f"  Calibrated log loss: "
    f"{validation_logloss:.6f}"
)
print(
    f"  ROC AUC: "
    f"{validation_auc:.6f}"
)

print("\n0.50 threshold diagnostic:")
print(
    f"  TN: {tn:,}"
)
print(
    f"  FP: {fp:,}"
)
print(
    f"  FN: {fn:,}"
)
print(
    f"  TP: {tp:,}"
)
print(
    f"  Precision: {precision:.6f}"
)
print(
    f"  Recall:    {recall:.6f}"
)
print(
    f"  F1:        {f1:.6f}"
)

print("\nRanking performance:")

print(
    ranking_summary[
        [
            "segment",
            "row_count",
            "positive_count",
            "positive_rate",
            "positive_capture_rate",
        ]
    ].to_string(index=False)
)

print("\nError counts:")
print(
    f"  False positives: "
    f"{len(false_positive):,}"
)
print(
    f"  False negatives: "
    f"{len(false_negative):,}"
)

print("\nOutputs written:")
print(f"  {error_scores_path}")
print(f"  {metrics_path}")
print(f"  {ranking_path}")
print(f"  {topk_path}")
print(f"  {error_summary_path}")
print(f"  {fp_path}")
print(f"  {fn_path}")
print(f"  {snapshot_path}")

print("\nDone.")