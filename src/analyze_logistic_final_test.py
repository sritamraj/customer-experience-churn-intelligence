from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# STEP 17 — FINAL FROZEN-MODEL TEST AUDIT
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "data" / "model"
ARTIFACT_DIR = MODEL_DIR / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "logistic_model.joblib"
CALIBRATOR_PATH = ARTIFACT_DIR / "sigmoid_calibrator.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
TEST_PATH = MODEL_DIR / "test.csv"

OUTPUT_METRICS = MODEL_DIR / "logistic_final_test_metrics.csv"
OUTPUT_RANKING = MODEL_DIR / "logistic_final_test_ranking.csv"
OUTPUT_SCORES = MODEL_DIR / "logistic_final_test_scores.csv"


EXPECTED_MODEL_VERSION = "logistic_16_sigmoid_v1"

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

EXPECTED_PIPELINE_STEPS = [
    "imputer",
    "scaler",
    "model",
]

TARGET_COLUMN = "future_purchase_flag"
SNAPSHOT_COLUMN = "snapshot_date"


def validate_artifacts(model, calibrator, metadata):
    print("Validating canonical artifacts...")

    if metadata.get("model_version") != EXPECTED_MODEL_VERSION:
        raise ValueError(
            f"Unexpected model version: "
            f"{metadata.get('model_version')}"
        )

    actual_features = metadata.get("features")

    if actual_features != LOCKED_FEATURES:
        raise ValueError(
            "Metadata feature list does not exactly match "
            "the locked 16-feature contract."
        )

    actual_steps = list(model.named_steps.keys())

    if actual_steps != EXPECTED_PIPELINE_STEPS:
        raise ValueError(
            f"Unexpected pipeline steps: {actual_steps}"
        )

    if not hasattr(calibrator, "predict_proba"):
        raise ValueError(
            "Canonical calibrator does not expose predict_proba()."
        )

    print(f"Model version: {metadata['model_version']}")
    print(f"Pipeline steps: {actual_steps}")
    print(f"Locked feature count: {len(LOCKED_FEATURES)}")
    print("Artifact validation: PASS")
    print()


def assign_cumulative_band(rank, total):
    pct = rank / total

    if pct <= 0.01:
        return "Top 1%"
    elif pct <= 0.05:
        return "Top 1-5%"
    elif pct <= 0.10:
        return "Top 5-10%"
    elif pct <= 0.20:
        return "Top 10-20%"
    else:
        return "Bottom 80%"


def main():
    print("=" * 70)
    print("STEP 17 — FINAL FROZEN-MODEL TEST AUDIT")
    print("=" * 70)

    for path in [
        MODEL_PATH,
        CALIBRATOR_PATH,
        METADATA_PATH,
        TEST_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact missing: {path}"
            )

    # --------------------------------------------------------
    # Load canonical artifacts
    # --------------------------------------------------------

    model = joblib.load(MODEL_PATH)
    calibrator = joblib.load(CALIBRATOR_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    validate_artifacts(
        model,
        calibrator,
        metadata,
    )

    # --------------------------------------------------------
    # Load final test
    # --------------------------------------------------------

    test = pd.read_csv(TEST_PATH)

    print(f"Test rows: {len(test):,}")

    if TARGET_COLUMN not in test.columns:
        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    if SNAPSHOT_COLUMN not in test.columns:
        raise ValueError(
            f"Missing snapshot column: {SNAPSHOT_COLUMN}"
        )

    snapshots = test[SNAPSHOT_COLUMN].dropna().unique()

    if len(snapshots) != 1:
        raise ValueError(
            f"Expected exactly one test snapshot, found {snapshots}"
        )

    test_snapshot = str(snapshots[0])

    if test_snapshot != "2018-06-19":
        raise ValueError(
            f"Unexpected final test snapshot: {test_snapshot}"
        )

    y = test[TARGET_COLUMN].astype(int)

    print(f"Test snapshot: {test_snapshot}")
    print(f"Test buyers: {int(y.sum()):,}")
    print(f"Test base rate: {y.mean():.6f}")
    print()

    # --------------------------------------------------------
    # Verify feature contract
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in LOCKED_FEATURES
        if feature not in test.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing locked features: {missing_features}"
        )

    forbidden_future_columns = [
        "future_purchase_flag",
        "future_delivered_order_count",
        "future_revenue",
        "future_item_revenue",
    ]

    model_features = set(LOCKED_FEATURES)

    overlap = model_features.intersection(
        forbidden_future_columns
    )

    if overlap:
        raise ValueError(
            f"Future outcome leakage detected: {overlap}"
        )

    X = test[LOCKED_FEATURES].copy()

    # --------------------------------------------------------
    # IMPORTANT:
    # NO FIT / RETRAIN / RECALIBRATION BELOW
    # --------------------------------------------------------

    classifier = model.named_steps["model"]
    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]

    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)

    raw_scores = classifier.decision_function(X_scaled)

    calibrated_probability = calibrator.predict_proba(
        raw_scores
    )

    # --------------------------------------------------------
    # Basic prediction validation
    # --------------------------------------------------------

    if not np.isfinite(raw_scores).all():
        raise ValueError(
            "Non-finite raw model scores detected."
        )

    if not np.isfinite(calibrated_probability).all():
        raise ValueError(
            "Non-finite calibrated probabilities detected."
        )

    if not np.all(
        (calibrated_probability >= 0)
        & (calibrated_probability <= 1)
    ):
        raise ValueError(
            "Calibrated probabilities outside [0,1]."
        )

    # --------------------------------------------------------
    # Build scored test table
    # --------------------------------------------------------

    scores = pd.DataFrame(
        {
            "row_index": np.arange(len(test)),
            "actual": y.to_numpy(),
            "raw_logit_score": raw_scores,
            "predicted_probability":
                calibrated_probability,
            "snapshot_date":
                test[SNAPSHOT_COLUMN].astype(str).to_numpy(),
        }
    )

    scores = scores.sort_values(
        "predicted_probability",
        ascending=False,
    ).reset_index(drop=True)

    scores["rank"] = np.arange(
        1,
        len(scores) + 1,
    )

    scores["rank_pct"] = (
        scores["rank"] / len(scores)
    )

    scores["ranking_segment"] = scores["rank"].apply(
        lambda rank: assign_cumulative_band(
            rank,
            len(scores),
        )
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    calibrated_log_loss = log_loss(
        y,
        calibrated_probability,
        labels=[0, 1],
    )

    roc_auc = roc_auc_score(
        y,
        calibrated_probability,
    )

    threshold = 0.50

    predicted_class = (
        calibrated_probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predicted_class,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(
        y,
        predicted_class,
    )

    precision = precision_score(
        y,
        predicted_class,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predicted_class,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predicted_class,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Ranking metrics
    # --------------------------------------------------------

    total_customers = len(scores)
    total_buyers = int(y.sum())
    base_rate = y.mean()

    ranking_rows = []

    for pct in [0.01, 0.05, 0.10, 0.20]:

        n = int(np.ceil(total_customers * pct))

        top = scores.iloc[:n]

        buyers = int(top["actual"].sum())

        buyer_rate = buyers / len(top)

        lift = buyer_rate / base_rate

        capture = (
            buyers / total_buyers
            if total_buyers > 0
            else np.nan
        )

        ranking_rows.append(
            {
                "target_pct": pct,
                "target_label": f"Top {int(pct * 100)}%",
                "customers_targeted": len(top),
                "actual_buyers": buyers,
                "buyer_rate": buyer_rate,
                "base_rate": base_rate,
                "lift": lift,
                "buyer_capture": capture,
            }
        )

    ranking = pd.DataFrame(ranking_rows)

    # --------------------------------------------------------
    # Metrics table
    # --------------------------------------------------------

    metrics = pd.DataFrame(
        [
            {
                "test_snapshot": test_snapshot,
                "test_customers": total_customers,
                "test_buyers": total_buyers,
                "base_rate": base_rate,
                "calibrated_log_loss":
                    calibrated_log_loss,
                "roc_auc": roc_auc,
                "threshold": threshold,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        ]
    )

    # --------------------------------------------------------
    # Write artifacts
    # --------------------------------------------------------

    metrics.to_csv(
        OUTPUT_METRICS,
        index=False,
    )

    ranking.to_csv(
        OUTPUT_RANKING,
        index=False,
    )

    scores.to_csv(
        OUTPUT_SCORES,
        index=False,
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print("FINAL TEST RESULTS")
    print("-" * 70)

    print(
        f"Calibrated log loss: "
        f"{calibrated_log_loss:.6f}"
    )

    print(
        f"ROC AUC: "
        f"{roc_auc:.6f}"
    )

    print()
    print("0.50 THRESHOLD")
    print("-" * 70)

    print(f"TN: {tn:,}")
    print(f"FP: {fp:,}")
    print(f"FN: {fn:,}")
    print(f"TP: {tp:,}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1: {f1:.6f}")

    print()
    print("RANKING PERFORMANCE")
    print("-" * 70)

    ranking_display = ranking.copy()

    ranking_display["buyer_rate"] = (
        ranking_display["buyer_rate"] * 100
    ).round(3)

    ranking_display["lift"] = (
        ranking_display["lift"]
    ).round(3)

    ranking_display["buyer_capture"] = (
        ranking_display["buyer_capture"] * 100
    ).round(3)

    print(
        ranking_display.to_string(index=False)
    )

    print()
    print("IMPORTANT")
    print("-" * 70)
    print(
        "This is the final untouched test evaluation."
    )
    print(
        "No model fitting, retraining, or recalibration "
        "was performed."
    )
    print(
        "The test result must not be used to tune the model."
    )

    print()
    print(f"Wrote: {OUTPUT_METRICS}")
    print(f"Wrote: {OUTPUT_RANKING}")
    print(f"Wrote: {OUTPUT_SCORES}")

    print("=" * 70)
    print("STEP 17 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()