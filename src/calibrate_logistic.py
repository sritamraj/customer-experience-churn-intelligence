from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "model" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "model" / "validation.csv"
TEST_PATH = PROJECT_ROOT / "data" / "model" / "test.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "model"

RANDOM_STATE = 42


# ============================================================
# LOCKED FEATURE SET
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

TARGET = "future_purchase_flag"


# ============================================================
# HELPERS
# ============================================================

def load_data(path):
    df = pd.read_csv(path)

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features in {path.name}: {missing_features}"
        )

    if TARGET not in df.columns:
        raise ValueError(
            f"Missing target '{TARGET}' in {path.name}"
        )

    return df


def build_base_model():
    """
    Locked Logistic 16 model.

    This must remain identical to the final Logistic 16 model.
    """

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


def evaluate_predictions(y_true, probabilities):
    """
    Classification + probability-quality metrics.
    """

    probabilities = np.clip(
        np.asarray(probabilities),
        1e-6,
        1 - 1e-6,
    )

    return {
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
        ),
        "mean_predicted_probability": probabilities.mean(),
        "observed_positive_rate": y_true.mean(),
    }


def calibration_table(y_true, probabilities, n_bins=10):
    """
    Equal-frequency calibration table.
    """

    data = pd.DataFrame(
        {
            "y_true": np.asarray(y_true),
            "probability": np.asarray(probabilities),
        }
    )

    data["bin"] = pd.qcut(
        data["probability"],
        q=n_bins,
        labels=False,
        duplicates="drop",
    ) + 1

    result = (
        data.groupby(
            "bin",
            observed=True,
        )
        .agg(
            customers=("y_true", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_purchase_rate=("y_true", "mean"),
        )
        .reset_index()
    )

    result["absolute_calibration_error"] = (
        result["mean_predicted_probability"]
        - result["observed_purchase_rate"]
    ).abs()

    return result


def print_metrics(title, metrics):
    print("\n" + title)
    print("-" * 70)

    for key, value in metrics.items():
        print(f"{key:35s}: {value:.8f}")


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOGISTIC CALIBRATION EXPERIMENT")
print("=" * 70)

train = load_data(TRAIN_PATH)
validation = load_data(VALIDATION_PATH)
test = load_data(TEST_PATH)

X_train = train[FEATURES]
y_train = train[TARGET].astype(int)

X_validation = validation[FEATURES]
y_validation = validation[TARGET].astype(int)

X_test = test[FEATURES]
y_test = test[TARGET].astype(int)


print("\nDATA")
print("-" * 70)

print(f"Train rows:        {len(train):,}")
print(f"Train buyers:      {y_train.sum():,}")

print(f"Validation rows:   {len(validation):,}")
print(f"Validation buyers: {y_validation.sum():,}")

print(f"Test rows:         {len(test):,}")
print(f"Test buyers:       {y_test.sum():,}")


# ============================================================
# TRAIN LOCKED LOGISTIC MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOCKED LOGISTIC 16")
print("=" * 70)

base_model = build_base_model()

base_model.fit(
    X_train,
    y_train,
)

validation_raw = base_model.predict_proba(
    X_validation
)[:, 1]

test_raw = base_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# RAW MODEL
# ============================================================

raw_validation_metrics = evaluate_predictions(
    y_validation,
    validation_raw,
)

print_metrics(
    "RAW LOGISTIC — VALIDATION",
    raw_validation_metrics,
)


# ============================================================
# SIGMOID / PLATT CALIBRATION
# ============================================================

print("\n" + "=" * 70)
print("SIGMOID / PLATT CALIBRATION")
print("=" * 70)

validation_raw_clipped = np.clip(
    validation_raw,
    1e-6,
    1 - 1e-6,
)

test_raw_clipped = np.clip(
    test_raw,
    1e-6,
    1 - 1e-6,
)

validation_logit = np.log(
    validation_raw_clipped
    / (1 - validation_raw_clipped)
)

test_logit = np.log(
    test_raw_clipped
    / (1 - test_raw_clipped)
)


platt_model = LogisticRegression(
    C=1.0,
    max_iter=2000,
    random_state=RANDOM_STATE,
)

platt_model.fit(
    validation_logit.reshape(-1, 1),
    y_validation,
)

validation_sigmoid = platt_model.predict_proba(
    validation_logit.reshape(-1, 1)
)[:, 1]

test_sigmoid = platt_model.predict_proba(
    test_logit.reshape(-1, 1)
)[:, 1]


sigmoid_validation_metrics = evaluate_predictions(
    y_validation,
    validation_sigmoid,
)

print_metrics(
    "SIGMOID — VALIDATION",
    sigmoid_validation_metrics,
)


# ============================================================
# ISOTONIC CALIBRATION
# ============================================================

print("\n" + "=" * 70)
print("ISOTONIC CALIBRATION")
print("=" * 70)

isotonic_model = IsotonicRegression(
    y_min=0.0,
    y_max=1.0,
    out_of_bounds="clip",
)

isotonic_model.fit(
    validation_raw,
    y_validation,
)

validation_isotonic = isotonic_model.predict(
    validation_raw
)

test_isotonic = isotonic_model.predict(
    test_raw
)


isotonic_validation_metrics = evaluate_predictions(
    y_validation,
    validation_isotonic,
)

print_metrics(
    "ISOTONIC — VALIDATION",
    isotonic_validation_metrics,
)


# ============================================================
# VALIDATION COMPARISON
# ============================================================

comparison = pd.DataFrame(
    [
        {
            "model": "raw_logistic",
            **raw_validation_metrics,
        },
        {
            "model": "sigmoid_platt",
            **sigmoid_validation_metrics,
        },
        {
            "model": "isotonic",
            **isotonic_validation_metrics,
        },
    ]
)

print("\n" + "=" * 70)
print("VALIDATION CALIBRATION COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}",
    )
)


# ============================================================
# CALIBRATION DECILES
# ============================================================

raw_calibration = calibration_table(
    y_validation,
    validation_raw,
)

raw_calibration["model"] = "raw_logistic"

sigmoid_calibration = calibration_table(
    y_validation,
    validation_sigmoid,
)

sigmoid_calibration["model"] = "sigmoid_platt"

isotonic_calibration = calibration_table(
    y_validation,
    validation_isotonic,
)

isotonic_calibration["model"] = "isotonic"

calibration_deciles = pd.concat(
    [
        raw_calibration,
        sigmoid_calibration,
        isotonic_calibration,
    ],
    ignore_index=True,
)


# ============================================================
# TEST SCORES
# ============================================================

test_results = test[
    [
        "customer_unique_id",
        "snapshot_date",
        TARGET,
    ]
].copy()

test_results["raw_logistic_probability"] = test_raw
test_results["sigmoid_probability"] = test_sigmoid
test_results["isotonic_probability"] = test_isotonic


# ============================================================
# TEST METRICS
# ============================================================

raw_test_metrics = evaluate_predictions(
    y_test,
    test_raw,
)

sigmoid_test_metrics = evaluate_predictions(
    y_test,
    test_sigmoid,
)

isotonic_test_metrics = evaluate_predictions(
    y_test,
    test_isotonic,
)

test_comparison = pd.DataFrame(
    [
        {
            "model": "raw_logistic",
            **raw_test_metrics,
        },
        {
            "model": "sigmoid_platt",
            **sigmoid_test_metrics,
        },
        {
            "model": "isotonic",
            **isotonic_test_metrics,
        },
    ]
)


print("\n" + "=" * 70)
print("TEST COMPARISON")
print("=" * 70)

print(
    test_comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}",
    )
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

comparison_path = (
    OUTPUT_DIR
    / "logistic_calibration_validation_comparison.csv"
)

deciles_path = (
    OUTPUT_DIR
    / "logistic_calibration_validation_deciles.csv"
)

test_scores_path = (
    OUTPUT_DIR
    / "logistic_calibration_test_scored.csv"
)

test_metrics_path = (
    OUTPUT_DIR
    / "logistic_calibration_test_metrics.csv"
)

comparison.to_csv(
    comparison_path,
    index=False,
)

calibration_deciles.to_csv(
    deciles_path,
    index=False,
)

test_results.to_csv(
    test_scores_path,
    index=False,
)

test_comparison.to_csv(
    test_metrics_path,
    index=False,
)


print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(comparison_path)
print(deciles_path)
print(test_scores_path)
print(test_metrics_path)

print("\nCalibration experiment complete.")