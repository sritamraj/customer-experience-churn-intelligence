import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "model"
ARTIFACT = MODEL_DIR / "artifacts"

sys.path.insert(0, str(ROOT / "src"))

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
EXPECTED_VERSION = "logistic_16_sigmoid_v1"
EXPECTED_TEST_SNAPSHOT = "2018-06-19"


print("=" * 70)
print("STEP 24 — PRODUCTION ARTIFACT VALIDATION")
print("=" * 70)


# ================================================================
# 1. METADATA
# ================================================================

metadata_path = ARTIFACT / "model_metadata.json"

assert metadata_path.exists(), (
    f"Missing metadata: {metadata_path}"
)

metadata = json.loads(
    metadata_path.read_text()
)

assert metadata["model_version"] == EXPECTED_VERSION
assert metadata["feature_count"] == 16
assert metadata["features"] == FEATURES
assert metadata["target"] == TARGET
assert metadata["test_snapshot"] == EXPECTED_TEST_SNAPSHOT


print("Metadata validation: PASS")


# ================================================================
# 2. MODEL ARTIFACT
# ================================================================

model_path = ARTIFACT / "logistic_model.joblib"

assert model_path.exists(), (
    f"Missing model artifact: {model_path}"
)

model = joblib.load(model_path)

assert hasattr(model, "named_steps")

assert list(model.named_steps.keys()) == [
    "imputer",
    "scaler",
    "model",
]

print("Model artifact validation: PASS")


# ================================================================
# 3. MODEL COMPONENTS
# ================================================================

assert model.named_steps["imputer"] is not None
assert model.named_steps["scaler"] is not None
assert model.named_steps["model"] is not None

print("Pipeline component validation: PASS")


# ================================================================
# 4. COEFFICIENT COUNT
# ================================================================

lr = model.named_steps["model"]

assert lr.coef_.shape == (1, 16)
assert lr.intercept_.shape == (1,)

assert np.isfinite(lr.coef_).all()
assert np.isfinite(lr.intercept_).all()

print("Coefficient validation: PASS")


# ================================================================
# 5. CALIBRATOR
# ================================================================

calibrator_path = (
    ARTIFACT / "sigmoid_calibrator.joblib"
)

assert calibrator_path.exists(), (
    f"Missing calibrator: {calibrator_path}"
)

calibrator = joblib.load(calibrator_path)

assert hasattr(
    calibrator,
    "predict_proba"
)

print("Calibrator validation: PASS")


# ================================================================
# 6. TEST DATA SCHEMA
# ================================================================

test = pd.read_csv(
    MODEL_DIR / "test.csv"
)

assert EXPECTED_TEST_SNAPSHOT in (
    test["snapshot_date"]
    .astype(str)
    .unique()
)

missing_features = [
    f for f in FEATURES
    if f not in test.columns
]

assert not missing_features, (
    f"Missing features: {missing_features}"
)

assert TARGET in test.columns

print("Test schema validation: PASS")


# ================================================================
# 7. NO FUTURE FEATURE NAMES
# ================================================================

future_terms = [
    "future",
    "next",
    "target",
    "label",
]

bad_features = [
    f for f in FEATURES
    if any(term in f.lower() for term in future_terms)
]

assert not bad_features, (
    f"Potential leakage features: {bad_features}"
)

print("Feature leakage-name validation: PASS")


# ================================================================
# 8. PRODUCTION-STYLE BATCH PREDICTION
# ================================================================

X = test[FEATURES].head(100).copy()

raw_probability = model.predict_proba(X)[:, 1]

calibrated_probability = calibrator.predict_proba(
    raw_probability
)

assert len(raw_probability) == len(X)
assert len(calibrated_probability) == len(X)

assert np.isfinite(raw_probability).all()
assert np.isfinite(calibrated_probability).all()

assert (
    (raw_probability >= 0)
    & (raw_probability <= 1)
).all()

assert (
    (calibrated_probability >= 0)
    & (calibrated_probability <= 1)
).all()

print("Production batch prediction: PASS")


# ================================================================
# 9. DETERMINISM
# ================================================================

raw_probability_2 = model.predict_proba(X)[:, 1]

calibrated_probability_2 = (
    calibrator.predict_proba(
        raw_probability_2
    )
)

assert np.allclose(
    raw_probability,
    raw_probability_2,
)

assert np.allclose(
    calibrated_probability,
    calibrated_probability_2,
)

print("Deterministic inference validation: PASS")


# ================================================================
# 10. SAMPLE OUTPUT
# ================================================================

output = pd.DataFrame(
    {
        "raw_probability": raw_probability[:10],
        "calibrated_probability":
            calibrated_probability[:10],
    }
)

print()
print("SAMPLE PRODUCTION OUTPUT")
print(output.to_string(index=False))

print()
print("=" * 70)
print("STEP 24 RESULT: PASS")
print("=" * 70)
print("Model remains frozen.")
print("No fitting or recalibration performed.")