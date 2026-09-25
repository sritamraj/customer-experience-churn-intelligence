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

EXPECTED_VERSION = "logistic_16_sigmoid_v1"


def load_model():
    return joblib.load(
        ARTIFACT / "logistic_model.joblib"
    )


def load_calibrator():
    return joblib.load(
        ARTIFACT / "sigmoid_calibrator.joblib"
    )


def load_test_data():
    return pd.read_csv(
        MODEL_DIR / "test.csv"
    )


def test_metadata_contract():
    metadata_path = ARTIFACT / "model_metadata.json"

    assert metadata_path.exists()

    metadata = json.loads(
        metadata_path.read_text()
    )

    assert metadata["model_version"] == EXPECTED_VERSION
    assert metadata["feature_count"] == 16
    assert metadata["features"] == FEATURES
    assert metadata["target"] == "future_purchase_flag"
    assert metadata["test_snapshot"] == "2018-06-19"


def test_model_pipeline_contract():
    model = load_model()

    assert hasattr(model, "named_steps")

    assert list(model.named_steps.keys()) == [
        "imputer",
        "scaler",
        "model",
    ]

    lr = model.named_steps["model"]

    assert lr.coef_.shape == (1, 16)
    assert lr.intercept_.shape == (1,)

    assert np.isfinite(lr.coef_).all()
    assert np.isfinite(lr.intercept_).all()


def test_calibrator_contract():
    calibrator = load_calibrator()

    assert hasattr(
        calibrator,
        "predict_proba",
    )


def test_production_batch_prediction():
    model = load_model()
    calibrator = load_calibrator()
    test = load_test_data()

    X = test[FEATURES].head(100).copy()

    raw_probability = model.predict_proba(X)[:, 1]

    calibrated_probability = (
        calibrator.predict_proba(raw_probability)
    )

    assert len(raw_probability) == 100
    assert len(calibrated_probability) == 100

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


def test_inference_is_deterministic():
    model = load_model()
    calibrator = load_calibrator()
    test = load_test_data()

    X = test[FEATURES].head(100).copy()

    raw_1 = model.predict_proba(X)[:, 1]
    calibrated_1 = calibrator.predict_proba(raw_1)

    raw_2 = model.predict_proba(X)[:, 1]
    calibrated_2 = calibrator.predict_proba(raw_2)

    assert np.allclose(raw_1, raw_2)
    assert np.allclose(calibrated_1, calibrated_2)


def test_required_feature_schema():
    test = load_test_data()

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in test.columns
    ]

    assert not missing_features


def test_feature_names_contain_no_future_information():
    future_terms = [
        "future",
        "next",
        "target",
        "label",
    ]

    bad_features = [
        feature
        for feature in FEATURES
        if any(
            term in feature.lower()
            for term in future_terms
        )
    ]

    assert not bad_features