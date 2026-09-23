from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from calibration import SigmoidCalibrator


BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = BASE_DIR / "data" / "model" / "train.csv"
VAL_PATH = BASE_DIR / "data" / "model" / "validation.csv"
ARTIFACT_DIR = BASE_DIR / "data" / "model" / "artifacts"


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


def main():

    print("=" * 70)
    print("BUILDING PRODUCTION MODEL ARTIFACTS")
    print("=" * 70)

    print("\nLoading data...")

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VAL_PATH)

    print(f"Train rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")

    required_columns = FEATURES + [TARGET]

    missing_train = [
        column
        for column in required_columns
        if column not in train.columns
    ]

    missing_validation = [
        column
        for column in required_columns
        if column not in validation.columns
    ]

    if missing_train:
        raise ValueError(
            f"Missing columns in train.csv: {missing_train}"
        )

    if missing_validation:
        raise ValueError(
            f"Missing columns in validation.csv: {missing_validation}"
        )

    leakage_columns = [
        column
        for column in FEATURES
        if "future" in column.lower()
    ]

    if leakage_columns:
        raise ValueError(
            f"Potential leakage features detected: {leakage_columns}"
        )

    X_train = train[FEATURES]
    y_train = train[TARGET].astype(int)

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET].astype(int)

    print(f"\nFeatures: {len(FEATURES)}")
    print(f"Training positives: {y_train.sum():,}")
    print(
        f"Training positive rate: "
        f"{y_train.mean():.6%}"
    )
    print(f"Validation positives: {y_validation.sum():,}")
    print(
        f"Validation positive rate: "
        f"{y_validation.mean():.6%}"
    )

    print("\nTraining locked Logistic Regression...")

    model = Pipeline(
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
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    C=1.0,
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    raw_validation_scores = model.decision_function(
        X_validation
    )

    print("\nFitting sigmoid/Platt calibrator...")
    print("Calibration data: validation snapshot only")

    calibrator = SigmoidCalibrator()

    calibrator.fit(
        raw_validation_scores,
        y_validation,
    )

    calibrated_validation_probability = (
        calibrator.predict_proba(
            raw_validation_scores
        )
    )

    validation_log_loss = log_loss(
        y_validation,
        calibrated_validation_probability,
    )

    print(
        f"Validation calibrated log loss: "
        f"{validation_log_loss:.8f}"
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        ARTIFACT_DIR / "logistic_model.joblib"
    )

    calibrator_path = (
        ARTIFACT_DIR / "sigmoid_calibrator.joblib"
    )

    metadata_path = (
        ARTIFACT_DIR / "model_metadata.json"
    )

    print("\nSaving artifacts...")

    joblib.dump(
        model,
        model_path,
    )

    joblib.dump(
        calibrator,
        calibrator_path,
    )

    metadata = {
        "model_version": "logistic_16_sigmoid_v1",
        "model_type": "LogisticRegression",
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "target": TARGET,
        "imputation": "median",
        "scaling": "StandardScaler",
        "class_weight": "balanced",
        "C": 1.0,
        "max_iter": 2000,
        "random_state": 42,
        "calibration": "sigmoid_platt",
        "training_snapshots": [
            "2017-09-01",
            "2017-12-01",
        ],
        "calibration_snapshot": "2018-03-01",
        "test_snapshot": "2018-06-19",
        "test_used_for_training": False,
        "test_used_for_calibration": False,
        "validation_calibrated_log_loss": float(
            validation_log_loss
        ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print("\nReload verification...")

    loaded_model = joblib.load(model_path)
    loaded_calibrator = joblib.load(
        calibrator_path
    )

    reloaded_raw_scores = (
        loaded_model.decision_function(
            X_validation
        )
    )

    reloaded_probability = (
        loaded_calibrator.predict_proba(
            reloaded_raw_scores
        )
    )

    reloaded_log_loss = log_loss(
        y_validation,
        reloaded_probability,
    )

    if not np.isclose(
        validation_log_loss,
        reloaded_log_loss,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise RuntimeError(
            "Reload verification failed."
        )

    print("Model reload: PASS")
    print("Calibrator reload: PASS")
    print("Prediction verification: PASS")

    print("\nArtifacts created:")
    print(f"  {model_path}")
    print(f"  {calibrator_path}")
    print(f"  {metadata_path}")

    print("\n" + "=" * 70)
    print("STEP 14A COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()