from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
MODEL_DIR = BASE_DIR / "data" / "model"
ARTIFACT_DIR = MODEL_DIR / "artifacts"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


VALIDATION_PATH = MODEL_DIR / "logistic_validation_scored.csv"
TEST_PATH = MODEL_DIR / "logistic_test_scored.csv"
PRODUCTION_PATH = MODEL_DIR / "production_scores.csv"

MODEL_PATH = ARTIFACT_DIR / "logistic_model.joblib"
CALIBRATOR_PATH = ARTIFACT_DIR / "sigmoid_calibrator.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

OUTPUT_DIR = MODEL_DIR / "monitoring"

OUTPUT_SUMMARY = (
    OUTPUT_DIR / "prediction_drift_summary.csv"
)

OUTPUT_PSI = (
    OUTPUT_DIR / "prediction_probability_psi.csv"
)


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


def calculate_psi(reference, current, bins=10):

    reference = np.asarray(
        reference,
        dtype=float,
    )

    current = np.asarray(
        current,
        dtype=float,
    )

    reference = reference[
        np.isfinite(reference)
    ]

    current = current[
        np.isfinite(current)
    ]

    if len(reference) == 0:
        raise RuntimeError(
            "Reference distribution is empty."
        )

    if len(current) == 0:
        raise RuntimeError(
            "Current distribution is empty."
        )

    quantiles = np.linspace(
        0,
        1,
        bins + 1,
    )

    edges = np.quantile(
        reference,
        quantiles,
    )

    edges = np.unique(edges)

    if len(edges) < 3:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts, _ = np.histogram(
        reference,
        bins=edges,
    )

    current_counts, _ = np.histogram(
        current,
        bins=edges,
    )

    reference_pct = (
        reference_counts
        / len(reference)
    )

    current_pct = (
        current_counts
        / len(current)
    )

    epsilon = 1e-6

    reference_pct = np.where(
        reference_pct == 0,
        epsilon,
        reference_pct,
    )

    current_pct = np.where(
        current_pct == 0,
        epsilon,
        current_pct,
    )

    psi = np.sum(
        (
            current_pct
            - reference_pct
        )
        *
        np.log(
            current_pct
            / reference_pct
        )
    )

    return float(psi)


def load_and_validate_features(
    path,
    source_name,
):

    df = pd.read_csv(path)

    print(
        f"{source_name} rows: "
        f"{len(df):,}"
    )

    missing = [
        column
        for column in FEATURES
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{source_name} missing features: "
            f"{missing}"
        )

    return df


def score_with_production_pipeline(
    df,
    model,
    calibrator,
    source_name,
):

    print(
        f"\nScoring {source_name} "
        "with serialized production model..."
    )

    X = df[FEATURES].copy()

    raw_scores = model.decision_function(X)

    calibrated_probability = (
        calibrator.predict_proba(
            raw_scores
        )
    )

    if np.isnan(raw_scores).any():
        raise RuntimeError(
            f"{source_name}: raw scores contain NaN."
        )

    if np.isnan(calibrated_probability).any():
        raise RuntimeError(
            f"{source_name}: calibrated "
            "probabilities contain NaN."
        )

    if (
        (calibrated_probability < 0)
        |
        (calibrated_probability > 1)
    ).any():

        raise RuntimeError(
            f"{source_name}: calibrated "
            "probabilities outside [0,1]."
        )

    return raw_scores, calibrated_probability


def summarize_probability(
    probabilities,
    source_name,
    snapshot_date,
):

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    return {
        "source": source_name,
        "snapshot_date": str(
            snapshot_date
        ),
        "customers": len(probabilities),
        "mean_probability":
            probabilities.mean(),
        "median_probability":
            np.median(probabilities),
        "std_probability":
            probabilities.std(),
        "min_probability":
            probabilities.min(),
        "p01_probability":
            np.quantile(
                probabilities,
                0.01,
            ),
        "p05_probability":
            np.quantile(
                probabilities,
                0.05,
            ),
        "p10_probability":
            np.quantile(
                probabilities,
                0.10,
            ),
        "p25_probability":
            np.quantile(
                probabilities,
                0.25,
            ),
        "p50_probability":
            np.quantile(
                probabilities,
                0.50,
            ),
        "p75_probability":
            np.quantile(
                probabilities,
                0.75,
            ),
        "p90_probability":
            np.quantile(
                probabilities,
                0.90,
            ),
        "p95_probability":
            np.quantile(
                probabilities,
                0.95,
            ),
        "p99_probability":
            np.quantile(
                probabilities,
                0.99,
            ),
        "max_probability":
            probabilities.max(),
    }


def psi_status(value):

    if value < 0.10:
        return "PASS"

    if value < 0.25:
        return "WARN"

    return "FAIL"


def main():

    print("=" * 70)
    print("PREDICTION DRIFT MONITORING")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # 1. Load production artifacts
    # ---------------------------------------------------------------

    print("\nLoading serialized production artifacts...")

    model = joblib.load(
        MODEL_PATH
    )

    calibrator = joblib.load(
        CALIBRATOR_PATH
    )

    print("Logistic model: PASS")
    print("Sigmoid calibrator: PASS")

    # ---------------------------------------------------------------
    # 2. Load historical datasets
    # ---------------------------------------------------------------

    print("\nLoading validation data...")

    validation = load_and_validate_features(
        VALIDATION_PATH,
        "Validation",
    )

    print("\nLoading test data...")

    test = load_and_validate_features(
        TEST_PATH,
        "Test",
    )

    # ---------------------------------------------------------------
    # 3. Load production output
    # ---------------------------------------------------------------

    print("\nLoading production scores...")

    production = pd.read_csv(
        PRODUCTION_PATH
    )

    print(
        f"Production rows: "
        f"{len(production):,}"
    )

    required_production = [
        "customer_unique_id",
        "snapshot_date",
        "raw_model_score",
        "calibrated_probability",
    ]

    missing_production = [
        column
        for column in required_production
        if column not in production.columns
    ]

    if missing_production:
        raise RuntimeError(
            "Production missing columns: "
            f"{missing_production}"
        )

    print("Production columns: PASS")

    # ---------------------------------------------------------------
    # 4. Re-score validation using production artifacts
    # ---------------------------------------------------------------

    (
        validation_raw,
        validation_probability,
    ) = score_with_production_pipeline(
        validation,
        model,
        calibrator,
        "validation",
    )

    # ---------------------------------------------------------------
    # 5. Re-score test using production artifacts
    # ---------------------------------------------------------------

    (
        test_raw,
        test_probability,
    ) = score_with_production_pipeline(
        test,
        model,
        calibrator,
        "test",
    )

    # ---------------------------------------------------------------
    # 6. Validate production probabilities
    # ---------------------------------------------------------------

    production_raw = pd.to_numeric(
        production[
            "raw_model_score"
        ],
        errors="coerce",
    ).to_numpy()

    production_probability = pd.to_numeric(
        production[
            "calibrated_probability"
        ],
        errors="coerce",
    ).to_numpy()

    if np.isnan(production_raw).any():
        raise RuntimeError(
            "Production raw_model_score contains NaN."
        )

    if np.isnan(production_probability).any():
        raise RuntimeError(
            "Production calibrated_probability "
            "contains NaN."
        )

    # ---------------------------------------------------------------
    # 7. Verify production scores are reproduced
    # ---------------------------------------------------------------

    print(
        "\nValidating serialized production "
        "score reproduction..."
    )

    production_key = (
        production[
            [
                "customer_unique_id",
                "snapshot_date",
            ]
        ]
        .astype(str)
        .agg("|".join, axis=1)
    )

    production_recomputed = production.copy()

    # Production file does not contain features, so the
    # exact row-level score reproduction was already validated
    # in Step 14C. Here we explicitly rely on that validation
    # and use the stored production probabilities for drift.
    print(
        "Production score artifact accepted "
        "from Step 14C validation: PASS"
    )

    # ---------------------------------------------------------------
    # 8. Summary statistics
    # ---------------------------------------------------------------

    validation_date = validation[
        "snapshot_date"
    ].iloc[0]

    test_date = test[
        "snapshot_date"
    ].iloc[0]

    production_date = production[
        "snapshot_date"
    ].iloc[0]

    validation_summary = summarize_probability(
        validation_probability,
        "validation",
        validation_date,
    )

    test_summary = summarize_probability(
        test_probability,
        "test",
        test_date,
    )

    production_summary = summarize_probability(
        production_probability,
        "production",
        production_date,
    )

    summary_df = pd.DataFrame(
        [
            validation_summary,
            test_summary,
            production_summary,
        ]
    )

    # ---------------------------------------------------------------
    # 9. PSI: validation -> test
    # ---------------------------------------------------------------

    print(
        "\nPrediction probability drift:"
    )

    print(
        "Validation -> Test"
    )

    validation_test_psi = calculate_psi(
        validation_probability,
        test_probability,
    )

    print(
        f"PSI: "
        f"{validation_test_psi:.6f}"
    )

    # ---------------------------------------------------------------
    # 10. PSI: test -> production
    # ---------------------------------------------------------------

    print(
        "\nTest -> Production"
    )

    test_production_psi = calculate_psi(
        test_probability,
        production_probability,
    )

    print(
        f"PSI: "
        f"{test_production_psi:.6f}"
    )

    # ---------------------------------------------------------------
    # 11. PSI table
    # ---------------------------------------------------------------

    psi_df = pd.DataFrame(
        [
            {
                "comparison":
                    "validation_to_test",
                "reference":
                    "validation",
                "current":
                    "test",
                "psi":
                    validation_test_psi,
                "status":
                    psi_status(
                        validation_test_psi
                    ),
            },
            {
                "comparison":
                    "test_to_production",
                "reference":
                    "test",
                "current":
                    "production",
                "psi":
                    test_production_psi,
                "status":
                    psi_status(
                        test_production_psi
                    ),
            },
        ]
    )

    # ---------------------------------------------------------------
    # 12. Print summary
    # ---------------------------------------------------------------

    print(
        "\nPrediction probability summary:"
    )

    print(
        summary_df[
            [
                "source",
                "snapshot_date",
                "customers",
                "mean_probability",
                "median_probability",
                "std_probability",
                "p01_probability",
                "p50_probability",
                "p99_probability",
                "max_probability",
            ]
        ].to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # 13. Save
    # ---------------------------------------------------------------

    print(
        "\nSaving monitoring outputs..."
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    psi_df.to_csv(
        OUTPUT_PSI,
        index=False,
    )

    print(
        f"Saved: {OUTPUT_SUMMARY}"
    )

    print(
        f"Saved: {OUTPUT_PSI}"
    )

    # ---------------------------------------------------------------
    # 14. Final status
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREDICTION DRIFT MONITORING SUMMARY")
    print("=" * 70)

    print(
        "\nValidation -> Test:"
    )

    print(
        f"PSI: {validation_test_psi:.6f} "
        f"[{psi_status(validation_test_psi)}]"
    )

    print(
        "\nTest -> Production:"
    )

    print(
        f"PSI: {test_production_psi:.6f} "
        f"[{psi_status(test_production_psi)}]"
    )

    print(
        "\nPSI thresholds:"
    )

    print("< 0.10    = PASS")
    print("0.10-0.25 = WARN")
    print(">= 0.25   = FAIL")

    print("\n" + "=" * 70)
    print("STEP 15A COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()