from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "data" / "model"
OUTPUT_DIR = MODEL_DIR / "monitoring"

VALIDATION_PATH = MODEL_DIR / "logistic_validation_scored.csv"
TEST_PATH = MODEL_DIR / "logistic_test_scored.csv"

OUTPUT_PSI = OUTPUT_DIR / "feature_drift_psi.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "feature_drift_summary.csv"


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

    reference = pd.to_numeric(
        pd.Series(reference),
        errors="coerce",
    ).to_numpy(dtype=float)

    current = pd.to_numeric(
        pd.Series(current),
        errors="coerce",
    ).to_numpy(dtype=float)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

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

    # Constant feature
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
        reference_counts / len(reference)
    )

    current_pct = (
        current_counts / len(current)
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
            current_pct - reference_pct
        )
        *
        np.log(
            current_pct / reference_pct
        )
    )

    return float(psi)


def psi_status(psi):

    if psi < 0.10:
        return "PASS"

    if psi < 0.25:
        return "WARN"

    return "FAIL"


def main():

    print("=" * 70)
    print("FEATURE DRIFT MONITORING")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # 1. Load datasets
    # ---------------------------------------------------------------

    print("\nLoading validation data...")

    validation = pd.read_csv(
        VALIDATION_PATH
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    print("\nLoading test data...")

    test = pd.read_csv(
        TEST_PATH
    )

    print(
        f"Test rows: "
        f"{len(test):,}"
    )

    # ---------------------------------------------------------------
    # 2. Validate feature availability
    # ---------------------------------------------------------------

    missing_validation = [
        feature
        for feature in FEATURES
        if feature not in validation.columns
    ]

    missing_test = [
        feature
        for feature in FEATURES
        if feature not in test.columns
    ]

    if missing_validation:
        raise RuntimeError(
            "Validation missing features: "
            f"{missing_validation}"
        )

    if missing_test:
        raise RuntimeError(
            "Test missing features: "
            f"{missing_test}"
        )

    print(
        "\nLocked 16 features available: PASS"
    )

    # ---------------------------------------------------------------
    # 3. Check missingness
    # ---------------------------------------------------------------

    print(
        "\nChecking feature missingness..."
    )

    rows = []

    for feature in FEATURES:

        validation_missing = (
            validation[feature]
            .isna()
            .mean()
        )

        test_missing = (
            test[feature]
            .isna()
            .mean()
        )

        rows.append(
            {
                "feature": feature,
                "validation_missing_pct":
                    validation_missing * 100,
                "test_missing_pct":
                    test_missing * 100,
                "missing_pct_change":
                    (
                        test_missing
                        - validation_missing
                    )
                    * 100,
            }
        )

    missingness_df = pd.DataFrame(rows)

    # ---------------------------------------------------------------
    # 4. Calculate PSI
    # ---------------------------------------------------------------

    print(
        "\nCalculating feature PSI..."
    )

    psi_rows = []

    for feature in FEATURES:

        validation_values = validation[
            feature
        ]

        test_values = test[
            feature
        ]

        validation_non_null = (
            pd.to_numeric(
                validation_values,
                errors="coerce",
            )
            .dropna()
        )

        test_non_null = (
            pd.to_numeric(
                test_values,
                errors="coerce",
            )
            .dropna()
        )

        psi = calculate_psi(
            validation_non_null,
            test_non_null,
        )

        validation_mean = (
            validation_non_null.mean()
        )

        test_mean = (
            test_non_null.mean()
        )

        validation_median = (
            validation_non_null.median()
        )

        test_median = (
            test_non_null.median()
        )

        psi_rows.append(
            {
                "feature": feature,
                "validation_mean":
                    validation_mean,
                "test_mean":
                    test_mean,
                "mean_change":
                    test_mean
                    - validation_mean,
                "validation_median":
                    validation_median,
                "test_median":
                    test_median,
                "median_change":
                    test_median
                    - validation_median,
                "psi":
                    psi,
                "status":
                    psi_status(psi),
            }
        )

    psi_df = pd.DataFrame(
        psi_rows
    )

    # ---------------------------------------------------------------
    # 5. Merge missingness
    # ---------------------------------------------------------------

    summary_df = psi_df.merge(
        missingness_df,
        on="feature",
        how="left",
    )

    summary_df = summary_df.sort_values(
        "psi",
        ascending=False,
    ).reset_index(drop=True)

    # ---------------------------------------------------------------
    # 6. Print results
    # ---------------------------------------------------------------

    print(
        "\nFeature PSI:"
    )

    print(
        summary_df[
            [
                "feature",
                "psi",
                "status",
                "validation_missing_pct",
                "test_missing_pct",
                "missing_pct_change",
            ]
        ].to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # 7. Overall statistics
    # ---------------------------------------------------------------

    pass_count = (
        summary_df["status"] == "PASS"
    ).sum()

    warn_count = (
        summary_df["status"] == "WARN"
    ).sum()

    fail_count = (
        summary_df["status"] == "FAIL"
    ).sum()

    max_psi_row = summary_df.iloc[0]

    mean_psi = (
        summary_df["psi"].mean()
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FEATURE DRIFT SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Features monitored: "
        f"{len(summary_df)}"
    )

    print(
        f"PASS: {pass_count}"
    )

    print(
        f"WARN: {warn_count}"
    )

    print(
        f"FAIL: {fail_count}"
    )

    print(
        f"Mean PSI: "
        f"{mean_psi:.6f}"
    )

    print(
        f"Maximum PSI: "
        f"{max_psi_row['psi']:.6f}"
    )

    print(
        f"Highest-drift feature: "
        f"{max_psi_row['feature']}"
    )

    # ---------------------------------------------------------------
    # 8. Save
    # ---------------------------------------------------------------

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    psi_df.to_csv(
        OUTPUT_PSI,
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_SUMMARY
    )

    print(
        OUTPUT_PSI
    )

    print(
        "\nPSI thresholds:"
    )

    print(
        "< 0.10    = PASS"
    )

    print(
        "0.10-0.25 = WARN"
    )

    print(
        ">= 0.25   = FAIL"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "STEP 15B COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()