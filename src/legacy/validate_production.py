from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "data" / "model"

REFERENCE_PATH = MODEL_DIR / "final_logistic_test_scored.csv"
PRODUCTION_PATH = MODEL_DIR / "production_scores.csv"


MODEL_VERSION = "logistic_16_sigmoid_v1"

KEY_COLUMNS = [
    "customer_unique_id",
    "snapshot_date",
]


def main():

    print("=" * 70)
    print("PRODUCTION VS EVALUATION VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Load files
    # ---------------------------------------------------------------

    print("\nLoading reference evaluation scores...")

    reference = pd.read_csv(REFERENCE_PATH)

    print(f"Reference rows: {len(reference):,}")

    print("\nLoading production scores...")

    production = pd.read_csv(PRODUCTION_PATH)

    print(f"Production rows: {len(production):,}")

    # ---------------------------------------------------------------
    # 2. Basic validation
    # ---------------------------------------------------------------

    print("\nBasic validation...")

    if len(reference) != len(production):
        raise RuntimeError(
            "Reference and production row counts differ."
        )

    print(f"Row count: {len(production):,}  PASS")

    required_reference = [
        "customer_unique_id",
        "snapshot_date",
        "predicted_probability",
    ]

    required_production = [
        "customer_unique_id",
        "snapshot_date",
        "raw_model_score",
        "calibrated_probability",
        "rank",
        "rank_percentile",
        "target_segment",
        "model_version",
    ]

    missing_reference = [
        col
        for col in required_reference
        if col not in reference.columns
    ]

    missing_production = [
        col
        for col in required_production
        if col not in production.columns
    ]

    if missing_reference:
        raise RuntimeError(
            f"Missing columns in reference: {missing_reference}"
        )

    if missing_production:
        raise RuntimeError(
            f"Missing columns in production: {missing_production}"
        )

    print("Required columns: PASS")

    # ---------------------------------------------------------------
    # 3. Duplicate validation
    # ---------------------------------------------------------------

    print("\nDuplicate validation...")

    reference_duplicates = reference.duplicated(
        KEY_COLUMNS
    ).sum()

    production_duplicates = production.duplicated(
        KEY_COLUMNS
    ).sum()

    if reference_duplicates > 0:
        raise RuntimeError(
            f"Reference contains {reference_duplicates:,} duplicate rows."
        )

    if production_duplicates > 0:
        raise RuntimeError(
            f"Production contains {production_duplicates:,} duplicate rows."
        )

    print("Reference duplicates: 0  PASS")
    print("Production duplicates: 0  PASS")

    # ---------------------------------------------------------------
    # 4. Sort identically
    # ---------------------------------------------------------------

    reference_sorted = (
        reference
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )

    production_sorted = (
        production
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # 5. Customer/snapshot identity
    # ---------------------------------------------------------------

    print("\nCustomer/snapshot identity check...")

    ids_match = (
        reference_sorted[KEY_COLUMNS]
        .astype(str)
        .eq(
            production_sorted[KEY_COLUMNS].astype(str)
        )
        .all()
        .all()
    )

    if not ids_match:
        raise RuntimeError(
            "Customer/snapshot rows do not match."
        )

    print("Customer/snapshot identity: PASS")

    # ---------------------------------------------------------------
    # 6. Validate original evaluation probability
    # ---------------------------------------------------------------

    print("\nReference model probability validation...")

    reference_probability = (
        reference_sorted[
            "predicted_probability"
        ].to_numpy()
    )

    if np.isnan(reference_probability).any():
        raise RuntimeError(
            "Reference predicted_probability contains NaN."
        )

    if (
        (reference_probability < 0)
        |
        (reference_probability > 1)
    ).any():
        raise RuntimeError(
            "Reference predicted_probability contains invalid values."
        )

    print("Reference probabilities in [0, 1]: PASS")

    # ---------------------------------------------------------------
    # 7. Production calibrated probability validation
    # ---------------------------------------------------------------

    print("\nProduction probability validation...")

    production_probability = (
        production_sorted[
            "calibrated_probability"
        ].to_numpy()
    )

    if np.isnan(production_probability).any():
        raise RuntimeError(
            "Production calibrated_probability contains NaN."
        )

    if (
        (production_probability < 0)
        |
        (production_probability > 1)
    ).any():
        raise RuntimeError(
            "Production calibrated_probability contains invalid values."
        )

    print("Production calibrated probabilities in [0, 1]: PASS")

    # ---------------------------------------------------------------
    # 8. Validate raw score
    # ---------------------------------------------------------------

    print("\nRaw model score validation...")

    raw_scores = (
        production_sorted[
            "raw_model_score"
        ].to_numpy()
    )

    if np.isnan(raw_scores).any():
        raise RuntimeError(
            "Production raw_model_score contains NaN."
        )

    print("Raw model scores contain no NaN: PASS")

    # ---------------------------------------------------------------
    # 9. Important distinction
    # ---------------------------------------------------------------

    print("\nEvaluation vs production probability distinction...")

    print(
        "Reference predicted_probability = original "
        "uncalibrated model probability"
    )

    print(
        "Production calibrated_probability = "
        "sigmoid/Platt-calibrated probability"
    )

    print(
        "Direct numerical equality is NOT expected "
        "because these are different probability layers."
    )

    print("Probability-layer distinction: PASS")

    # ---------------------------------------------------------------
    # 10. Ranking reconstruction
    # ---------------------------------------------------------------

    print("\nRanking reconstruction...")

    reconstructed_rank = (
        production_sorted
        .groupby("snapshot_date")[
            "calibrated_probability"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
        .to_numpy()
    )

    stored_rank = (
        production_sorted["rank"]
        .to_numpy()
    )

    if not np.array_equal(
        reconstructed_rank,
        stored_rank,
    ):
        raise RuntimeError(
            "Stored production ranking cannot be reconstructed."
        )

    print("Stored ranking reconstruction: PASS")

    # ---------------------------------------------------------------
    # 11. Rank percentile
    # ---------------------------------------------------------------

    print("\nRank percentile validation...")

    snapshot_sizes = (
        production_sorted
        .groupby("snapshot_date")[
            "customer_unique_id"
        ]
        .transform("size")
        .to_numpy()
    )

    expected_percentile = (
        stored_rank / snapshot_sizes
    )

    stored_percentile = (
        production_sorted[
            "rank_percentile"
        ].to_numpy()
    )

    percentile_difference = np.abs(
        expected_percentile
        -
        stored_percentile
    )

    max_percentile_difference = (
        percentile_difference.max()
    )

    print(
        f"Max percentile difference: "
        f"{max_percentile_difference:.12f}"
    )

    if max_percentile_difference > 1e-12:
        raise RuntimeError(
            "Rank percentile calculation mismatch."
        )

    print("Rank percentile: PASS")

    # ---------------------------------------------------------------
    # 12. Rank uniqueness
    # ---------------------------------------------------------------

    print("\nRank uniqueness validation...")

    rank_summary = (
        production_sorted
        .groupby("snapshot_date")["rank"]
        .agg(
            ["min", "max", "count", "nunique"]
        )
    )

    print(rank_summary.to_string())

    if not (
        (rank_summary["min"] == 1)
        &
        (
            rank_summary["max"]
            == rank_summary["count"]
        )
        &
        (
            rank_summary["nunique"]
            == rank_summary["count"]
        )
    ).all():

        raise RuntimeError(
            "Rank sequence is not exactly 1..N within snapshot."
        )

    print("Exact rank sequence: PASS")

    # ---------------------------------------------------------------
    # 13. Top-K consistency against reference ranking
    # ---------------------------------------------------------------

    print("\nTop-K ranking consistency...")

    reference_rank = (
        reference_sorted[
            "predicted_probability"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    production_rank = (
        production_sorted[
            "rank"
        ]
    )

    reference_rank_map = pd.Series(
        reference_rank.to_numpy(),
        index=reference_sorted["customer_unique_id"],
    )

    production_rank_map = pd.Series(
        production_rank.to_numpy(),
        index=production_sorted["customer_unique_id"],
    )

    top_k_values = [
        0.005,
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
    ]

    total_rows = len(reference_sorted)

    for fraction in top_k_values:

        cutoff = int(
            np.ceil(total_rows * fraction)
        )

        reference_top_k = set(
            reference_rank_map[
                reference_rank_map <= cutoff
            ].index
        )

        production_top_k = set(
            production_rank_map[
                production_rank_map <= cutoff
            ].index
        )

        overlap = (
            len(
                reference_top_k
                &
                production_top_k
            )
            / cutoff
        )

        print(
            f"Top {fraction * 100:.1f}%: "
            f"{len(reference_top_k):,} reference, "
            f"{len(production_top_k):,} production, "
            f"overlap={overlap:.6f}"
        )

        if overlap < 0.999999:
            raise RuntimeError(
                f"Top-{fraction * 100:.1f}% "
                "ranking does not match the reference evaluation."
            )

        print(
            f"Top {fraction * 100:.1f}%: PASS"
        )

    # ---------------------------------------------------------------
    # 14. Model version
    # ---------------------------------------------------------------

    print("\nModel version check...")

    versions = (
        production_sorted[
            "model_version"
        ]
        .dropna()
        .unique()
    )

    print(
        f"Model versions found: {list(versions)}"
    )

    if len(versions) != 1:
        raise RuntimeError(
            "Multiple model versions found."
        )

    if versions[0] != MODEL_VERSION:
        raise RuntimeError(
            "Unexpected model version."
        )

    print("Model version: PASS")

    # ---------------------------------------------------------------
    # 15. Target segment validation
    # ---------------------------------------------------------------

    print("\nTarget segment validation...")

    allowed_segments = {
        "top_0.5pct",
        "0.5_1pct",
        "1_2pct",
        "2_5pct",
        "5_10pct",
        "10_20pct",
        "bottom_80pct",
    }

    actual_segments = set(
        production_sorted[
            "target_segment"
        ].dropna().unique()
    )

    unexpected_segments = (
        actual_segments - allowed_segments
    )

    if unexpected_segments:
        raise RuntimeError(
            f"Unexpected target segments: "
            f"{unexpected_segments}"
        )

    print("Target segment labels: PASS")

    # ---------------------------------------------------------------
    # 16. Final summary
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("PRODUCTION VALIDATION SUMMARY")
    print("=" * 70)

    print("\nRow count:                    PASS")
    print("Required columns:            PASS")
    print("Duplicate validation:        PASS")
    print("Customer/snapshot IDs:       PASS")
    print("Reference probabilities:     PASS")
    print("Production probabilities:    PASS")
    print("Raw model scores:            PASS")
    print("Probability-layer distinction: PASS")
    print("Ranking reconstruction:      PASS")
    print("Rank percentile:             PASS")
    print("Exact rank sequence:         PASS")
    print("Top-K ranking consistency:   PASS")
    print("Model version:               PASS")
    print("Target segments:             PASS")

    print("\n" + "=" * 70)
    print("STEP 14C COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()