from pathlib import Path
from pathlib import Path
import json

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

# STEP 14B DEMONSTRATION INPUT
# Using held-out test.csv only for inference/scoring.
# No retraining or recalibration is performed here.
INPUT_PATH = BASE_DIR / "data" / "model" / "test.csv"

ARTIFACT_DIR = BASE_DIR / "data" / "model" / "artifacts"
OUTPUT_DIR = BASE_DIR / "data" / "model"

MODEL_PATH = ARTIFACT_DIR / "logistic_model.joblib"
CALIBRATOR_PATH = ARTIFACT_DIR / "sigmoid_calibrator.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

OUTPUT_PATH = OUTPUT_DIR / "production_scores.csv"


def assign_segment(percentile):
    """
    Assign non-overlapping targeting bands.

    These are bands, not cumulative top-X% groups.
    """

    if percentile <= 0.005:
        return "top_0.5pct"

    elif percentile <= 0.01:
        return "0.5_1pct"

    elif percentile <= 0.02:
        return "1_2pct"

    elif percentile <= 0.05:
        return "2_5pct"

    elif percentile <= 0.10:
        return "5_10pct"

    elif percentile <= 0.20:
        return "10_20pct"

    else:
        return "bottom_80pct"


def main():

    print("=" * 70)
    print("PRODUCTION CUSTOMER SCORING")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. LOAD PRODUCTION ARTIFACTS
    # ---------------------------------------------------------------

    print("\nLoading production artifacts...")

    model = joblib.load(MODEL_PATH)
    calibrator = joblib.load(CALIBRATOR_PATH)

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    features = metadata["features"]
    model_version = metadata["model_version"]

    print(f"Model version: {model_version}")
    print(f"Feature count: {len(features)}")

    # ---------------------------------------------------------------
    # 2. LOAD SCORING INPUT
    # ---------------------------------------------------------------

    print("\nLoading scoring input...")

    df = pd.read_csv(INPUT_PATH)

    print(f"Input rows: {len(df):,}")

    print(
        f"Unique customers: "
        f"{df['customer_unique_id'].nunique():,}"
    )

    print(
        f"Snapshots: "
        f"{df['snapshot_date'].nunique():,}"
    )

    # ---------------------------------------------------------------
    # 3. VALIDATE REQUIRED COLUMNS
    # ---------------------------------------------------------------

    required_columns = [
        "customer_unique_id",
        "snapshot_date",
    ] + features

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Validate customer/snapshot grain
    if df[
        [
            "customer_unique_id",
            "snapshot_date",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate customer/snapshot rows detected in input."
        )

    X = df[features]

    # ---------------------------------------------------------------
    # 4. GENERATE RAW MODEL SCORES
    # ---------------------------------------------------------------

    print("\nGenerating raw model scores...")

    raw_scores = model.decision_function(X)

    # ---------------------------------------------------------------
    # 5. GENERATE CALIBRATED PROBABILITIES
    # ---------------------------------------------------------------

    print("Generating calibrated probabilities...")

    calibrated_probability = calibrator.predict_proba(
        raw_scores
    )

    # ---------------------------------------------------------------
    # 6. BUILD OUTPUT
    # ---------------------------------------------------------------

    result = df[
        [
            "customer_unique_id",
            "snapshot_date",
        ]
    ].copy()

    result["raw_model_score"] = raw_scores

    result["calibrated_probability"] = (
        calibrated_probability
    )

    # ---------------------------------------------------------------
    # 7. RANK CUSTOMERS WITHIN EACH SNAPSHOT
    # ---------------------------------------------------------------

    print("\nRanking customers within each snapshot...")

    result["rank"] = (
        result
        .groupby("snapshot_date")["calibrated_probability"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    snapshot_sizes = (
        result
        .groupby("snapshot_date")["customer_unique_id"]
        .transform("size")
    )

    result["rank_percentile"] = (
        result["rank"] / snapshot_sizes
    )

    # ---------------------------------------------------------------
    # 8. ASSIGN TARGETING BANDS
    # ---------------------------------------------------------------

    result["target_segment"] = (
        result["rank_percentile"]
        .apply(assign_segment)
    )

    result["model_version"] = model_version

    result = (
        result
        .sort_values(
            [
                "snapshot_date",
                "rank",
            ]
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # 9. VALIDATION CHECKS
    # ---------------------------------------------------------------

    print("\nValidation checks...")

    # Row count must remain unchanged
    if len(result) != len(df):

        raise RuntimeError(
            "Row count changed during scoring."
        )

    # Customer/snapshot grain must remain unique
    if result[
        [
            "customer_unique_id",
            "snapshot_date",
        ]
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate customer/snapshot rows detected."
        )

    # No null probabilities
    if result[
        "calibrated_probability"
    ].isna().any():

        raise RuntimeError(
            "Null calibrated probabilities detected."
        )

    # Probabilities must be between 0 and 1
    if (
        (result["calibrated_probability"] < 0)
        | (result["calibrated_probability"] > 1)
    ).any():

        raise RuntimeError(
            "Invalid calibrated probabilities detected."
        )

    # ---------------------------------------------------------------
    # 10. RANK VALIDATION
    # ---------------------------------------------------------------

    rank_check = (
        result
        .groupby("snapshot_date")["rank"]
        .agg(
            [
                "min",
                "max",
                "count",
                "nunique",
            ]
        )
    )

    print("\nRank validation:")
    print(rank_check)

    # Validate ranks are exactly 1...N within every snapshot
    for snapshot_date, group in result.groupby(
        "snapshot_date"
    ):

        expected_ranks = set(
            range(1, len(group) + 1)
        )

        actual_ranks = set(
            group["rank"]
        )

        if actual_ranks != expected_ranks:

            raise RuntimeError(
                f"Invalid ranking detected for snapshot "
                f"{snapshot_date}."
            )

    # ---------------------------------------------------------------
    # 11. SNAPSHOT SUMMARY
    # ---------------------------------------------------------------

    print("\nSnapshot summary:")

    snapshot_summary = (
        result
        .groupby("snapshot_date")
        .agg(
            customers=(
                "customer_unique_id",
                "size",
            ),
            mean_probability=(
                "calibrated_probability",
                "mean",
            ),
            max_probability=(
                "calibrated_probability",
                "max",
            ),
        )
        .reset_index()
    )

    print(
        snapshot_summary.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # 12. TARGET SEGMENT SUMMARY
    # ---------------------------------------------------------------

    print("\nTarget segment counts:")

    segment_summary = (
        result
        .groupby(
            [
                "snapshot_date",
                "target_segment",
            ]
        )
        .size()
        .reset_index(
            name="customers"
        )
    )

    print(
        segment_summary.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # 13. SAVE PRODUCTION SCORES
    # ---------------------------------------------------------------

    print("\nSaving production scores...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"\nSaved: {OUTPUT_PATH}")

    # ---------------------------------------------------------------
    # 14. OUTPUT COLUMNS
    # ---------------------------------------------------------------

    print("\nOutput columns:")

    for column in result.columns:
        print(f"  - {column}")

    # ---------------------------------------------------------------
    # COMPLETE
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 14B COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
