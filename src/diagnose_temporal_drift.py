import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "model"
ARTIFACT = MODEL_DIR / "artifacts"

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
SNAPSHOT = "snapshot_date"


# ================================================================
# VALIDATE CANONICAL METADATA
# ================================================================

metadata = json.loads(
    (ARTIFACT / "model_metadata.json").read_text()
)

assert metadata["model_version"] == "logistic_16_sigmoid_v1"
assert metadata["features"] == FEATURES
assert metadata["feature_count"] == 16


# ================================================================
# LOAD EXISTING MODEL DATA ONLY
# ================================================================

train = pd.read_csv(MODEL_DIR / "train.csv")
validation = pd.read_csv(MODEL_DIR / "validation.csv")
test = pd.read_csv(MODEL_DIR / "test.csv")

df = pd.concat(
    [train, validation, test],
    ignore_index=True
)

df[SNAPSHOT] = pd.to_datetime(df[SNAPSHOT])


# ================================================================
# POPULATION SUMMARY
# ================================================================

population_rows = []

for snapshot, group in df.groupby(SNAPSHOT, sort=True):

    population_rows.append(
        {
            "snapshot_date": snapshot.date().isoformat(),
            "rows": len(group),
            "buyers": int(group[TARGET].sum()),
            "base_rate": float(group[TARGET].mean()),
        }
    )

population = pd.DataFrame(population_rows)

population.to_csv(
    MODEL_DIR / "temporal_drift_population_summary.csv",
    index=False
)


# ================================================================
# PSI FUNCTION
# ================================================================

def psi_numeric(reference, current, bins=10):

    reference = (
        pd.Series(reference)
        .dropna()
        .astype(float)
    )

    current = (
        pd.Series(current)
        .dropna()
        .astype(float)
    )

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)

    edges = np.unique(
        reference.quantile(quantiles).values
    )

    if len(edges) < 3:
        return 0.0

    reference_counts, _ = np.histogram(
        reference,
        bins=edges
    )

    current_counts, _ = np.histogram(
        current,
        bins=edges
    )

    reference_pct = (
        reference_counts /
        max(reference_counts.sum(), 1)
    )

    current_pct = (
        current_counts /
        max(current_counts.sum(), 1)
    )

    epsilon = 1e-6

    reference_pct = np.clip(
        reference_pct,
        epsilon,
        None
    )

    current_pct = np.clip(
        current_pct,
        epsilon,
        None
    )

    return float(
        np.sum(
            (current_pct - reference_pct)
            * np.log(current_pct / reference_pct)
        )
    )


# ================================================================
# REFERENCE SNAPSHOT
# ================================================================

reference_date = pd.Timestamp("2017-09-01")

reference = df[
    df[SNAPSHOT] == reference_date
]

assert len(reference) > 0, (
    "Reference snapshot 2017-09-01 missing"
)


# ================================================================
# FEATURE DRIFT
# ================================================================

records = []

for current_date in sorted(df[SNAPSHOT].unique()):

    current = df[
        df[SNAPSHOT] == current_date
    ]

    for feature in FEATURES:

        reference_values = reference[feature]
        current_values = current[feature]

        records.append(
            {
                "reference_snapshot":
                    reference_date.date().isoformat(),

                "current_snapshot":
                    pd.Timestamp(
                        current_date
                    ).date().isoformat(),

                "feature":
                    feature,

                "reference_missing_pct":
                    float(
                        reference_values.isna().mean()
                        * 100
                    ),

                "current_missing_pct":
                    float(
                        current_values.isna().mean()
                        * 100
                    ),

                "missingness_change_pct_points":
                    float(
                        (
                            current_values.isna().mean()
                            -
                            reference_values.isna().mean()
                        )
                        * 100
                    ),

                "reference_mean":
                    float(
                        reference_values.mean()
                    ),

                "current_mean":
                    float(
                        current_values.mean()
                    ),

                "reference_median":
                    float(
                        reference_values.median()
                    ),

                "current_median":
                    float(
                        current_values.median()
                    ),

                "psi":
                    psi_numeric(
                        reference_values,
                        current_values
                    ),
            }
        )


drift = pd.DataFrame(records)


# ================================================================
# PSI INTERPRETATION
# ================================================================

def psi_label(value):

    if pd.isna(value):
        return "unavailable"

    if value < 0.10:
        return "low"

    if value < 0.25:
        return "moderate"

    return "high"


drift["psi_level"] = drift["psi"].apply(
    psi_label
)


drift.to_csv(
    MODEL_DIR / "temporal_feature_drift_psi.csv",
    index=False
)


# ================================================================
# MISSINGNESS SUMMARY
# ================================================================

missing_rows = []

for snapshot, group in df.groupby(
    SNAPSHOT,
    sort=True
):

    for feature in FEATURES:

        missing_rows.append(
            {
                "snapshot_date":
                    snapshot.date().isoformat(),

                "feature":
                    feature,

                "missing_count":
                    int(
                        group[feature].isna().sum()
                    ),

                "missing_pct":
                    float(
                        group[feature].isna().mean()
                        * 100
                    ),
            }
        )


missing = pd.DataFrame(missing_rows)

missing.to_csv(
    MODEL_DIR / "temporal_missingness_summary.csv",
    index=False
)


# ================================================================
# CONSOLE REPORT
# ================================================================

print("=" * 70)
print("STEP 23 — TEMPORAL DRIFT & STABILITY DIAGNOSTIC")
print("=" * 70)

print("Frozen model: logistic_16_sigmoid_v1")
print("Model retrained: NO")
print("Model recalibrated: NO")
print()

print("POPULATION BY SNAPSHOT")
print(
    population.to_string(index=False)
)

print()

print(
    "FEATURE DRIFT AGAINST 2017-09-01"
)

summary = (
    drift[
        drift["current_snapshot"]
        != reference_date.date().isoformat()
    ]
    .sort_values(
        ["current_snapshot", "psi"],
        ascending=[True, False]
    )
)

print(
    summary[
        [
            "current_snapshot",
            "feature",
            "psi",
            "psi_level",
            "reference_mean",
            "current_mean",
            "missingness_change_pct_points",
        ]
    ].to_string(index=False)
)

print()

print("TOP DRIFTED FEATURES BY SNAPSHOT")

for snapshot in sorted(
    summary["current_snapshot"].unique()
):

    top = (
        summary[
            summary["current_snapshot"]
            == snapshot
        ]
        .head(5)
    )

    print()
    print(snapshot)

    print(
        top[
            [
                "feature",
                "psi",
                "psi_level",
            ]
        ].to_string(index=False)
    )


print()
print("Artifacts written:")

print(
    MODEL_DIR
    / "temporal_drift_population_summary.csv"
)

print(
    MODEL_DIR
    / "temporal_feature_drift_psi.csv"
)

print(
    MODEL_DIR
    / "temporal_missingness_summary.csv"
)