from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "model" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "model" / "validation.csv"

EXPLANATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "model"
    / "logistic_validation_local_explanations.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "model"

TARGET = "future_purchase_flag"


# ============================================================
# LOCKED LOGISTIC 16
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


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOGISTIC SEGMENT DRIVER ANALYSIS")
print("=" * 70)

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)
explanations = pd.read_csv(EXPLANATION_PATH)


# ============================================================
# VALIDATE
# ============================================================

missing_train = [
    f for f in FEATURES
    if f not in train.columns
]

missing_validation = [
    f for f in FEATURES
    if f not in validation.columns
]

if missing_train:
    raise ValueError(
        f"Missing train features: {missing_train}"
    )

if missing_validation:
    raise ValueError(
        f"Missing validation features: {missing_validation}"
    )

if "model_rank" not in explanations.columns:
    raise ValueError(
        "Expected model_rank in local explanations file."
    )


# ============================================================
# CREATE RANKED VALIDATION DATASET
# ============================================================

ranked = validation[
    [
        "customer_unique_id",
        "snapshot_date",
        TARGET,
    ]
    + FEATURES
].copy()

ranked = ranked.merge(
    explanations[
        [
            "customer_unique_id",
            "snapshot_date",
            "model_rank",
            "raw_model_probability",
            "calibrated_probability",
        ]
    ],
    on=[
        "customer_unique_id",
        "snapshot_date",
    ],
    how="left",
)


if ranked["model_rank"].isna().any():
    raise ValueError(
        "Some validation customers do not have model explanations."
    )


# ============================================================
# DEFINE SEGMENTS
# ============================================================

n = len(ranked)

ranked["segment"] = np.select(
    [
        ranked["model_rank"] <= int(n * 0.01),

        ranked["model_rank"] <= int(n * 0.05),

        ranked["model_rank"] <= int(n * 0.10),

        ranked["model_rank"] <= int(n * 0.20),
    ],
    [
        "Top 1%",

        "Top 1-5%",

        "Top 5-10%",

        "Top 10-20%",
    ],
    default="Bottom 80%",
)


# ============================================================
# SEGMENT ORDER
# ============================================================

SEGMENT_ORDER = [
    "Top 1%",
    "Top 1-5%",
    "Top 5-10%",
    "Top 10-20%",
    "Bottom 80%",
]


ranked["segment"] = pd.Categorical(
    ranked["segment"],
    categories=SEGMENT_ORDER,
    ordered=True,
)


# ============================================================
# TRAINING DISTRIBUTION
# ============================================================

train_mean = train[FEATURES].mean()

train_std = (
    train[FEATURES]
    .std()
    .replace(0, np.nan)
)


# ============================================================
# SEGMENT SUMMARY
# ============================================================

segment_summary_rows = []

for segment in SEGMENT_ORDER:

    group = ranked[
        ranked["segment"] == segment
    ]

    row = {
        "segment": segment,
        "customers": len(group),
        "buyers": int(
            group[TARGET].sum()
        ),
        "purchase_rate": group[TARGET].mean(),
        "mean_model_score": group[
            "raw_model_probability"
        ].mean(),
        "mean_calibrated_probability": group[
            "calibrated_probability"
        ].mean(),
    }

    for feature in FEATURES:

        segment_mean = group[feature].mean()

        row[f"{feature}_mean"] = segment_mean

        row[
            f"{feature}_std_diff"
        ] = (
            segment_mean
            - train_mean[feature]
        ) / train_std[feature]

    segment_summary_rows.append(row)


segment_summary = pd.DataFrame(
    segment_summary_rows
)


# ============================================================
# FEATURE DRIVER TABLE
# ============================================================

driver_rows = []


for segment in SEGMENT_ORDER:

    group = ranked[
        ranked["segment"] == segment
    ]

    for feature in FEATURES:

        segment_mean = group[feature].mean()

        standardized_difference = (
            segment_mean
            - train_mean[feature]
        ) / train_std[feature]

        driver_rows.append(
            {
                "segment": segment,

                "feature": feature,

                "segment_mean": segment_mean,

                "train_mean": train_mean[feature],

                "train_std": train_std[feature],

                "standardized_difference": (
                    standardized_difference
                ),

                "absolute_standardized_difference": abs(
                    standardized_difference
                ),
            }
        )


driver_table = pd.DataFrame(
    driver_rows
)


# ============================================================
# TOP SEGMENT DRIVERS
# ============================================================

print("\n" + "=" * 70)
print("TOP 1% DRIVER PROFILE")
print("=" * 70)

top1_drivers = (
    driver_table[
        driver_table["segment"]
        == "Top 1%"
    ]
    .sort_values(
        "absolute_standardized_difference",
        ascending=False,
    )
)

print(
    top1_drivers[
        [
            "feature",
            "segment_mean",
            "train_mean",
            "standardized_difference",
        ]
    ]
    .head(10)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# ALL SEGMENT DRIVER RANKINGS
# ============================================================

print("\n" + "=" * 70)
print("TOP DRIVERS BY SEGMENT")
print("=" * 70)

for segment in SEGMENT_ORDER:

    print("\n" + "-" * 70)
    print(segment)

    segment_drivers = (
        driver_table[
            driver_table["segment"]
            == segment
        ]
        .sort_values(
            "absolute_standardized_difference",
            ascending=False,
        )
        .head(5)
    )

    print(
        segment_drivers[
            [
                "feature",
                "standardized_difference",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# ============================================================
# SEGMENT BUSINESS METRICS
# ============================================================

print("\n" + "=" * 70)
print("SEGMENT BUSINESS METRICS")
print("=" * 70)

business_metrics = segment_summary[
    [
        "segment",
        "customers",
        "buyers",
        "purchase_rate",
        "mean_model_score",
        "mean_calibrated_probability",
    ]
]

print(
    business_metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# POSITIVE / NEGATIVE SEGMENT SHIFTS
# ============================================================

positive_shift_rows = []
negative_shift_rows = []


for segment in SEGMENT_ORDER:

    segment_drivers = driver_table[
        driver_table["segment"]
        == segment
    ]

    positives = (
        segment_drivers[
            segment_drivers[
                "standardized_difference"
            ] > 0
        ]
        .sort_values(
            "standardized_difference",
            ascending=False,
        )
        .head(5)
    )

    negatives = (
        segment_drivers[
            segment_drivers[
                "standardized_difference"
            ] < 0
        ]
        .sort_values(
            "standardized_difference",
            ascending=True,
        )
        .head(5)
    )

    for _, row in positives.iterrows():

        positive_shift_rows.append(
            {
                "segment": segment,
                "feature": row["feature"],
                "standardized_difference": row[
                    "standardized_difference"
                ],
            }
        )

    for _, row in negatives.iterrows():

        negative_shift_rows.append(
            {
                "segment": segment,
                "feature": row["feature"],
                "standardized_difference": row[
                    "standardized_difference"
                ],
            }
        )


positive_shifts = pd.DataFrame(
    positive_shift_rows
)

negative_shifts = pd.DataFrame(
    negative_shift_rows
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


segment_summary_path = (
    OUTPUT_DIR
    / "logistic_segment_driver_summary.csv"
)

driver_table_path = (
    OUTPUT_DIR
    / "logistic_segment_feature_shifts.csv"
)

positive_shift_path = (
    OUTPUT_DIR
    / "logistic_segment_positive_shifts.csv"
)

negative_shift_path = (
    OUTPUT_DIR
    / "logistic_segment_negative_shifts.csv"
)


segment_summary.to_csv(
    segment_summary_path,
    index=False,
)

driver_table.to_csv(
    driver_table_path,
    index=False,
)

positive_shifts.to_csv(
    positive_shift_path,
    index=False,
)

negative_shifts.to_csv(
    negative_shift_path,
    index=False,
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(segment_summary_path)
print(driver_table_path)
print(positive_shift_path)
print(negative_shift_path)

print("\nSegment driver analysis complete.")