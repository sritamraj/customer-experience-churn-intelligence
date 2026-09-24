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


KEY_COLUMNS = [
    "customer_unique_id",
    "snapshot_date",
]


EXPLANATION_COLUMNS = [
    "customer_unique_id",
    "snapshot_date",
    "model_rank",
    "raw_model_probability",
    "calibrated_probability",
]


SEGMENT_ORDER = [
    "Top 1%",
    "Top 1-5%",
    "Top 5-10%",
    "Top 10-20%",
    "Bottom 80%",
]


# ============================================================
# START
# ============================================================

print("=" * 70)
print("LOGISTIC SEGMENT DRIVER ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading data...")

train = pd.read_csv(TRAIN_PATH)
validation = pd.read_csv(VALIDATION_PATH)
explanations = pd.read_csv(EXPLANATION_PATH)

print(f"Train rows: {len(train)}")
print(f"Validation rows: {len(validation)}")
print(f"Explanation rows: {len(explanations)}")


# ============================================================
# PRODUCTION CONTRACT
# ============================================================

print("\n" + "=" * 70)
print("PRODUCTION MODEL CONTRACT")
print("=" * 70)

print("Model version: logistic_16_sigmoid_v1")
print("Locked features: 16")
print("Calibration: frozen production artifact")
print("Retraining: NO")
print("Recalibration: NO")
print("Test data used: NO")


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("BASIC VALIDATION")
print("=" * 70)


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

missing_explanation = [
    c for c in EXPLANATION_COLUMNS
    if c not in explanations.columns
]

if missing_explanation:
    raise ValueError(
        "Missing explanation columns: "
        f"{missing_explanation}"
    )

if TARGET not in validation.columns:
    raise ValueError(
        f"Missing target column: {TARGET}"
    )

print("Feature contract: PASS")
print("Explanation columns: PASS")


# ============================================================
# KEY VALIDATION
# ============================================================

train_duplicates = train.duplicated(
    KEY_COLUMNS
).sum()

validation_duplicates = validation.duplicated(
    KEY_COLUMNS
).sum()

explanation_duplicates = explanations.duplicated(
    KEY_COLUMNS
).sum()


if train_duplicates:
    raise ValueError(
        f"Train duplicate keys: {train_duplicates}"
    )

if validation_duplicates:
    raise ValueError(
        f"Validation duplicate keys: {validation_duplicates}"
    )

if explanation_duplicates:
    raise ValueError(
        f"Explanation duplicate keys: {explanation_duplicates}"
    )

print("Train customer-snapshot uniqueness: PASS")
print("Validation customer-snapshot uniqueness: PASS")
print("Explanation customer-snapshot uniqueness: PASS")


# ============================================================
# ROW COUNT VALIDATION
# ============================================================

if len(validation) != len(explanations):
    raise ValueError(
        "Validation/explanation row mismatch: "
        f"{len(validation)} vs {len(explanations)}"
    )

print("Validation/explanation row count: PASS")


# ============================================================
# CREATE RANKED DATASET
# ============================================================

ranked = validation[
    KEY_COLUMNS + [TARGET] + FEATURES
].copy()


ranked = ranked.merge(
    explanations[EXPLANATION_COLUMNS],
    on=KEY_COLUMNS,
    how="left",
    validate="one_to_one",
)


# ============================================================
# MERGE VALIDATION
# ============================================================

for column in [
    "model_rank",
    "raw_model_probability",
    "calibrated_probability",
]:

    missing_count = ranked[column].isna().sum()

    if missing_count:
        raise ValueError(
            f"{missing_count} rows missing {column}."
        )

print("Explanation merge: PASS")


# ============================================================
# RANK VALIDATION
# ============================================================

n = len(ranked)

ranks = (
    ranked["model_rank"]
    .astype(int)
    .sort_values()
    .to_numpy()
)

expected_ranks = np.arange(
    1,
    n + 1,
)

if not np.array_equal(
    ranks,
    expected_ranks,
):
    raise ValueError(
        "model_rank is not exactly 1..N."
    )

print("Model ranking integrity: PASS")


# ============================================================
# PROBABILITY VALIDATION
# ============================================================

for column in [
    "raw_model_probability",
    "calibrated_probability",
]:

    if not ranked[column].between(
        0,
        1,
    ).all():

        raise ValueError(
            f"{column} contains values outside [0,1]."
        )

print("Probability range validation: PASS")


# ============================================================
# SEGMENT CUTPOINTS
# ============================================================

top1_cutoff = max(
    1,
    int(np.ceil(n * 0.01)),
)

top5_cutoff = max(
    top1_cutoff,
    int(np.ceil(n * 0.05)),
)

top10_cutoff = max(
    top5_cutoff,
    int(np.ceil(n * 0.10)),
)

top20_cutoff = max(
    top10_cutoff,
    int(np.ceil(n * 0.20)),
)


# ============================================================
# SEGMENT ASSIGNMENT
# ============================================================

ranked["segment"] = np.select(
    [
        ranked["model_rank"] <= top1_cutoff,

        ranked["model_rank"] <= top5_cutoff,

        ranked["model_rank"] <= top10_cutoff,

        ranked["model_rank"] <= top20_cutoff,
    ],
    [
        "Top 1%",
        "Top 1-5%",
        "Top 5-10%",
        "Top 10-20%",
    ],
    default="Bottom 80%",
)


ranked["segment"] = pd.Categorical(
    ranked["segment"],
    categories=SEGMENT_ORDER,
    ordered=True,
)


# ============================================================
# TRAINING DISTRIBUTION
# ============================================================

train_mean = train[
    FEATURES
].mean()

train_std = (
    train[
        FEATURES
    ]
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

        "purchase_rate": group[
            TARGET
        ].mean(),

        "mean_model_score": group[
            "raw_model_probability"
        ].mean(),

        "mean_calibrated_probability": group[
            "calibrated_probability"
        ].mean(),
    }


    for feature in FEATURES:

        segment_mean = group[
            feature
        ].mean()

        row[
            f"{feature}_mean"
        ] = segment_mean

        row[
            f"{feature}_std_diff"
        ] = (
            segment_mean
            - train_mean[feature]
        ) / train_std[feature]


    segment_summary_rows.append(
        row
    )


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

        segment_mean = group[
            feature
        ].mean()

        standardized_difference = (
            segment_mean
            - train_mean[feature]
        ) / train_std[feature]


        driver_rows.append(
            {
                "segment": segment,

                "feature": feature,

                "segment_mean": segment_mean,

                "train_mean": train_mean[
                    feature
                ],

                "train_std": train_std[
                    feature
                ],

                "standardized_difference":
                    standardized_difference,

                "absolute_standardized_difference":
                    abs(
                        standardized_difference
                    ),
            }
        )


driver_table = pd.DataFrame(
    driver_rows
)


# ============================================================
# TOP 1% DRIVER PROFILE
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
# ALL SEGMENT DRIVERS
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
        ]
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# ============================================================
# BUSINESS METRICS
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
# POSITIVE / NEGATIVE SHIFTS
# ============================================================

positive_shift_rows = []
negative_shift_rows = []


for segment in SEGMENT_ORDER:

    segment_drivers = driver_table[
        driver_table["segment"] == segment
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

                "feature": row[
                    "feature"
                ],

                "standardized_difference":
                    row[
                        "standardized_difference"
                    ],
            }
        )


    for _, row in negatives.iterrows():

        negative_shift_rows.append(
            {
                "segment": segment,

                "feature": row[
                    "feature"
                ],

                "standardized_difference":
                    row[
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
print("VALIDATION")
print("=" * 70)

print(
    f"Validation rows: {n}"
)

print(
    f"Explanation rows: {len(explanations)}"
)

print(
    f"Top 1% cutoff: rank <= {top1_cutoff}"
)

print(
    f"Top 5% cutoff: rank <= {top5_cutoff}"
)

print(
    f"Top 10% cutoff: rank <= {top10_cutoff}"
)

print(
    f"Top 20% cutoff: rank <= {top20_cutoff}"
)


print("\nSegment counts:")

print(
    ranked[
        "segment"
    ]
    .value_counts()
    .reindex(SEGMENT_ORDER)
    .fillna(0)
    .astype(int)
    .to_string()
)


print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(segment_summary_path)
print(driver_table_path)
print(positive_shift_path)
print(negative_shift_path)


print(
    "\nSegment driver analysis complete."
)

print(
    "\nNOTE: standardized segment differences are "
    "descriptive associations, not causal effects."
)