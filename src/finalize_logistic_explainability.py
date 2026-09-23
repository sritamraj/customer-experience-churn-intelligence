from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "data" / "model"

GLOBAL_IMPORTANCE_PATH = (
    MODEL_DIR / "logistic_global_feature_importance.csv"
)

LOCAL_IMPORTANCE_PATH = (
    MODEL_DIR / "logistic_local_contribution_importance.csv"
)

SEGMENT_SHIFT_PATH = (
    MODEL_DIR / "logistic_segment_feature_shifts.csv"
)

SEGMENT_SUMMARY_PATH = (
    MODEL_DIR / "logistic_segment_driver_summary.csv"
)

OUTPUT_DIR = MODEL_DIR


# ============================================================
# LOCKED FEATURES
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
# LOAD
# ============================================================

print("=" * 70)
print("FINAL LOGISTIC EXPLAINABILITY AUDIT")
print("=" * 70)

global_importance = pd.read_csv(
    GLOBAL_IMPORTANCE_PATH
)

local_importance = pd.read_csv(
    LOCAL_IMPORTANCE_PATH
)

segment_shifts = pd.read_csv(
    SEGMENT_SHIFT_PATH
)

segment_summary = pd.read_csv(
    SEGMENT_SUMMARY_PATH
)


# ============================================================
# VALIDATION
# ============================================================

required_global = {
    "feature",
    "coefficient",
    "odds_ratio",
    "absolute_coefficient",
    "importance_rank",
}

required_local = {
    "feature",
    "mean_absolute_contribution",
    "mean_contribution",
    "importance_rank",
}

required_segment = {
    "segment",
    "feature",
    "standardized_difference",
    "absolute_standardized_difference",
}

if not required_global.issubset(
    global_importance.columns
):
    raise ValueError(
        "Global importance file missing required columns."
    )

if not required_local.issubset(
    local_importance.columns
):
    raise ValueError(
        "Local importance file missing required columns."
    )

if not required_segment.issubset(
    segment_shifts.columns
):
    raise ValueError(
        "Segment shift file missing required columns."
    )


# ============================================================
# GLOBAL + LOCAL MERGE
# ============================================================

audit = global_importance[
    [
        "feature",
        "coefficient",
        "odds_ratio",
        "absolute_coefficient",
        "direction",
        "importance_rank",
    ]
].copy()

audit = audit.rename(
    columns={
        "importance_rank":
            "global_importance_rank"
    }
)

audit = audit.merge(
    local_importance[
        [
            "feature",
            "mean_absolute_contribution",
            "mean_contribution",
            "importance_rank",
        ]
    ].rename(
        columns={
            "importance_rank":
                "local_importance_rank"
        }
    ),
    on="feature",
    how="left",
)


# ============================================================
# TOP 1% SEGMENT SHIFTS
# ============================================================

top1_shift = segment_shifts[
    segment_shifts["segment"]
    == "Top 1%"
].copy()

top1_shift = top1_shift[
    [
        "feature",
        "segment_mean",
        "train_mean",
        "standardized_difference",
        "absolute_standardized_difference",
    ]
]

top1_shift = top1_shift.rename(
    columns={
        "segment_mean":
            "top1_segment_mean",
        "train_mean":
            "training_mean",
        "standardized_difference":
            "top1_standardized_shift",
        "absolute_standardized_difference":
            "top1_absolute_shift",
    }
)


audit = audit.merge(
    top1_shift,
    on="feature",
    how="left",
)


# ============================================================
# RANK CORRELATIONS
# ============================================================

global_rank = audit[
    "global_importance_rank"
]

local_rank = audit[
    "local_importance_rank"
]

top1_rank = (
    audit[
        "top1_absolute_shift"
    ]
    .rank(
        ascending=False,
        method="first",
    )
)


global_local_spearman = (
    global_rank
    .corr(
        local_rank,
        method="spearman",
    )
)

global_top1_spearman = (
    global_rank
    .corr(
        top1_rank,
        method="spearman",
    )
)

local_top1_spearman = (
    local_rank
    .corr(
        top1_rank,
        method="spearman",
    )
)


# ============================================================
# RANK DIFFERENCES
# ============================================================

audit["global_vs_local_rank_gap"] = (
    audit["global_importance_rank"]
    - audit["local_importance_rank"]
).abs()

audit["global_vs_top1_rank_gap"] = (
    audit["global_importance_rank"]
    - top1_rank
).abs()


# ============================================================
# TOP DRIVER FLAGS
# ============================================================

audit["global_top5"] = (
    audit["global_importance_rank"]
    <= 5
)

audit["local_top5"] = (
    audit["local_importance_rank"]
    <= 5
)

audit["top1_shift_top5"] = (
    top1_rank <= 5
)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("GLOBAL vs LOCAL vs TOP-1%")
print("=" * 70)

display_columns = [
    "feature",
    "coefficient",
    "odds_ratio",
    "global_importance_rank",
    "mean_absolute_contribution",
    "local_importance_rank",
    "top1_standardized_shift",
    "top1_absolute_shift",
]

print(
    audit.sort_values(
        "global_importance_rank"
    )[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# CORRELATIONS
# ============================================================

print("\n" + "=" * 70)
print("RANK AGREEMENT")
print("=" * 70)

print(
    f"Global vs Local Spearman: "
    f"{global_local_spearman:.4f}"
)

print(
    f"Global vs Top-1% Spearman: "
    f"{global_top1_spearman:.4f}"
)

print(
    f"Local vs Top-1% Spearman: "
    f"{local_top1_spearman:.4f}"
)


# ============================================================
# TOP 5 COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("TOP 5 — GLOBAL MODEL")
print("=" * 70)

print(
    audit.sort_values(
        "global_importance_rank"
    )[
        [
            "feature",
            "global_importance_rank",
        ]
    ]
    .head(5)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("TOP 5 — LOCAL CONTRIBUTION")
print("=" * 70)

print(
    audit.sort_values(
        "local_importance_rank"
    )[
        [
            "feature",
            "local_importance_rank",
        ]
    ]
    .head(5)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("TOP 5 — TOP-1% SEGMENT SHIFT")
print("=" * 70)

print(
    audit.sort_values(
        "top1_absolute_shift",
        ascending=False,
    )[
        [
            "feature",
            "top1_absolute_shift",
        ]
    ]
    .head(5)
    .to_string(
        index=False
    )
)


# ============================================================
# BIGGEST DISAGREEMENTS
# ============================================================

print("\n" + "=" * 70)
print("BIGGEST GLOBAL vs LOCAL DISAGREEMENTS")
print("=" * 70)

disagreements = (
    audit.sort_values(
        "global_vs_local_rank_gap",
        ascending=False,
    )
    [
        [
            "feature",
            "global_importance_rank",
            "local_importance_rank",
            "global_vs_local_rank_gap",
        ]
    ]
)

print(
    disagreements
    .head(5)
    .to_string(
        index=False
    )
)


# ============================================================
# BUSINESS SEGMENT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION SEGMENT OUTCOMES")
print("=" * 70)

print(
    segment_summary[
        [
            "segment",
            "customers",
            "buyers",
            "purchase_rate",
            "mean_model_score",
            "mean_calibrated_probability",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# FINAL AUDIT FLAGS
# ============================================================

print("\n" + "=" * 70)
print("AUDIT FLAGS")
print("=" * 70)

print(
    "Global/local rank correlation: "
    + (
        "STRONG"
        if global_local_spearman >= 0.70
        else "MODERATE"
        if global_local_spearman >= 0.40
        else "WEAK"
    )
)

print(
    "Global/Top-1% rank correlation: "
    + (
        "STRONG"
        if global_top1_spearman >= 0.70
        else "MODERATE"
        if global_top1_spearman >= 0.40
        else "WEAK"
    )
)

print(
    "Local/Top-1% rank correlation: "
    + (
        "STRONG"
        if local_top1_spearman >= 0.70
        else "MODERATE"
        if local_top1_spearman >= 0.40
        else "WEAK"
    )
)


# ============================================================
# SAVE FINAL AUDIT
# ============================================================

OUTPUT_PATH = (
    OUTPUT_DIR
    / "logistic_final_explainability_audit.csv"
)

audit.to_csv(
    OUTPUT_PATH,
    index=False,
)


print("\n" + "=" * 70)
print("FILE SAVED")
print("=" * 70)

print(OUTPUT_PATH)

print("\nFinal explainability audit complete.")