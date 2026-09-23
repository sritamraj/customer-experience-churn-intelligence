from pathlib import Path

import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

DB_PATH = Path("data/olist.duckdb")
OUTPUT_DIR = Path("data/model")

TARGET = "future_purchase_flag"

FEATURES = [
    # Core historical behavior
    "recency_days",
    "customer_age_days",
    "frequency",
    "monetary",

    # Historical customer experience
    "historical_avg_order_value",
    "historical_avg_review_score",
    "historical_avg_delivery_days",
    "historical_avg_delivery_delay",

    # Historical economics / basket behavior
    "historical_freight_value",
    "historical_item_count",
    "historical_product_count",
    "historical_seller_count",

    # Historical operational behavior
    "non_delivered_order_count",

    # Normalized historical behavior
    "avg_freight_per_order",
    "avg_items_per_order",
    "avg_sellers_per_order",

    # Recent purchase behavior
    "orders_last_30d",
    "orders_last_60d",
    "orders_last_90d",

    # Recent spending behavior
    "spend_last_30d",
    "spend_last_60d",
    "spend_last_90d",
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PREPARING TEMPORAL MODEL DATA")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH))

    # --------------------------------------------------------
    # Load modeling table
    # --------------------------------------------------------

    df = con.execute(
        "SELECT * FROM modeling_table"
    ).fetchdf()

    con.close()

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "snapshot_date",
        "customer_unique_id",
        TARGET,
        *FEATURES,
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "\nMissing required columns from modeling_table:\n"
            + "\n".join(f"  - {col}" for col in missing_columns)
        )

    # --------------------------------------------------------
    # Basic dataset checks
    # --------------------------------------------------------

    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

    df = df.sort_values(
        ["snapshot_date", "customer_unique_id"]
    ).reset_index(drop=True)

    print("\nDataset sizes:")

    # --------------------------------------------------------
    # Temporal split
    #
    # Train:
    #   2017-09-01
    #   2017-12-01
    #
    # Validation:
    #   2018-03-01
    #
    # Test:
    #   2018-06-19
    # --------------------------------------------------------

    train_dates = [
        pd.Timestamp("2017-09-01"),
        pd.Timestamp("2017-12-01"),
    ]

    validation_date = pd.Timestamp("2018-03-01")
    test_date = pd.Timestamp("2018-06-19")

    train_df = df[
        df["snapshot_date"].isin(train_dates)
    ].copy()

    validation_df = df[
        df["snapshot_date"] == validation_date
    ].copy()

    test_df = df[
        df["snapshot_date"] == test_date
    ].copy()

    # --------------------------------------------------------
    # Validate temporal ordering
    # --------------------------------------------------------

    if train_df["snapshot_date"].max() >= validation_date:
        raise ValueError("Temporal leakage: train overlaps validation.")

    if validation_date >= test_date:
        raise ValueError("Temporal ordering is invalid.")

    if validation_df["snapshot_date"].max() >= test_df["snapshot_date"].min():
        raise ValueError("Temporal leakage: validation overlaps test.")

    # --------------------------------------------------------
    # Dataset sizes
    # --------------------------------------------------------

    print(f"Train      : {len(train_df):,}")
    print(f"Validation : {len(validation_df):,}")
    print(f"Test       : {len(test_df):,}")

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print("\nTarget distribution:")

    datasets = {
        "TRAIN": train_df,
        "VALIDATION": validation_df,
        "TEST": test_df,
    }

    for name, data in datasets.items():

        rows = len(data)
        buyers = int(data[TARGET].sum())
        purchase_rate = (
            buyers / rows * 100
            if rows > 0
            else 0
        )

        print(
            f"{name:<12}"
            f"rows={rows:>7,} "
            f"buyers={buyers:>5,} "
            f"purchase_rate={purchase_rate:.3f}%"
        )

    # --------------------------------------------------------
    # Feature validation
    # --------------------------------------------------------

    print("\nFeature count:")
    print(f"Total model features: {len(FEATURES)}")

    print("\nModel features:")

    for i, feature in enumerate(FEATURES, start=1):
        print(f"{i:>2}. {feature}")

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing values in model features:")

    missing_counts = df[FEATURES].isna().sum()

    for feature in FEATURES:
        print(
            f"{feature:<36}"
            f"{missing_counts[feature]:>8,}"
        )

    # --------------------------------------------------------
    # Ensure numeric features
    # --------------------------------------------------------

    non_numeric = [
        feature
        for feature in FEATURES
        if not pd.api.types.is_numeric_dtype(df[feature])
    ]

    if non_numeric:
        raise TypeError(
            "\nNon-numeric model features detected:\n"
            + "\n".join(f"  - {col}" for col in non_numeric)
        )

    # --------------------------------------------------------
    # Save only required columns
    #
    # Keep snapshot/customer IDs for traceability.
    # --------------------------------------------------------

    output_columns = [
        "snapshot_date",
        "customer_unique_id",
        *FEATURES,
        TARGET,
        "future_delivered_order_count",
        "future_revenue",
        "future_item_revenue",
    ]

    train_output = train_df[output_columns].copy()
    validation_output = validation_df[output_columns].copy()
    test_output = test_df[output_columns].copy()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train_output.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    validation_output.to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False
    )

    test_output.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    print("\nSaved:")
    print("data/model/train.csv")
    print("data/model/validation.csv")
    print("data/model/test.csv")

    # --------------------------------------------------------
    # Final schema check
    # --------------------------------------------------------

    print("\nSaved feature counts:")

    for name, data in {
        "train.csv": train_output,
        "validation.csv": validation_output,
        "test.csv": test_output,
    }.items():

        actual_features = [
            col
            for col in FEATURES
            if col in data.columns
        ]

        print(
            f"{name:<18}"
            f"{len(actual_features)}/{len(FEATURES)} features"
        )

    print("\nTemporal split preparation complete.")


if __name__ == "__main__":
    main()