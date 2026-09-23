"""Starter feature-engineering template.

IMPORTANT:
This is intentionally conservative. You MUST map the columns of your chosen
dataset and define the prediction cutoff/outcome window before using this for ML.
"""
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/data.csv")
OUTPUT_PATH = Path("data/processed/customer_features.csv")

# CHANGE THESE AFTER INSPECTING YOUR DATASET
CUSTOMER_COL = "customer_id"
ORDER_DATE_COL = "order_date"
ORDER_VALUE_COL = "order_value"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Put the dataset at data/raw/data.csv first.")

    df = pd.read_csv(DATA_PATH)

    required = [CUSTOMER_COL, ORDER_DATE_COL, ORDER_VALUE_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Update the column mapping at the top "
            "of this script for your dataset."
        )

    df[ORDER_DATE_COL] = pd.to_datetime(df[ORDER_DATE_COL], errors="coerce")
    df[ORDER_VALUE_COL] = pd.to_numeric(df[ORDER_VALUE_COL], errors="coerce")

    df = df.dropna(subset=[CUSTOMER_COL, ORDER_DATE_COL])
    df = df[df[ORDER_VALUE_COL].fillna(0) >= 0]

    # IMPORTANT: This is descriptive feature construction only.
    # For final modeling, build features using only data BEFORE cutoff T.
    customer = (
        df.groupby(CUSTOMER_COL)
        .agg(
            orders=(ORDER_DATE_COL, "count"),
            revenue=(ORDER_VALUE_COL, "sum"),
            average_order_value=(ORDER_VALUE_COL, "mean"),
            last_order_date=(ORDER_DATE_COL, "max"),
        )
        .reset_index()
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    customer.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(customer):,} customer records to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
