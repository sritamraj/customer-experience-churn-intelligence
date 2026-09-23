from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

FILES = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]


def main():
    print("=" * 70)
    print("OLIST DATA QUALITY AUDIT")
    print("=" * 70)

    total_rows = 0

    for filename in FILES:
        path = DATA_DIR / filename

        if not path.exists():
            print(f"\n❌ MISSING: {filename}")
            continue

        df = pd.read_csv(path)

        print(f"\n{'=' * 70}")
        print(f"FILE: {filename}")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        print("\nColumns:")
        print(list(df.columns))

        print("\nMissing values:")
        missing = df.isna().sum()
        missing = missing[missing > 0]

        if len(missing):
            print(missing.sort_values(ascending=False).to_string())
        else:
            print("None")

        print(f"\nDuplicate rows: {df.duplicated().sum():,}")

        total_rows += len(df)

    print("\n" + "=" * 70)
    print(f"TOTAL ROWS ACROSS TABLES: {total_rows:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()