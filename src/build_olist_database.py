from pathlib import Path
import duckdb

RAW = Path("data/raw")
DB = Path("data/olist.duckdb")

def main():
    con = duckdb.connect(str(DB))

    print("=" * 70)
    print("BUILDING OLIST DUCKDB DATABASE")
    print("=" * 70)

    tables = {
        "customers": "olist_customers_dataset.csv",
        "orders": "olist_orders_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "order_payments": "olist_order_payments_dataset.csv",
        "order_reviews": "olist_order_reviews_dataset.csv",
        "products": "olist_products_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "geolocation": "olist_geolocation_dataset.csv",
        "category_translation": "product_category_name_translation.csv",
    }

    for table, filename in tables.items():
        path = (RAW / filename).resolve()

        print(f"\nCreating table: {table}")

        con.execute(
            f"""
            CREATE OR REPLACE TABLE {table} AS
            SELECT *
            FROM read_csv_auto('{path.as_posix()}')
            """
        )

        count = con.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]

        print(f"Rows: {count:,}")

    print("\n" + "=" * 70)
    print("DATABASE CREATED")
    print(f"Location: {DB}")
    print("=" * 70)

    con.close()


if __name__ == "__main__":
    main()