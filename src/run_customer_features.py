from pathlib import Path
import duckdb

DB = Path("data/olist.duckdb")
SQL_FILE = Path("sql/02_customer_snapshot_features.sql")


def main():
    con = duckdb.connect(str(DB))

    print("=" * 70)
    print("BUILDING CUSTOMER SNAPSHOT FEATURES")
    print("=" * 70)

    sql = SQL_FILE.read_text(encoding="utf-8")

    con.execute(sql)

    print("\nSnapshot counts:")

    result = con.execute("""
        SELECT
            snapshot_date,
            COUNT(*) AS customers,
            COUNT(DISTINCT customer_unique_id) AS unique_customers
        FROM customer_snapshot_features
        GROUP BY snapshot_date
        ORDER BY snapshot_date
    """).fetchdf()

    print(result.to_string(index=False))

    print("\nFeature sample:")

    sample = con.execute("""
        SELECT *
        FROM customer_snapshot_features
        ORDER BY snapshot_date, customer_unique_id
        LIMIT 10
    """).fetchdf()

    print(sample.to_string(index=False))

    print("\nFeature table validation:")

    validation = con.execute("""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT customer_unique_id) AS customers,
            COUNT(DISTINCT snapshot_date) AS snapshots
        FROM customer_snapshot_features
    """).fetchone()

    print(f"Rows: {validation[0]:,}")
    print(f"Customers: {validation[1]:,}")
    print(f"Snapshots: {validation[2]:,}")

    con.close()


if __name__ == "__main__":
    main()