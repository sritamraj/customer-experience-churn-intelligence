from pathlib import Path
import duckdb

DB = Path("data/olist.duckdb")
SQL_DIR = Path("sql")

def main():
    con = duckdb.connect(str(DB))

    sql_file = SQL_DIR / "01_order_level_features.sql"

    print("=" * 70)
    print("RUNNING ORDER-LEVEL SQL")
    print("=" * 70)

    sql = sql_file.read_text(encoding="utf-8")

    con.execute(sql)

    result = con.execute("""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT order_id) AS unique_orders,
            COUNT(DISTINCT customer_unique_id) AS unique_customers
        FROM order_level_features
    """).fetchone()

    print(f"Rows: {result[0]:,}")
    print(f"Unique orders: {result[1]:,}")
    print(f"Unique customers: {result[2]:,}")

    print("\nSample:")
    print(
        con.execute("""
            SELECT *
            FROM order_level_features
            LIMIT 5
        """).fetchdf().to_string(index=False)
    )

    con.close()

if __name__ == "__main__":
    main()