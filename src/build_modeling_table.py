from pathlib import Path
import duckdb

DB = Path("data/olist.duckdb")
SQL_FILE = Path("sql/04_modeling_table.sql")


def main():

    con = duckdb.connect(str(DB))

    print("=" * 70)
    print("BUILDING MODELING TABLE")
    print("=" * 70)

    sql = SQL_FILE.read_text(encoding="utf-8")

    con.execute(sql)

    # ------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------

    print("\nBasic validation:")

    result = con.execute("""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT customer_unique_id)
                AS customers,
            COUNT(DISTINCT snapshot_date)
                AS snapshots
        FROM modeling_table
    """).fetchone()

    print(f"Rows: {result[0]:,}")
    print(f"Unique customers: {result[1]:,}")
    print(f"Snapshots: {result[2]}")

    # ------------------------------------------------------------
    # Snapshot distribution
    # ------------------------------------------------------------

    print("\nRows by snapshot:")

    snapshots = con.execute("""
        SELECT
            snapshot_date,
            COUNT(*) AS rows,
            SUM(future_purchase_flag)
                AS future_buyers,
            ROUND(
                100.0 * AVG(future_purchase_flag),
                3
            ) AS purchase_rate_pct
        FROM modeling_table
        GROUP BY snapshot_date
        ORDER BY snapshot_date
    """).fetchdf()

    print(snapshots.to_string(index=False))

    # ------------------------------------------------------------
    # Missing-value check
    # ------------------------------------------------------------

    print("\nImportant feature null counts:")

    nulls = con.execute("""
        SELECT

            SUM(
                CASE
                    WHEN recency_days IS NULL
                    THEN 1 ELSE 0
                END
            ) AS recency_nulls,

            SUM(
                CASE
                    WHEN frequency IS NULL
                    THEN 1 ELSE 0
                END
            ) AS frequency_nulls,

            SUM(
                CASE
                    WHEN monetary IS NULL
                    THEN 1 ELSE 0
                END
            ) AS monetary_nulls,

            SUM(
                CASE
                    WHEN historical_avg_review_score IS NULL
                    THEN 1 ELSE 0
                END
            ) AS review_nulls,

            SUM(
                CASE
                    WHEN historical_avg_delivery_days IS NULL
                    THEN 1 ELSE 0
                END
            ) AS delivery_nulls

        FROM modeling_table
    """).fetchone()

    print(f"Recency nulls: {nulls[0]:,}")
    print(f"Frequency nulls: {nulls[1]:,}")
    print(f"Monetary nulls: {nulls[2]:,}")
    print(f"Review nulls: {nulls[3]:,}")
    print(f"Delivery nulls: {nulls[4]:,}")

    # ------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------

    print("\nSample rows:")

    sample = con.execute("""
        SELECT *
        FROM modeling_table
        ORDER BY snapshot_date, customer_unique_id
        LIMIT 10
    """).fetchdf()

    print(sample.to_string(index=False))

    con.close()


if __name__ == "__main__":
    main()