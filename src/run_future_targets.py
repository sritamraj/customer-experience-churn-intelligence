from pathlib import Path
import duckdb

DB = Path("data/olist.duckdb")
SQL_FILE = Path("sql/03_future_targets.sql")


def main():
    con = duckdb.connect(str(DB))

    print("=" * 70)
    print("BUILDING FUTURE CUSTOMER TARGETS")
    print("=" * 70)

    # ------------------------------------------------------------
    # Build / rebuild future targets
    # ------------------------------------------------------------
    sql = SQL_FILE.read_text(encoding="utf-8")
    con.execute(sql)

    # ------------------------------------------------------------
    # Target distribution by snapshot
    # ------------------------------------------------------------
    print("\nTarget distribution by snapshot:")

    result = con.execute("""
        SELECT
            snapshot_date,

            COUNT(*) AS customers,

            SUM(future_purchase_flag)
                AS future_buyers,

            ROUND(
                100.0 * AVG(future_purchase_flag),
                3
            ) AS future_purchase_rate_pct,

            ROUND(
                AVG(future_delivered_order_count),
                3
            ) AS avg_future_orders,

            ROUND(
                AVG(future_revenue),
                2
            ) AS avg_future_revenue,

            ROUND(
                APPROX_QUANTILE(future_revenue, 0.50),
                2
            ) AS median_future_revenue,

            ROUND(
                APPROX_QUANTILE(future_revenue, 0.90),
                2
            ) AS p90_future_revenue,

            ROUND(
                APPROX_QUANTILE(future_revenue, 0.99),
                2
            ) AS p99_future_revenue

        FROM customer_future_targets

        GROUP BY snapshot_date

        ORDER BY snapshot_date
    """).fetchdf()

    print(result.to_string(index=False))

    # ------------------------------------------------------------
    # Overall target distribution
    # ------------------------------------------------------------
    print("\nOverall target distribution:")

    overall = con.execute("""
        SELECT

            COUNT(*) AS rows,

            SUM(future_purchase_flag)
                AS future_buyers,

            ROUND(
                100.0 * AVG(future_purchase_flag),
                3
            ) AS future_purchase_rate_pct,

            ROUND(
                AVG(future_delivered_order_count),
                3
            ) AS avg_future_orders,

            ROUND(
                AVG(future_revenue),
                2
            ) AS avg_future_revenue,

            ROUND(
                APPROX_QUANTILE(future_revenue, 0.50),
                2
            ) AS median_future_revenue,

            ROUND(
                APPROX_QUANTILE(future_revenue, 0.90),
                2
            ) AS p90_future_revenue,

            ROUND(
                APPROX_QUANTILE(future_revenue, 0.99),
                2
            ) AS p99_future_revenue

        FROM customer_future_targets
    """).fetchone()

    labels = [
        "Rows",
        "Future buyers",
        "Future purchase rate %",
        "Average future orders",
        "Median future revenue",
        "P90 future revenue",
        "P99 future revenue",
    ]

    # Print first values manually so labels stay aligned.
    print(f"Rows: {overall[0]:,}")
    print(f"Future buyers: {overall[1]:,}")
    print(f"Future purchase rate %: {overall[2]}")
    print(f"Average future orders: {overall[3]}")
    print(f"Average future revenue: {overall[4]}")
    print(f"Median future revenue: {overall[5]}")
    print(f"P90 future revenue: {overall[6]}")
    print(f"P99 future revenue: {overall[7]}")

    # ------------------------------------------------------------
    # Zero-revenue analysis
    # ------------------------------------------------------------
    print("\nZero-revenue analysis:")

    zero = con.execute("""
        SELECT

            COUNT(*) AS total_rows,

            SUM(
                CASE
                    WHEN future_revenue = 0
                    THEN 1
                    ELSE 0
                END
            ) AS zero_revenue_rows,

            ROUND(
                100.0 * AVG(
                    CASE
                        WHEN future_revenue = 0
                        THEN 1
                        ELSE 0
                    END
                ),
                3
            ) AS zero_revenue_pct

        FROM customer_future_targets
    """).fetchone()

    print(f"Total rows: {zero[0]:,}")
    print(f"Zero revenue rows: {zero[1]:,}")
    print(f"Zero revenue %: {zero[2]}")

    # ------------------------------------------------------------
    # Target validation
    # ------------------------------------------------------------
    print("\nTarget validation:")

    validation = con.execute("""
        SELECT

            COUNT(*) AS rows,

            COUNT(DISTINCT customer_unique_id)
                AS unique_customers,

            COUNT(DISTINCT snapshot_date)
                AS snapshots,

            MIN(snapshot_date)
                AS first_snapshot,

            MAX(snapshot_date)
                AS last_snapshot,

            MIN(future_revenue)
                AS min_future_revenue,

            MAX(future_revenue)
                AS max_future_revenue,

            SUM(
                CASE
                    WHEN future_purchase_flag = 1
                         AND future_revenue <= 0
                    THEN 1
                    ELSE 0
                END
            ) AS flag_revenue_mismatch

        FROM customer_future_targets
    """).fetchone()

    print(f"Rows: {validation[0]:,}")
    print(f"Unique customers: {validation[1]:,}")
    print(f"Snapshots: {validation[2]}")
    print(f"First snapshot: {validation[3]}")
    print(f"Last snapshot: {validation[4]}")
    print(f"Minimum future revenue: {validation[5]}")
    print(f"Maximum future revenue: {validation[6]}")
    print(f"Purchase/revenue mismatches: {validation[7]}")

    con.close()


if __name__ == "__main__":
    main()