import duckdb

con = duckdb.connect("data/olist.duckdb")

query = """
WITH snapshots AS (
    SELECT DISTINCT
        snapshot_date,
        snapshot_date + INTERVAL '120 days' AS horizon_date,
        snapshot_date + INTERVAL '121 days' AS exclusive_end
    FROM customer_snapshot_features
),

current_definition AS (
    SELECT
        s.snapshot_date,
        COUNT(DISTINCT o.order_id) AS purchases_current
    FROM snapshots s
    JOIN order_level_features o
      ON o.order_purchase_timestamp > s.snapshot_date
     AND o.order_purchase_timestamp <= s.horizon_date
    GROUP BY s.snapshot_date
),

full_day_definition AS (
    SELECT
        s.snapshot_date,
        COUNT(DISTINCT o.order_id) AS purchases_full_day
    FROM snapshots s
    JOIN order_level_features o
      ON o.order_purchase_timestamp > s.snapshot_date
     AND o.order_purchase_timestamp < s.exclusive_end
    GROUP BY s.snapshot_date
)

SELECT
    s.snapshot_date,
    COALESCE(c.purchases_current, 0) AS purchases_current,
    COALESCE(f.purchases_full_day, 0) AS purchases_full_day,
    COALESCE(f.purchases_full_day, 0)
      - COALESCE(c.purchases_current, 0) AS boundary_purchases
FROM snapshots s
LEFT JOIN current_definition c
    ON s.snapshot_date = c.snapshot_date
LEFT JOIN full_day_definition f
    ON s.snapshot_date = f.snapshot_date
ORDER BY s.snapshot_date;
"""

result = con.execute(query).fetchdf()
print(result.to_string(index=False))

con.close()