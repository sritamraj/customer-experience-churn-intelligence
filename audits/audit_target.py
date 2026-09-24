import duckdb

con = duckdb.connect("data/olist.duckdb")

query = """
WITH snapshots AS (
    SELECT DISTINCT
        snapshot_date,
        snapshot_date + INTERVAL '120 days' AS horizon_end
    FROM customer_snapshot_features
),

purchase_counts AS (
    SELECT
        s.snapshot_date,
        o.customer_unique_id,
        COUNT(DISTINCT o.order_id) AS purchases
    FROM snapshots s
    JOIN order_level_features o
      ON o.order_purchase_timestamp > s.snapshot_date
     AND o.order_purchase_timestamp <= s.horizon_end
    GROUP BY
        s.snapshot_date,
        o.customer_unique_id
)

SELECT
    s.snapshot_date,
    COUNT(*) AS customers,
    COUNT(p.customer_unique_id) AS purchase_within_120d,
    SUM(
        CASE
            WHEN p.purchases > 0 THEN 1
            ELSE 0
        END
    ) AS buyers
FROM customer_snapshot_features s
LEFT JOIN purchase_counts p
  ON s.snapshot_date = p.snapshot_date
 AND s.customer_unique_id = p.customer_unique_id
GROUP BY s.snapshot_date
ORDER BY s.snapshot_date;
"""

result = con.execute(query).fetchdf()
print(result.to_string(index=False))

con.close()