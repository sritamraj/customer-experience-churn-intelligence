import duckdb

con = duckdb.connect("data/olist.duckdb")

query = """
WITH snapshots AS (
    SELECT DISTINCT
        snapshot_date,
        snapshot_date + INTERVAL '120 days' AS horizon_start,
        snapshot_date + INTERVAL '121 days' AS exclusive_end
    FROM customer_snapshot_features
),

purchase_only AS (
    SELECT
        s.snapshot_date,
        o.customer_unique_id
    FROM snapshots s
    JOIN order_level_features o
      ON o.order_purchase_timestamp > s.snapshot_date
     AND o.order_purchase_timestamp < s.exclusive_end
    GROUP BY
        s.snapshot_date,
        o.customer_unique_id
),

current_target AS (
    SELECT
        snapshot_date,
        customer_unique_id,
        future_purchase_flag
    FROM customer_future_targets
)

SELECT
    s.snapshot_date,

    COUNT(*) AS customers,

    SUM(
        CASE
            WHEN p.customer_unique_id IS NOT NULL
            THEN 1 ELSE 0
        END
    ) AS corrected_positive,

    SUM(
        CASE
            WHEN c.future_purchase_flag = 1
            THEN 1 ELSE 0
        END
    ) AS current_positive,

    SUM(
        CASE
            WHEN p.customer_unique_id IS NOT NULL
             AND c.future_purchase_flag = 0
            THEN 1 ELSE 0
        END
    ) AS customers_becoming_positive,

    SUM(
        CASE
            WHEN p.customer_unique_id IS NULL
             AND c.future_purchase_flag = 1
            THEN 1 ELSE 0
        END
    ) AS customers_becoming_negative

FROM customer_snapshot_features s

LEFT JOIN purchase_only p
  ON s.snapshot_date = p.snapshot_date
 AND s.customer_unique_id = p.customer_unique_id

LEFT JOIN current_target c
  ON s.snapshot_date = c.snapshot_date
 AND s.customer_unique_id = c.customer_unique_id

GROUP BY s.snapshot_date
ORDER BY s.snapshot_date;
"""

result = con.execute(query).fetchdf()
print(result.to_string(index=False))

con.close()