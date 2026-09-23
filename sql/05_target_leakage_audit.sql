-- ============================================================
-- 05_target_leakage_audit.sql
--
-- Check whether future orders purchased inside the 120-day
-- prediction window were delivered after the prediction window.
-- ============================================================

WITH snapshot_windows AS (

    SELECT
        snapshot_date,
        snapshot_date + INTERVAL '120 days' AS horizon_end
    FROM customer_snapshot_features
    GROUP BY snapshot_date

),

future_orders AS (

    SELECT
        s.snapshot_date,
        s.horizon_end,
        o.order_id,
        o.customer_id,
        o.order_purchase_timestamp,
        o.order_delivered_customer_date

    FROM snapshot_windows s

    INNER JOIN order_level_features o
        ON o.order_purchase_timestamp > s.snapshot_date
        AND o.order_purchase_timestamp <= s.horizon_end

),

summary AS (

    SELECT

        snapshot_date,

        COUNT(*) AS future_orders,

        SUM(
            CASE
                WHEN order_delivered_customer_date IS NOT NULL
                THEN 1
                ELSE 0
            END
        ) AS eventually_delivered_orders,

        SUM(
            CASE
                WHEN order_delivered_customer_date IS NOT NULL
                 AND order_delivered_customer_date > horizon_end
                THEN 1
                ELSE 0
            END
        ) AS delivered_after_horizon

    FROM future_orders

    GROUP BY snapshot_date

)

SELECT
    snapshot_date,
    future_orders,
    eventually_delivered_orders,
    delivered_after_horizon,

    ROUND(
        100.0 * delivered_after_horizon
        / NULLIF(eventually_delivered_orders, 0),
        3
    ) AS pct_delivered_after_horizon

FROM summary
ORDER BY snapshot_date;