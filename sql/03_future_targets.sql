-- ============================================================
-- 03_future_targets.sql
--
-- Leakage-safe future customer behavior targets.
--
-- TARGET DEFINITION:
--
-- A future purchase is counted only when:
--
-- 1. The order is purchased AFTER the snapshot date.
-- 2. The order is purchased within the next 120 days.
-- 3. The order is delivered.
-- 4. The delivery is completed ON OR BEFORE the 120-day
--    prediction horizon.
--
-- This prevents target construction from using delivery
-- information that occurs after the prediction window.
-- ============================================================

CREATE OR REPLACE TABLE customer_future_targets AS

WITH snapshot_windows AS (

    SELECT DISTINCT

        snapshot_date,

        snapshot_date + INTERVAL '120 days'
            AS horizon_end

    FROM customer_snapshot_features

),

future_orders AS (

    SELECT

        s.snapshot_date,
        s.horizon_end,

        o.order_id,
        o.customer_unique_id,

        o.order_purchase_timestamp,
        o.order_delivered_customer_date,

        o.payment_value,
        o.item_revenue

    FROM snapshot_windows s

    INNER JOIN order_level_features o

        ON o.order_purchase_timestamp > s.snapshot_date

        AND o.order_purchase_timestamp <= s.horizon_end

        AND o.order_delivered_customer_date IS NOT NULL

        AND o.order_delivered_customer_date <= s.horizon_end

),

customer_targets AS (

    SELECT

        snapshot_date,
        customer_unique_id,

        COUNT(DISTINCT order_id)
            AS future_delivered_order_count,

        SUM(payment_value)
            AS future_revenue,

        SUM(item_revenue)
            AS future_item_revenue

    FROM future_orders

    GROUP BY

        snapshot_date,
        customer_unique_id

)

SELECT

    s.snapshot_date,
    s.customer_unique_id,

    CASE

        WHEN c.future_delivered_order_count > 0
        THEN 1

        ELSE 0

    END AS future_purchase_flag,

    COALESCE(
        c.future_delivered_order_count,
        0
    ) AS future_delivered_order_count,

    COALESCE(
        c.future_revenue,
        0
    ) AS future_revenue,

    COALESCE(
        c.future_item_revenue,
        0
    ) AS future_item_revenue

FROM customer_snapshot_features s

LEFT JOIN customer_targets c

    ON s.snapshot_date = c.snapshot_date

    AND s.customer_unique_id = c.customer_unique_id;