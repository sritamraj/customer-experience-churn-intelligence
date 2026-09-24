-- ============================================================
-- 03_future_targets.sql
--
-- Leakage-safe future customer behavior targets.
--
-- PRIMARY TARGET:
--
-- future_purchase_flag = 1 when the customer makes at least
-- one purchase after the snapshot date during the next
-- 120 calendar days.
--
-- Purchase timing defines the primary propensity target.
-- Delivery timing is NOT required for future_purchase_flag.
--
-- FUTURE OUTCOME METRICS:
--
-- future_delivered_order_count, future_revenue, and
-- future_item_revenue remain restricted to orders that were
-- delivered within the same 120-day horizon.
--
-- This keeps the primary target aligned with purchase propensity
-- while preserving delivery-qualified downstream outcomes.
-- ============================================================

CREATE OR REPLACE TABLE customer_future_targets AS

WITH snapshot_windows AS (

    SELECT DISTINCT

        snapshot_date,

        -- Exclusive upper boundary.
        -- This includes the complete 120th calendar day.
        snapshot_date + INTERVAL '121 days'
            AS exclusive_end

    FROM customer_snapshot_features

),

-- ------------------------------------------------------------
-- PURCHASE-BASED TARGET
--
-- A purchase counts when:
--
-- 1. Purchase occurs after the snapshot.
-- 2. Purchase occurs before the exclusive end of the
--    120-calendar-day prediction window.
--
-- Delivery is intentionally NOT required.
-- ------------------------------------------------------------

future_purchases AS (

    SELECT

        s.snapshot_date,
        o.customer_unique_id,

        COUNT(DISTINCT o.order_id)
            AS future_purchase_order_count

    FROM snapshot_windows s

    INNER JOIN order_level_features o

        ON o.order_purchase_timestamp > s.snapshot_date

        AND o.order_purchase_timestamp < s.exclusive_end

    GROUP BY

        s.snapshot_date,
        o.customer_unique_id

),

-- ------------------------------------------------------------
-- DELIVERY-QUALIFIED FUTURE OUTCOMES
--
-- These remain restricted to orders delivered within the
-- same 120-day horizon.
-- ------------------------------------------------------------

future_delivered_orders AS (

    SELECT

        s.snapshot_date,
        o.customer_unique_id,

        COUNT(DISTINCT o.order_id)
            AS future_delivered_order_count,

        SUM(o.payment_value)
            AS future_revenue,

        SUM(o.item_revenue)
            AS future_item_revenue

    FROM snapshot_windows s

    INNER JOIN order_level_features o

        ON o.order_purchase_timestamp > s.snapshot_date

        AND o.order_purchase_timestamp < s.exclusive_end

        AND o.order_delivered_customer_date IS NOT NULL

        AND o.order_delivered_customer_date < s.exclusive_end

    GROUP BY

        s.snapshot_date,
        o.customer_unique_id

)

SELECT

    s.snapshot_date,
    s.customer_unique_id,

    -- PRIMARY PURCHASE-PROPENSITY TARGET
    CASE

        WHEN p.future_purchase_order_count > 0
        THEN 1

        ELSE 0

    END AS future_purchase_flag,

    -- DELIVERY-QUALIFIED FUTURE OUTCOMES
    COALESCE(
        d.future_delivered_order_count,
        0
    ) AS future_delivered_order_count,

    COALESCE(
        d.future_revenue,
        0
    ) AS future_revenue,

    COALESCE(
        d.future_item_revenue,
        0
    ) AS future_item_revenue

FROM customer_snapshot_features s

LEFT JOIN future_purchases p

    ON s.snapshot_date = p.snapshot_date

    AND s.customer_unique_id = p.customer_unique_id

LEFT JOIN future_delivered_orders d

    ON s.snapshot_date = d.snapshot_date

    AND s.customer_unique_id = d.customer_unique_id;