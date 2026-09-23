-- ============================================================
-- 02_customer_snapshot_features.sql
--
-- Customer-level historical features at fixed snapshot dates.
--
-- IMPORTANT:
-- Only information available ON OR BEFORE the snapshot date
-- is allowed into the feature table.
--
-- Recent-window features are also leakage-safe:
--   - purchase timestamp must be before snapshot
--   - order must be delivered
--   - delivery date must be before snapshot
-- ============================================================

CREATE OR REPLACE TABLE customer_snapshot_features AS

WITH snapshot_dates AS (
    SELECT DATE '2017-09-01' AS snapshot_date
    UNION ALL
    SELECT DATE '2017-12-01'
    UNION ALL
    SELECT DATE '2018-03-01'
    UNION ALL
    SELECT DATE '2018-06-19'
),

customer_orders AS (
    SELECT
        customer_unique_id,
        order_id,
        order_purchase_timestamp,
        order_status,
        item_count,
        product_count,
        seller_count,
        item_revenue,
        freight_value,
        avg_item_price,
        payment_value,
        payment_count,
        payment_type_count,
        max_installments,
        review_score,
        review_count,
        delivery_days,
        delivery_delay_days,
        order_delivered_customer_date
    FROM order_level_features
    WHERE customer_unique_id IS NOT NULL
),

snapshots AS (
    SELECT
        s.snapshot_date,
        c.customer_unique_id
    FROM snapshot_dates s
    CROSS JOIN (
        SELECT DISTINCT customer_unique_id
        FROM customer_orders
    ) c
),

historical AS (
    SELECT
        s.snapshot_date,
        s.customer_unique_id,

        -- ====================================================
        -- ALL HISTORICAL FEATURES
        -- ====================================================

        COUNT(o.order_id) AS historical_order_count,

        SUM(
            CASE
                WHEN o.order_status = 'delivered'
                THEN COALESCE(o.item_revenue, 0)
                ELSE 0
            END
        ) AS historical_item_revenue,

        SUM(
            CASE
                WHEN o.order_status = 'delivered'
                THEN COALESCE(o.payment_value, 0)
                ELSE 0
            END
        ) AS historical_payment_value,

        AVG(
            CASE
                WHEN o.order_status = 'delivered'
                THEN o.payment_value
                ELSE NULL
            END
        ) AS historical_avg_order_value,

        AVG(
            CASE
                WHEN o.order_status = 'delivered'
                THEN o.review_score
                ELSE NULL
            END
        ) AS historical_avg_review_score,

        AVG(
            CASE
                WHEN o.order_status = 'delivered'
                THEN o.delivery_days
                ELSE NULL
            END
        ) AS historical_avg_delivery_days,

        AVG(
            CASE
                WHEN o.order_status = 'delivered'
                THEN o.delivery_delay_days
                ELSE NULL
            END
        ) AS historical_avg_delivery_delay,

        SUM(
            CASE
                WHEN o.order_status = 'delivered'
                THEN COALESCE(o.freight_value, 0)
                ELSE 0
            END
        ) AS historical_freight_value,

        SUM(
            CASE
                WHEN o.order_status = 'delivered'
                THEN COALESCE(o.item_count, 0)
                ELSE 0
            END
        ) AS historical_item_count,

        SUM(
            CASE
                WHEN o.order_status = 'delivered'
                THEN COALESCE(o.product_count, 0)
                ELSE 0
            END
        ) AS historical_product_count,

        SUM(
            CASE
                WHEN o.order_status = 'delivered'
                THEN COALESCE(o.seller_count, 0)
                ELSE 0
            END
        ) AS historical_seller_count,

        COUNT(
            CASE
                WHEN o.order_status = 'delivered'
                THEN 1
                ELSE NULL
            END
        ) AS delivered_order_count,

        COUNT(
            CASE
                WHEN o.order_status <> 'delivered'
                THEN 1
                ELSE NULL
            END
        ) AS non_delivered_order_count,

        MAX(
            CASE
                WHEN o.order_status = 'delivered'
                THEN o.order_purchase_timestamp
                ELSE NULL
            END
        ) AS last_delivered_order_timestamp,

        MIN(
            CASE
                WHEN o.order_status = 'delivered'
                THEN o.order_purchase_timestamp
                ELSE NULL
            END
        ) AS first_delivered_order_timestamp,

        -- ====================================================
        -- RECENT ORDER FEATURES
        -- ====================================================

        COUNT(
            CASE
                WHEN
                    o.order_status = 'delivered'
                    AND o.order_delivered_customer_date IS NOT NULL
                    AND CAST(o.order_delivered_customer_date AS DATE) < s.snapshot_date
                    AND o.order_purchase_timestamp >= s.snapshot_date - INTERVAL '30 days'
                    AND o.order_purchase_timestamp < s.snapshot_date
                THEN 1
                ELSE NULL
            END
        ) AS orders_last_30d,

        COUNT(
            CASE
                WHEN
                    o.order_status = 'delivered'
                    AND o.order_delivered_customer_date IS NOT NULL
                    AND CAST(o.order_delivered_customer_date AS DATE) < s.snapshot_date
                    AND o.order_purchase_timestamp >= s.snapshot_date - INTERVAL '60 days'
                    AND o.order_purchase_timestamp < s.snapshot_date
                THEN 1
                ELSE NULL
            END
        ) AS orders_last_60d,

        COUNT(
            CASE
                WHEN
                    o.order_status = 'delivered'
                    AND o.order_delivered_customer_date IS NOT NULL
                    AND CAST(o.order_delivered_customer_date AS DATE) < s.snapshot_date
                    AND o.order_purchase_timestamp >= s.snapshot_date - INTERVAL '90 days'
                    AND o.order_purchase_timestamp < s.snapshot_date
                THEN 1
                ELSE NULL
            END
        ) AS orders_last_90d,

        -- ====================================================
        -- RECENT SPEND FEATURES
        -- ====================================================

        SUM(
            CASE
                WHEN
                    o.order_status = 'delivered'
                    AND o.order_delivered_customer_date IS NOT NULL
                    AND CAST(o.order_delivered_customer_date AS DATE) < s.snapshot_date
                    AND o.order_purchase_timestamp >= s.snapshot_date - INTERVAL '30 days'
                    AND o.order_purchase_timestamp < s.snapshot_date
                THEN COALESCE(o.payment_value, 0)
                ELSE 0
            END
        ) AS spend_last_30d,

        SUM(
            CASE
                WHEN
                    o.order_status = 'delivered'
                    AND o.order_delivered_customer_date IS NOT NULL
                    AND CAST(o.order_delivered_customer_date AS DATE) < s.snapshot_date
                    AND o.order_purchase_timestamp >= s.snapshot_date - INTERVAL '60 days'
                    AND o.order_purchase_timestamp < s.snapshot_date
                THEN COALESCE(o.payment_value, 0)
                ELSE 0
            END
        ) AS spend_last_60d,

        SUM(
            CASE
                WHEN
                    o.order_status = 'delivered'
                    AND o.order_delivered_customer_date IS NOT NULL
                    AND CAST(o.order_delivered_customer_date AS DATE) < s.snapshot_date
                    AND o.order_purchase_timestamp >= s.snapshot_date - INTERVAL '90 days'
                    AND o.order_purchase_timestamp < s.snapshot_date
                THEN COALESCE(o.payment_value, 0)
                ELSE 0
            END
        ) AS spend_last_90d

    FROM snapshots s

    LEFT JOIN customer_orders o
        ON s.customer_unique_id = o.customer_unique_id

        AND CAST(o.order_purchase_timestamp AS DATE) < s.snapshot_date

        AND (
            o.order_status <> 'delivered'
            OR
            (
                o.order_delivered_customer_date IS NOT NULL
                AND CAST(o.order_delivered_customer_date AS DATE) < s.snapshot_date
            )
        )

    GROUP BY
        s.snapshot_date,
        s.customer_unique_id
)

SELECT
    snapshot_date,
    customer_unique_id,

    historical_order_count,
    delivered_order_count,
    non_delivered_order_count,

    historical_item_revenue,
    historical_payment_value,
    historical_avg_order_value,
    historical_avg_review_score,
    historical_avg_delivery_days,
    historical_avg_delivery_delay,
    historical_freight_value,
    historical_item_count,
    historical_product_count,
    historical_seller_count,

    last_delivered_order_timestamp,
    first_delivered_order_timestamp,

    DATE_DIFF(
        'day',
        CAST(last_delivered_order_timestamp AS DATE),
        snapshot_date
    ) AS recency_days,

    DATE_DIFF(
        'day',
        CAST(first_delivered_order_timestamp AS DATE),
        snapshot_date
    ) AS customer_age_days,

    -- ========================================================
    -- RECENT BEHAVIOR FEATURES
    -- ========================================================

    orders_last_30d,
    orders_last_60d,
    orders_last_90d,

    spend_last_30d,
    spend_last_60d,
    spend_last_90d

FROM historical

WHERE delivered_order_count > 0;