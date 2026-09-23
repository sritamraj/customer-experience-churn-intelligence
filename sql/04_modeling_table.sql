-- ============================================================
-- 04_modeling_table.sql
--
-- Combine leakage-safe historical features with future targets.
--
-- IMPORTANT:
-- Everything in the feature section comes from information
-- available before the snapshot date.
--
-- Everything in the target section comes AFTER the snapshot.
-- ============================================================

CREATE OR REPLACE TABLE modeling_table AS

SELECT

    -- --------------------------------------------------------
    -- Identifiers / time
    -- --------------------------------------------------------

    f.snapshot_date,

    f.customer_unique_id,

    -- --------------------------------------------------------
    -- RFM / customer history
    -- --------------------------------------------------------

    f.recency_days,

    f.customer_age_days,

    f.delivered_order_count
        AS frequency,

    f.historical_payment_value
        AS monetary,

    f.historical_avg_order_value,

    -- --------------------------------------------------------
    -- Historical experience
    -- --------------------------------------------------------

    f.historical_avg_review_score,

    f.historical_avg_delivery_days,

    f.historical_avg_delivery_delay,

    f.historical_freight_value,

    -- --------------------------------------------------------
    -- Historical behavior
    -- --------------------------------------------------------

    f.historical_item_count,

    f.historical_product_count,

    f.historical_seller_count,

    f.non_delivered_order_count,

    -- --------------------------------------------------------
    -- Recent behavior
    -- --------------------------------------------------------

    f.orders_last_30d,

    f.orders_last_60d,

    f.orders_last_90d,

    f.spend_last_30d,

    f.spend_last_60d,

    f.spend_last_90d,

    -- --------------------------------------------------------
    -- Derived ratios
    -- --------------------------------------------------------

    CASE
        WHEN f.delivered_order_count > 0
        THEN
            f.historical_freight_value
            / f.delivered_order_count
        ELSE 0
    END AS avg_freight_per_order,

    CASE
        WHEN f.delivered_order_count > 0
        THEN
            f.historical_item_count
            / f.delivered_order_count
        ELSE 0
    END AS avg_items_per_order,

    CASE
        WHEN f.delivered_order_count > 0
        THEN
            f.historical_seller_count
            / f.delivered_order_count
        ELSE 0
    END AS avg_sellers_per_order,

    -- --------------------------------------------------------
    -- TARGETS
    -- --------------------------------------------------------

    t.future_purchase_flag,

    t.future_delivered_order_count,

    t.future_revenue,

    t.future_item_revenue

FROM customer_snapshot_features f

INNER JOIN customer_future_targets t

    ON f.snapshot_date = t.snapshot_date

    AND f.customer_unique_id = t.customer_unique_id;