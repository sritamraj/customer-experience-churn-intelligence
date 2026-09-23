-- ============================================================
-- 01_order_level_features.sql
-- Build exactly one analytical row per order.
-- ============================================================

CREATE OR REPLACE TABLE order_item_summary AS
SELECT
    order_id,
    COUNT(*) AS item_count,
    COUNT(DISTINCT product_id) AS product_count,
    COUNT(DISTINCT seller_id) AS seller_count,
    SUM(price) AS item_revenue,
    SUM(freight_value) AS freight_value,
    AVG(price) AS avg_item_price
FROM order_items
GROUP BY order_id;


CREATE OR REPLACE TABLE order_payment_summary AS
SELECT
    order_id,
    SUM(payment_value) AS payment_value,
    COUNT(*) AS payment_count,
    COUNT(DISTINCT payment_type) AS payment_type_count,
    MAX(payment_installments) AS max_installments
FROM order_payments
GROUP BY order_id;


CREATE OR REPLACE TABLE order_review_summary AS
SELECT
    order_id,
    AVG(review_score) AS review_score,
    COUNT(*) AS review_count
FROM order_reviews
GROUP BY order_id;


CREATE OR REPLACE TABLE order_level_features AS
SELECT
    o.order_id,

    c.customer_unique_id,

    o.customer_id,

    o.order_status,

    CAST(o.order_purchase_timestamp AS TIMESTAMP)
        AS order_purchase_timestamp,

    CAST(o.order_approved_at AS TIMESTAMP)
        AS order_approved_at,

    CAST(o.order_delivered_carrier_date AS TIMESTAMP)
        AS order_delivered_carrier_date,

    CAST(o.order_delivered_customer_date AS TIMESTAMP)
        AS order_delivered_customer_date,

    CAST(o.order_estimated_delivery_date AS TIMESTAMP)
        AS order_estimated_delivery_date,

    i.item_count,

    i.product_count,

    i.seller_count,

    i.item_revenue,

    i.freight_value,

    i.avg_item_price,

    p.payment_value,

    p.payment_count,

    p.payment_type_count,

    p.max_installments,

    r.review_score,

    r.review_count,

    CASE
        WHEN o.order_delivered_customer_date IS NOT NULL
        THEN
            DATE_DIFF(
                'day',
                CAST(o.order_purchase_timestamp AS DATE),
                CAST(o.order_delivered_customer_date AS DATE)
            )
        ELSE NULL
    END AS delivery_days,

    CASE
        WHEN o.order_delivered_customer_date IS NOT NULL
        THEN
            DATE_DIFF(
                'day',
                CAST(o.order_estimated_delivery_date AS DATE),
                CAST(o.order_delivered_customer_date AS DATE)
            )
        ELSE NULL
    END AS delivery_delay_days

FROM orders o

LEFT JOIN customers c
    ON o.customer_id = c.customer_id

LEFT JOIN order_item_summary i
    ON o.order_id = i.order_id

LEFT JOIN order_payment_summary p
    ON o.order_id = p.order_id

LEFT JOIN order_review_summary r
    ON o.order_id = r.order_id;