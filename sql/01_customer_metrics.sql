-- Adapt table/column names to your selected dataset.
-- These are portfolio SQL templates; do not claim results until executed.

-- 1. Monthly revenue
SELECT
    DATE_TRUNC('month', order_date) AS month,
    SUM(order_value) AS revenue
FROM orders
GROUP BY 1
ORDER BY 1;

-- 2. Orders and revenue per customer
SELECT
    customer_id,
    COUNT(DISTINCT order_id) AS orders,
    SUM(order_value) AS revenue,
    AVG(order_value) AS average_order_value
FROM orders
GROUP BY customer_id;

-- 3. Repeat customers
SELECT
    customer_id,
    COUNT(DISTINCT order_id) AS order_count
FROM orders
GROUP BY customer_id
HAVING COUNT(DISTINCT order_id) >= 2;

-- 4. Customer return rate
SELECT
    customer_id,
    AVG(CASE WHEN returned_flag = 1 THEN 1.0 ELSE 0.0 END) AS return_rate
FROM orders
GROUP BY customer_id;

-- 5. Monthly active customers
SELECT
    DATE_TRUNC('month', order_date) AS month,
    COUNT(DISTINCT customer_id) AS active_customers
FROM orders
GROUP BY 1
ORDER BY 1;
