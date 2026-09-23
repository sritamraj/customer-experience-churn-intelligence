-- Adapt names to your dataset.

-- Delivery delay rate
SELECT
    customer_id,
    AVG(CASE WHEN delivery_delay_days > 0 THEN 1.0 ELSE 0.0 END)
        AS late_delivery_rate,
    AVG(delivery_delay_days) AS average_delivery_delay
FROM orders
GROUP BY customer_id;

-- Cancellation rate
SELECT
    customer_id,
    AVG(CASE WHEN cancelled_flag = 1 THEN 1.0 ELSE 0.0 END)
        AS cancellation_rate
FROM orders
GROUP BY customer_id;
