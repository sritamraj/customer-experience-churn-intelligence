-- Template only.
-- Churn must be defined with an observation cutoff and FUTURE outcome window.
-- Do not use future outcomes as predictive features.

-- Example descriptive churn aggregation after you have created a valid churn label:
SELECT
    customer_segment,
    AVG(churn) AS churn_rate,
    COUNT(*) AS customers
FROM customer_model_table
GROUP BY customer_segment
ORDER BY churn_rate DESC;
