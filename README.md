# Customer Experience Purchase Propensity Intelligence

A leakage-safe, temporally validated customer purchase-propensity ranking system designed to estimate which customers are more likely to make a purchase within the next 120 days.

The project combines **SQL analytics, statistical feature engineering, predictive modeling, probability calibration, customer ranking, error analysis, and production-style inference validation** to support customer prioritization and data-driven business decisions under severe class imbalance.

---

## 1. Business Problem

Given a historical customer snapshot, estimate each customer's probability of making at least one purchase during the following 120 days.

The resulting propensity scores can be used to rank customers for potential prioritization in:

* customer engagement
* retention workflows
* marketing campaigns
* customer-experience analysis

The system uses only information available at the prediction snapshot.

The objective is not simply to maximize classification accuracy. The goal is to produce a leakage-safe propensity model that can:

* estimate purchase probability
* rank customers by relative propensity
* support capacity-constrained targeting
* provide interpretable analytical signals
* support evaluation under severe class imbalance

---

## 2. End-to-End Analytical Pipeline

```text
Historical Customer / Order Data
            |
            v
      SQL Data Analysis
            |
            v
 Customer-Level Feature Engineering
            |
            v
 Leakage Checks + Temporal Snapshots
            |
            v
 Temporal Train / Validation / Test
            |
            v
 Baseline + Candidate Model Comparison
            |
            v
 Model Selection
            |
            v
 Probability Calibration
            |
            v
 Frozen Final Evaluation
            |
            v
 Customer Propensity Ranking
            |
            v
 Lift / Capture / Capacity Analysis
            |
            v
 Error Analysis + Explainability
            |
            v
 Temporal Drift Diagnostics
            |
            v
 Production Artifact Validation
```

---

## 3. Target Definition

The prediction target is:

```text
future_purchase_flag = 1
```

when a customer makes at least one purchase during the following 120-day horizon.

The target is constructed from future customer behavior, while model features are restricted to information available at the historical prediction snapshot.

This separation prevents future information from entering the feature set and creating target leakage.

---

## 4. SQL Analytics

SQL is used as part of the analytical workflow to transform transactional information into customer-level analytical features and modeling data.

The analysis covers:

* customer purchase frequency
* monetary value
* recency
* order-level aggregation
* product and seller behavior
* delivery and customer-experience variables
* customer-level historical statistics
* temporal snapshot construction
* future-purchase target construction
* modeling-table validation
* segment-level analysis
* missing-value investigation
* leakage-prone field checks

The SQL layer connects raw transactional data with the downstream statistical and machine-learning workflow.

### SQL techniques used in the analytical workflow

* `SELECT`
* `WHERE`
* `JOIN`
* `GROUP BY`
* `HAVING`
* `CASE`
* `COUNT`
* `COUNT(DISTINCT ...)`
* `SUM`
* `AVG`
* `NULL` handling
* conditional aggregation
* date/time operations
* subqueries
* CTE-based transformations
* customer-level aggregation

---

## 5. Customer-Level Feature Engineering

The final model uses 16 customer-level features.

### Behavioral Features

* `recency_days`
* `customer_age_days`
* `frequency`
* `monetary`

### Customer-Experience Features

* `historical_avg_order_value`
* `historical_avg_review_score`
* `historical_avg_delivery_days`
* `historical_avg_delivery_delay`
* `historical_freight_value`

### Historical Order / Product Features

* `historical_item_count`
* `historical_product_count`
* `historical_seller_count`
* `non_delivered_order_count`

### Average Behavioral Features

* `avg_freight_per_order`
* `avg_items_per_order`
* `avg_sellers_per_order`

These features describe historical customer behavior without directly exposing future purchase outcomes.

---

## 6. Leakage Prevention and Temporal Validation

A random train/test split would not accurately represent the intended prediction setting because the model is intended to predict future customer behavior from earlier information.

The project therefore uses chronological customer snapshots.

```text
Earlier Historical Data
        |
        v
Training Snapshot
2017-09-01
        |
        v
Training Snapshot
2017-12-01
        |
        v
Validation Snapshot
2018-03-01
        |
        v
Final Frozen Test
2018-06-19
```

The final test snapshot is later than the training and validation periods.

The final test period is not used for:

* model fitting
* model selection
* calibration

### Development limitation

Only one genuine temporal development fold is available for model selection because of the available snapshot structure.

Therefore, the project does not claim extensive rolling-window validation.

---

## 7. Model Development and Selection

Several candidate approaches were evaluated during development.

| Model               | Log Loss | ROC-AUC |  PR-AUC |
| ------------------- | -------: | ------: | ------: |
| Frequency Baseline  |  0.05046 | 0.50000 | 0.00879 |
| Logistic Regression |  0.04997 | 0.57051 | 0.01776 |
| Random Forest       |  0.11044 | 0.57312 | 0.01459 |
| XGBoost             |  0.05101 | 0.59506 | 0.01599 |

The canonical model is **Logistic Regression** with:

* 16 standardized features
* median imputation
* `StandardScaler`
* `class_weight="balanced"`
* `C=1.0`
* `max_iter=2000`
* `random_state=42`

### Why Logistic Regression?

XGBoost achieved a higher development ROC-AUC, but model selection was not based on ROC-AUC alone.

The documented selection criteria also consider:

* log loss
* PR-AUC
* severe class imbalance
* probability quality
* interpretability
* calibration
* ranking usefulness
* business decision requirements

Logistic Regression was selected based on the documented development criteria, particularly its log loss and PR-AUC, together with its interpretability and suitability for probability-based analysis.

This makes the model-selection decision explainable rather than simply choosing the algorithm with the highest single metric.

---

## 8. Probability Calibration

The canonical Logistic Regression model is followed by **sigmoid probability calibration**.

Calibration is important because the model output is intended to represent a customer purchase propensity score rather than only a binary class label.

The validation-period calibrator is fitted during development and frozen before final evaluation.

Validation calibrated log loss:

```text
0.045555
```

The final test period is not used to fit or recalibrate the model.

---

## 9. Final Frozen Test Results

The final frozen test contains:

```text
Customers: 76,857
Buyers:       314
Buyer Rate: 0.4086%
```

### Final Metrics

| Metric                       |   Result |
| ---------------------------- | -------: |
| Calibrated Log Loss          | 0.027060 |
| ROC-AUC                      | 0.600515 |
| PR-AUC                       | 0.011995 |
| Accuracy (threshold = 0.50)  |  99.586% |
| Precision (threshold = 0.50) |    25.0% |
| Recall (threshold = 0.50)    |   0.637% |
| F1 (threshold = 0.50)        |  0.01242 |

The severe class imbalance makes raw accuracy an insufficient primary metric.

A model can achieve high accuracy by predicting most customers as non-buyers while failing to identify the rare positive cases.

Therefore, the project emphasizes:

* log loss
* PR-AUC
* ROC-AUC
* calibration
* ranking performance
* lift
* capture

---

## 10. Customer Ranking and Campaign Capacity

The primary business use case is ranking customers by predicted purchase propensity.

### Frozen-Test Ranking Results

| Targeted Population | Customers Targeted | Buyers Found | Buyer Rate |  Lift | Capture |
| ------------------- | -----------------: | -----------: | ---------: | ----: | ------: |
| Top 1%              |                769 |           24 |     3.121% | 7.64x |   7.64% |
| Top 5%              |              3,843 |           41 |     1.067% | 2.61x |  13.06% |
| Top 10%             |              7,686 |           63 |     0.820% | 2.01x |  20.06% |
| Top 20%             |             15,372 |          103 |     0.670% | 1.64x |  32.80% |

For example, the top 1% ranked segment has a 3.121% observed buyer rate compared with a 0.4086% overall buyer rate.

This corresponds to approximately:

```text
7.64x lift
```

The ranking analysis is intended to support capacity-constrained customer prioritization. It does not establish that contacting a customer causes that customer to purchase.

### Capacity-Aware Targeting

Different campaign capacities can produce different operating points.

For example:

```text
Top 1%
769 customers
24 buyers
7.64% capture
7.64x lift
```

versus:

```text
Top 10%
7,686 customers
63 buyers
20.06% capture
2.01x lift
```

The appropriate operating point should depend on business capacity, cost, customer experience, and measured incremental impact.

---

## 11. Business Recommendations

The model should be treated as a prioritization signal rather than an automatic customer-action decision.

### 1. Use Ranking for Capacity-Constrained Targeting

Ranking customers by propensity is more aligned with campaign-capacity decisions than relying only on a fixed 0.50 classification threshold.

### 2. Monitor Ranking Quality

Relevant monitoring metrics include:

* log loss
* PR-AUC
* calibration
* lift
* capture
* buyer prevalence

### 3. Monitor Distribution Shift

Feature distribution changes can affect model reliability. PSI and temporal distribution comparisons can help identify potential stability issues.

### 4. Do Not Interpret Predictive Associations as Causal Effects

A high-propensity customer is not necessarily a customer who will purchase because of a campaign.

To measure incremental business impact, controlled treatment/control experimentation would be required.

---

## 12. Error Analysis and Explainability

At the 0.50 classification threshold, the final test error breakdown is:

```text
True Positives:    2
False Positives:   6
False Negatives: 312
True Negatives: 76537
```

This illustrates why the project focuses on ranking rather than relying only on binary classification at a fixed threshold.

### Standardized Logistic Regression Coefficients

| Feature                      | Standardized Coefficient |
| ---------------------------- | -----------------------: |
| `historical_avg_order_value` |                   -0.443 |
| `historical_seller_count`    |                   -0.407 |
| `recency_days`               |                   -0.402 |
| `frequency`                  |                   +0.370 |
| `monetary`                   |                   +0.304 |
| `avg_sellers_per_order`      |                   +0.260 |
| `customer_age_days`          |                   +0.257 |
| `historical_item_count`      |                   +0.243 |
| `avg_items_per_order`        |                   -0.237 |

For the standardized Logistic Regression model, a coefficient represents the change in log-odds associated with a one-standard-deviation increase in the feature, holding other model inputs constant.

These coefficients represent **predictive associations, not causal effects**.

---

## 13. Drift and Monitoring

The project evaluates feature distribution changes across temporal snapshots.

The strongest observed PSI values include approximately:

```text
recency_days       ~= 0.518
customer_age_days  ~= 0.523
```

These indicate substantial distributional change according to the project's configured monitoring thresholds.

### Buyer-Rate Change

| Snapshot   | Buyer Rate |
| ---------- | ---------: |
| 2017-09-01 |     1.019% |
| 2017-12-01 |     0.879% |
| 2018-03-01 |     0.796% |
| 2018-06-19 |     0.409% |

The observed distributional changes are relevant for model monitoring.

The project does not claim that feature drift alone caused the decline in buyer prevalence.

### Production Monitoring Areas

A future production implementation should monitor:

**Data Quality**

* schema changes
* missing values
* feature ranges
* unexpected values
* row counts
* duplicate records

**Model Quality**

* log loss
* PR-AUC
* ROC-AUC
* calibration
* lift
* capture

**Business Outcomes**

* purchase rate
* campaign response
* customer engagement
* campaign capacity
* incremental impact

**Distribution Shift**

* PSI
* feature distributions
* target prevalence
* temporal cohort behavior

---

## 14. Production-Style Artifacts and Reproducibility

The repository contains model artifacts including:

```text
data/model/artifacts/logistic_model.joblib
data/model/artifacts/sigmoid_calibrator.joblib
data/model/artifacts/model_metadata.json
```

The artifacts are validated before inference.

Validation checks include:

* model version
* feature count
* feature names
* pipeline structure
* coefficient dimensions
* calibrator availability
* input schema
* leakage-prone feature names
* probability bounds
* batch prediction
* deterministic inference

Current validation status:

```text
STEP 24 PASS
```

The final model artifact is frozen after evaluation.

### Reproducibility

The project records direct dependency versions including:

```text
Python        3.14.7
scikit-learn  1.9.1
pandas        3.0.6
numpy         2.5.3
scipy         1.18.1
duckdb        1.5.5
joblib        1.6.0
xgboost       3.4.1
```

The workflow separates:

```text
Data Preparation
      |
Model Development
      |
Model Selection
      |
Calibration
      |
Artifact Locking
      |
Frozen Evaluation
      |
Error Analysis
      |
Explainability
      |
Drift Diagnostics
      |
Production Validation
```

The project includes production-style artifact and inference validation but does **not** claim a live production deployment.

---

## 15. Interview-Relevant Design Decisions

### Why Purchase Propensity Instead of Ordinary Classification?

The intended business action is customer prioritization. Ranking customers by relative purchase propensity is more useful than producing only a yes/no prediction.

### Why Not Use a Random Train/Test Split?

The prediction setting is temporal. Random splitting can allow future-period patterns to influence training and produce an overly optimistic evaluation.

### How Was Leakage Prevented?

Features are constructed from information available at the prediction snapshot, while future purchase behavior is reserved for target construction.

### Why Is Accuracy Misleading?

Only 0.4086% of final-test customers purchase within the target horizon. High accuracy can therefore coexist with poor identification of buyers.

### Why Use PR-AUC?

PR-AUC is useful for evaluating rare positive classes because it focuses on precision-recall behavior under severe class imbalance.

### Why Use Log Loss?

The system produces probability estimates, so probability quality matters in addition to ranking.

### Why Use Logistic Regression?

It provides an interpretable model while supporting probability estimation and coefficient-level analysis.

### Why Not Automatically Select XGBoost?

XGBoost achieved higher development ROC-AUC, but model selection also considered log loss, PR-AUC, calibration, interpretability, and the business objective.

### Why Calibrate Probabilities?

A propensity system benefits from probability estimates that are better calibrated for downstream ranking and analysis.

### Why Freeze Calibration?

Using the final test period to fit or recalibrate the model would contaminate the final evaluation.

### What Does 7.64x Lift Mean?

The top 1% ranked segment has an observed purchase rate approximately 7.64 times the overall test-set buyer rate.

### Is the Model Causal?

No. The model estimates predictive propensity and does not establish that a marketing intervention causes a purchase.

### What Is the Largest Methodological Limitation?

Only one genuine temporal development fold is available for model selection.

### What Would Be Improved With Additional Data?

Potential improvements include:

* additional temporal snapshots
* rolling-window validation
* stronger calibration evaluation
* calibrated tree-based models
* automated drift monitoring
* campaign-capacity optimization
* controlled incremental-impact experiments
* more recent or live data

---

## 16. Key Limitations

The project explicitly documents the following limitations:

1. **Limited temporal development data** - only one genuine temporal development fold is available for model selection.
2. **Severe class imbalance** - the final buyer rate is 0.4086%.
3. **Historical final test** - the final test is a frozen historical evaluation period rather than a live production stream.
4. **No live deployment** - the project contains production-style validation but does not claim live production deployment.
5. **No causal inference** - the model estimates predictive propensity rather than incremental treatment effect.
6. **Temporal distribution shift** - feature distributions and buyer prevalence change across temporal snapshots.

These limitations are retained explicitly so that the results are interpreted within their proper scope.

---

## 17. Repository Structure and Project Status

```text
customer-experience-purchase-propensity-intelligence/
|
├── audits/
|
├── data/
│   └── model/
│       └── artifacts/
|
├── docs/
│   └── ARCHITECTURE.md
|
├── models/
|
├── notebooks/
|
├── sql/
|
├── src/
|
├── tests/
|
├── .gitignore
├── README.md
└── requirements.txt
```

The repository separates analytical code, SQL workflows, model artifacts, documentation, tests, and notebooks.

### Project Status

The project includes:

* SQL-based analytical workflow
* customer-level feature engineering
* temporal validation
* leakage prevention
* baseline comparison
* Logistic Regression modeling
* Random Forest comparison
* XGBoost comparison
* probability calibration
* frozen final evaluation
* propensity ranking
* lift and capture analysis
* campaign-capacity analysis
* error analysis
* coefficient-based explainability
* temporal drift diagnostics
* production artifact validation
* reproducibility checks
* automated tests
* documented limitations

The project is intended as an end-to-end **Data Science / Customer Analytics / Predictive Modeling** portfolio project demonstrating the workflow from analytical data preparation through model evaluation and business-oriented decision support.
