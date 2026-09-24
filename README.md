# Customer Experience Purchase Propensity Intelligence

A leakage-safe, temporally validated customer purchase-propensity ranking system designed to identify customers who are more likely to make a purchase within the next 120 days.

The project focuses on **customer prioritization under severe class imbalance**, rather than maximizing raw classification accuracy.

---

## 1. Business Problem

Given a historical customer snapshot, estimate each customer's probability of making at least one purchase during the following 120 days.

The resulting propensity scores can be used to rank customers for potential prioritization in retention, engagement, or marketing workflows.

The system uses only information available at the prediction snapshot.

### Primary target

```text
future_purchase_flag = 1
```

when the customer makes at least one purchase after the snapshot date and within the following 120-day horizon.

Purchase timing defines the primary propensity target. Delivery timing is not required for the primary purchase label.

---

## 2. Why Temporal Validation?

Random train/test splitting can allow observations from later periods to influence evaluation of earlier-like observations.

This project therefore uses chronological customer snapshots:

| Dataset    | Snapshot   | Purpose                             |
| ---------- | ---------- | ----------------------------------- |
| Train      | 2017-09-01 | Model development                   |
| Train      | 2017-12-01 | Model development                   |
| Validation | 2018-03-01 | Temporal validation and calibration |
| Test       | 2018-06-19 | Final frozen evaluation             |

The final test snapshot is later than every training and validation snapshot.

The final test set was not used for model fitting, model selection, or calibration.

---

## 3. Feature Contract

The final model uses exactly 16 customer-level features:

```text
recency_days
customer_age_days
frequency
monetary
historical_avg_order_value
historical_avg_review_score
historical_avg_delivery_days
historical_avg_delivery_delay
historical_freight_value
historical_item_count
historical_product_count
historical_seller_count
non_delivered_order_count
avg_freight_per_order
avg_items_per_order
avg_sellers_per_order
```

The feature pipeline performs:

```text
Median imputation
        ↓
StandardScaler
        ↓
Logistic Regression
```

Two features contain missing values in the modeling data:

* `historical_avg_review_score`
* `historical_avg_order_value`

Missing values are handled by the model pipeline's median imputer.

No future outcome fields are supplied to the model.

---

## 4. Development Model Comparison

The development stage compares several approaches on a chronological development fold:

```text
Train:      2017-09-01
Validation: 2017-12-01
```

Compared models:

* Frequency baseline
* Logistic Regression
* Random Forest
* XGBoost

Development results:

| Model               |    Log Loss |     ROC-AUC |      PR-AUC |
| ------------------- | ----------: | ----------: | ----------: |
| Frequency baseline  |     0.05046 |     0.50000 |     0.00879 |
| Logistic Regression | **0.04997** |     0.57051 | **0.01776** |
| Random Forest       |     0.11044 |     0.57312 |     0.01459 |
| XGBoost             |     0.05101 | **0.59506** |     0.01599 |

The canonical Logistic Regression model was retained based on the development evidence, particularly its development log loss and PR-AUC.

This is a development-fold result, not a claim that Logistic Regression is universally superior to the other models.

Only one genuine temporal development fold was available because the training data contains two training snapshots.

---

## 5. Canonical Model

Model version:

```text
logistic_16_sigmoid_v1
```

Pipeline:

```text
16 customer features
        ↓
Median Imputation
        ↓
StandardScaler
        ↓
LogisticRegression
        ↓
Sigmoid calibration
        ↓
Calibrated purchase propensity
```

Configuration:

```text
class_weight = balanced
C = 1.0
max_iter = 2000
random_state = 42
```

The final model and calibration artifacts are frozen.

---

## 6. Probability Calibration

The Logistic Regression model produces a raw probability/score.

A separate sigmoid calibration model was fitted during the validation stage and then frozen.

The frozen calibration artifact is applied during final scoring.

Validation calibrated log loss:

```text
0.045555
```

No recalibration is performed on the final test set.

---

## 7. Final Frozen Test Evaluation

Final test snapshot:

```text
2018-06-19
```

Test population:

```text
76,857 customers
314 buyers
0.4086% buyer rate
```

Final metrics:

| Metric                      |       Result |
| --------------------------- | -----------: |
| Calibrated Log Loss         | **0.027060** |
| ROC-AUC                     | **0.600515** |
| PR-AUC                      | **0.011995** |
| Accuracy at threshold 0.50  |      99.586% |
| Precision at threshold 0.50 |        25.0% |
| Recall at threshold 0.50    |       0.637% |
| F1                          |      0.01242 |

Because only 0.409% of test customers purchased, accuracy at a 0.50 threshold is not an informative primary business metric.

The project therefore emphasizes probability quality and customer ranking.

---

## 8. Ranking Performance

The final test ranking provides the following results:

| Customer segment | Customers | Buyers captured | Buyer rate |      Lift | Capture |
| ---------------- | --------: | --------------: | ---------: | --------: | ------: |
| Top 1%           |       769 |              24 |     3.121% | **7.64×** |   7.64% |
| Top 5%           |     3,843 |              41 |     1.067% | **2.61×** |  13.06% |
| Top 10%          |     7,686 |              63 |     0.820% | **2.01×** |  20.06% |
| Top 20%          |    15,372 |             103 |     0.670% | **1.64×** |  32.80% |

This makes the system more naturally suited to **prioritization/ranking** than binary classification at a default 0.50 threshold.

---

## 9. Error Analysis

The final frozen test evaluation includes:

* True-positive analysis
* True-negative analysis
* False-positive analysis
* False-negative analysis
* Segment-level diagnostics
* High-confidence false-negative inspection
* High-confidence false-positive inspection

At a 0.50 classification threshold:

```text
True positives:   2
False positives:  6
False negatives: 312
True negatives:   76,537
```

The large number of false negatives at this threshold reinforces why ranking metrics are more informative for this rare-event use case.

---

## 10. Model Explainability

The Logistic Regression coefficients are interpreted on standardized features.

The largest absolute standardized coefficients include:

```text
historical_avg_order_value   -0.443
historical_seller_count      -0.407
recency_days                 -0.402
frequency                    +0.370
monetary                     +0.304
avg_sellers_per_order        +0.260
customer_age_days            +0.257
historical_item_count        +0.243
avg_items_per_order          -0.237
```

For a standardized feature, `exp(coefficient)` represents the model's odds multiplier for a one-standard-deviation increase, holding the other model features constant.

These coefficients describe model associations, not causal effects.

Several features are also related mathematically or behaviorally, so individual coefficients should not be interpreted as independent causal drivers.

---

## 11. Temporal Drift and Stability

Temporal diagnostics compare the feature distributions of later snapshots against the earliest training snapshot.

The strongest observed drift was concentrated in:

* `recency_days`
* `customer_age_days`

By the final test snapshot, PSI was approximately:

```text
recency_days       0.518
customer_age_days  0.523
```

Both indicate substantial distributional change under the project's PSI thresholds.

Most other monitored behavioral and transaction-value features showed substantially lower PSI.

The target prevalence also changed over time:

```text
2017-09-01   1.019%
2017-12-01   0.879%
2018-03-01   0.796%
2018-06-19   0.409%
```

The project reports these changes as monitoring evidence. It does not claim that feature drift caused the target-prevalence decline.

---

## 12. Production Artifact Validation

The repository contains frozen production artifacts:

```text
data/model/artifacts/
├── logistic_model.joblib
├── sigmoid_calibrator.joblib
└── model_metadata.json
```

The production validation script checks:

* Model version
* Feature count
* Feature names
* Pipeline structure
* Coefficient dimensions
* Calibrator availability
* Test schema
* Leakage-prone feature names
* Batch prediction
* Probability bounds
* Deterministic inference

Current validation result:

```text
STEP 24 RESULT: PASS
Model remains frozen.
No fitting or recalibration performed.
```

This is an artifact/inference validation check, not a live production deployment or load test.

---

## 13. Reproducibility

The project pins its direct Python dependencies in:

```text
requirements.txt
```

The current development environment uses:

```text
Python 3.14.7
scikit-learn 1.9.1
pandas 3.0.6
numpy 2.5.3
scipy 1.18.1
duckdb 1.5.5
joblib 1.6.0
xgboost 3.4.1
```

The repository separates:

1. Data preparation
2. Model development
3. Model selection
4. Calibration
5. Artifact locking
6. Frozen evaluation
7. Error analysis
8. Explainability
9. Drift diagnostics
10. Production artifact validation

The final evaluation stage does not retrain or recalibrate the model.

---

## 14. Repository Structure

```text
customer-experience-churn-intelligence/
│
├── data/
│   └── model/
│       └── artifacts/
│
├── docs/
│   └── ARCHITECTURE.md
│
├── notebooks/
│
├── sql/
│   ├── customer metrics
│   ├── customer experience
│   ├── snapshot features
│   ├── future targets
│   ├── modeling table
│   └── leakage audit
│
├── src/
│   ├── data preparation
│   ├── feature engineering
│   ├── temporal validation
│   ├── model comparison
│   ├── model training
│   ├── calibration
│   ├── final evaluation
│   ├── error analysis
│   ├── explainability
│   ├── drift diagnostics
│   └── production validation
│
├── requirements.txt
└── README.md
```

Raw data and generated analysis outputs are excluded from version control.

---

## 15. Key Limitations

### Single development temporal fold

Only two snapshots are available in the training period, so model comparison uses one genuine expanding temporal development fold rather than a conventional multi-fold cross-validation design.

### Severe class imbalance

The final test buyer rate is only 0.409%. Threshold-based accuracy is therefore highly misleading.

### Historical dataset

The final test period is historical rather than a live production stream.

### No live deployment

The project validates production-style artifacts and deterministic inference but does not claim a deployed, monitored production service.

### No causal inference

Model coefficients and segment patterns describe statistical associations within the model. They do not establish that changing a feature will cause purchasing behavior to change.

### Temporal distribution shift

Recency and customer age exhibit substantial distributional drift between the earliest training snapshot and the final test snapshot.

---

## 16. Core Design Principles

The project prioritizes:

1. Temporal validation over random splitting
2. Leakage prevention over metric optimization
3. Frozen artifacts over repeated test-set experimentation
4. Ranking metrics for rare-event targeting
5. Probability quality through calibration
6. Interpretable models where appropriate
7. Explicit limitations over inflated claims
8. Reproducible production-style inference
9. Separation of development and final test evaluation
10. Transparent error and drift diagnostics

---

## 17. Project Status

```text
Data validation                         COMPLETE
Feature engineering                     COMPLETE
Temporal split                          COMPLETE
Model comparison                        COMPLETE
Canonical model selection               COMPLETE
Probability calibration                 COMPLETE
Frozen final test evaluation            COMPLETE
Error analysis                          COMPLETE
Explainability                          COMPLETE
Temporal drift diagnostics               COMPLETE
Production artifact validation           COMPLETE
Repository reproducibility               IN PROGRESS
Documentation                            IN PROGRESS
```
