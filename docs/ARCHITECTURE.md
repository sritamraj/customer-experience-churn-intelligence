# System Architecture

## 1. System Objective

The system estimates customer purchase propensity over a fixed 120-day future horizon using information available at each historical customer snapshot.

The primary output is a calibrated propensity score that can be used to rank customers for potential prioritization.

---

## 2. End-to-End Architecture

```text
Historical Raw Data
        |
        v
Data Validation
        |
        v
SQL / Customer-Level Feature Engineering
        |
        v
Temporal Customer Snapshots
        |
        v
Modeling Table
        |
        v
Leakage Audit
        |
        +-----------------------------+
        |                             |
        v                             v
 Development Data              Final Test Data
 2017-09-01                    2018-06-19
 2017-12-01                    PROTECTED
        |
        v
Temporal Model Comparison
        |
        +-----------------------------+
        |             |               |
        v             v               v
 Frequency       Logistic          XGBoost
 Baseline        Regression
                      |
                      +---- Random Forest
                      |
                      v
               Model Selection
                      |
                      v
              Logistic Regression
                      |
                      v
              Sigmoid Calibration
                      |
                      v
                Artifact Lock
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
     Final Test   Error       Explainability
     Evaluation   Analysis
          |
          v
   Temporal Drift
     Diagnostics
          |
          v
 Production Artifact
     Validation
```

---

## 3. Temporal Data Contract

| Dataset    | Snapshot   | Purpose                             |
| ---------- | ---------- | ----------------------------------- |
| Train      | 2017-09-01 | Model development                   |
| Train      | 2017-12-01 | Model development                   |
| Validation | 2018-03-01 | Temporal validation and calibration |
| Test       | 2018-06-19 | Final frozen evaluation             |

The test snapshot is strictly later than all training and validation snapshots.

The final test data is not used for:

* model fitting
* model selection
* hyperparameter tuning
* feature selection
* calibration

---

## 4. Target Contract

For a snapshot date `S`, the primary target is:

```text
future_purchase_flag = 1
```

when a customer makes at least one purchase after `S` and within the following 120-day horizon.

The purchase event defines the primary propensity label.

Delivery completion is not required for the primary purchase target.

---

## 5. Feature Contract

The model expects exactly 16 features:

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

No future outcome columns are included.

---

## 6. Model Pipeline

```text
Raw model features
        |
        v
Median Imputation
        |
        v
StandardScaler
        |
        v
Logistic Regression
        |
        v
Raw probability / decision score
        |
        v
Frozen Sigmoid Calibrator
        |
        v
Calibrated Purchase Propensity
        |
        v
Customer Ranking
```

Canonical model:

```text
logistic_16_sigmoid_v1
```

---

## 7. Model Configuration

```text
Model:
    LogisticRegression

class_weight:
    balanced

C:
    1.0

max_iter:
    2000

random_state:
    42
```

The calibration stage uses a separate frozen sigmoid model.

---

## 8. Model Selection Contract

Development model selection uses:

```text
Train:
    2017-09-01

Validation:
    2017-12-01
```

Compared approaches:

```text
Frequency baseline
Logistic Regression
Random Forest
XGBoost
```

Primary development metrics:

```text
Log loss
ROC-AUC
PR-AUC
Top-k lift
```

The final test snapshot is not part of model selection.

---

## 9. Calibration Contract

The selected Logistic Regression model produces a raw score.

A separate sigmoid calibration model is fitted during the development/validation stage.

After calibration:

```text
raw model score
      |
      v
frozen sigmoid calibrator
      |
      v
calibrated probability
```

The calibration artifact is frozen before final test evaluation.

---

## 10. Artifact Contract

```text
data/model/artifacts/
├── logistic_model.joblib
├── sigmoid_calibrator.joblib
└── model_metadata.json
```

Metadata specifies:

```text
model version
feature names
feature count
target
imputation strategy
scaling strategy
class weighting
model configuration
training snapshots
calibration snapshot
test snapshot
```

---

## 11. Final Test Contract

The final test process:

1. Loads the frozen model.
2. Loads the frozen calibrator.
3. Loads model metadata.
4. Validates model version.
5. Validates the 16-feature contract.
6. Validates pipeline structure.
7. Loads the protected test snapshot.
8. Produces raw model scores.
9. Applies the frozen calibration model.
10. Calculates final evaluation metrics.
11. Calculates ranking metrics.
12. Writes analysis outputs.

It does **not**:

```text
fit
retrain
recalibrate
tune
select features
modify model artifacts
```

---

## 12. Evaluation Architecture

The system evaluates three distinct properties.

### Probability quality

```text
Calibrated log loss
```

### Ranking/discrimination

```text
ROC-AUC
PR-AUC
```

### Business-oriented prioritization

```text
Top 1%
Top 5%
Top 10%
Top 20%

Lift
Buyer capture
```

Because the positive class is rare, ranking metrics receive greater emphasis than accuracy at a default 0.50 threshold.

---

## 13. Error Analysis Architecture

Frozen test predictions are analyzed by:

```text
True positives
True negatives
False positives
False negatives
```

Additional diagnostics include:

```text
Probability distributions
Segment-level error rates
High-confidence false negatives
High-confidence false positives
```

No model fitting occurs during error analysis.

---

## 14. Explainability Architecture

The canonical model is a standardized Logistic Regression model.

Interpretation uses:

```text
standardized coefficient
        |
        v
log-odds effect
        |
        v
exp(coefficient)
        |
        v
odds multiplier per +1 SD
```

Coefficient interpretation is conditional on the other model features and is not treated as causal evidence.

---

## 15. Temporal Drift Architecture

Feature distributions are compared across customer snapshots.

The project uses Population Stability Index (PSI) as a diagnostic:

```text
2017-09-01 reference
        |
        +---- 2017-12-01
        |
        +---- 2018-03-01
        |
        +---- 2018-06-19
```

The drift analysis also monitors:

* customer population size
* buyer prevalence
* missingness
* feature distribution changes

Drift diagnostics do not automatically retrain the model.

---

## 16. Production Validation Architecture

The production validation stage checks:

```text
Metadata
   ↓
Model artifact
   ↓
Pipeline structure
   ↓
Coefficient dimensions
   ↓
Calibrator
   ↓
Input schema
   ↓
Feature leakage names
   ↓
Batch prediction
   ↓
Probability bounds
   ↓
Deterministic inference
```

The validation confirms that the frozen artifacts can perform production-style inference.

It does not represent a live deployment, serving benchmark, or load test.

---

## 17. Repository Execution Layers

### Layer A — Data

```text
build_olist_database.py
run_sql.py
build_customer_features.py
build_modeling_table.py
run_future_targets.py
```

### Layer B — Model preparation

```text
prepare_model_data.py
evaluate_baseline.py
compare_temporal_models.py
validate_temporal_logistic.py
```

### Layer C — Canonical model

```text
train_final_logistic.py
calibrate_logistic.py
```

### Layer D — Frozen evaluation

```text
analyze_logistic_final_test.py
analyze_logistic_final_error_analysis.py
write_logistic_explainability.py
diagnose_temporal_drift.py
validate_production_artifacts.py
```

### Layer E — Supporting analysis

```text
analyze_logistic_segment_drivers.py
analyze_logistic_local_explanations.py
analyze_logistic_targeting.py
```

---

## 18. Design Principles

The system follows these principles:

1. Temporal ordering over random splitting.
2. Leakage prevention over metric optimization.
3. Development/test separation.
4. Frozen artifacts for final evaluation.
5. Ranking metrics for rare-event targeting.
6. Calibration for probability quality.
7. Explicit model and feature contracts.
8. Explainability without causal overclaiming.
9. Drift monitoring without automatic retraining.
10. Reproducible production-style inference.
11. Explicit limitations rather than inflated claims.
