\# System Architecture



\## 1. Objective



Build a leakage-safe customer purchase-propensity ranking system using historical customer behavior and customer-experience features.



The system predicts the probability of a future purchase and ranks customers for potential prioritization.



\---



\## 2. End-to-End Pipeline



```text

Raw Data

&#x20;  |

&#x20;  v

Data Validation

&#x20;  |

&#x20;  v

Temporal Customer Snapshots

&#x20;  |

&#x20;  v

Feature Engineering

&#x20;  |

&#x20;  v

Modeling Table

&#x20;  |

&#x20;  +----------------------+

&#x20;  |                      |

&#x20;  v                      v

TRAIN                  VALIDATION

2017-09                2018-03

2017-12

&#x20;  |                      |

&#x20;  +----------+-----------+

&#x20;             |

&#x20;             v

&#x20;   Logistic Regression

&#x20;             |

&#x20;             v

&#x20;      Sigmoid Calibration

&#x20;             |

&#x20;             v

&#x20;       Artifact Lock

&#x20;             |

&#x20;      +------+------+

&#x20;      |             |

&#x20;      v             v

&#x20;Error Analysis   Explainability

&#x20;      |             |

&#x20;      +------+------+

&#x20;             |

&#x20;             v

&#x20;      Business Targeting

&#x20;             |

&#x20;             v

&#x20;     FINAL TEST AUDIT

&#x20;        2018-06-19

```



\---



\## 3. Temporal Contract



| Dataset    | Period     | Purpose                    |

| ---------- | ---------- | -------------------------- |

| Train      | 2017-09-01 | Model development          |

| Train      | 2017-12-01 | Model development          |

| Validation | 2018-03-01 | Model evaluation / locking |

| Test       | 2018-06-19 | Final untouched audit      |



The final test period is strictly later than training and validation.



\---



\## 4. Feature Contract



The model expects exactly 16 features:



```text

recency\_days

customer\_age\_days

frequency

monetary

historical\_avg\_order\_value

historical\_avg\_review\_score

historical\_avg\_delivery\_days

historical\_avg\_delivery\_delay

historical\_freight\_value

historical\_item\_count

historical\_product\_count

historical\_seller\_count

non\_delivered\_order\_count

avg\_freight\_per\_order

avg\_items\_per\_order

avg\_sellers\_per\_order

```



Future outcome fields are never passed to the model.



\---



\## 5. Model Contract



```text

Pipeline

├── imputer

│   └── median strategy

├── scaler

│   └── StandardScaler

└── model

&#x20;   └── LogisticRegression

```



The logistic model uses balanced class weights.



A separate sigmoid calibrator transforms raw model scores into calibrated purchase probabilities.



\---



\## 6. Artifact Contract



```text

data/model/artifacts/

├── logistic\_model.joblib

├── sigmoid\_calibrator.joblib

└── model\_metadata.json

```



Model version:



```text

logistic\_16\_sigmoid\_v1

```



The metadata and scoring code enforce the expected model version, feature set, and pipeline structure.



\---



\## 7. Scoring Contract



For each customer:



```text

16 features

&#x20;  |

&#x20;  v

Median imputation

&#x20;  |

&#x20;  v

Standardization

&#x20;  |

&#x20;  v

Logistic decision score

&#x20;  |

&#x20;  v

Sigmoid calibration

&#x20;  |

&#x20;  v

Purchase propensity

&#x20;  |

&#x20;  v

Customer ranking

```



No fitting occurs during final test scoring.



\---



\## 8. Evaluation Contract



The project evaluates the model at three levels.



\### Probability quality



\* Calibrated log loss



\### Discrimination



\* ROC AUC



\### Business-oriented ranking



\* Top 1%

\* Top 5%

\* Top 10%

\* Top 20%

\* lift

\* buyer capture



Because the positive class is extremely rare, ranking metrics are emphasized over a default 0.50 classification threshold.



\---



\## 9. Final Test Contract



The final test script must:



1\. Load the frozen model.

2\. Load the frozen sigmoid calibrator.

3\. Load model metadata.

4\. Validate model version.

5\. Validate pipeline structure.

6\. Validate the 16-feature contract.

7\. Transform the test features.

8\. Score the test observations.

9\. Apply the existing calibrator.

10\. Calculate final metrics.

11\. Write final outputs.



The final test script must not:



\* fit

\* retrain

\* recalibrate

\* tune

\* select features

\* modify artifacts



\---



\## 10. Design Principles



The project follows these principles:



1\. \*\*Temporal ordering over random splitting\*\*

2\. \*\*Leakage prevention over metric optimization\*\*

3\. \*\*Locked artifacts over repeated experimentation\*\*

4\. \*\*Ranking evaluation for rare-event targeting\*\*

5\. \*\*Interpretability over unnecessary model complexity\*\*

6\. \*\*Explicit limitations over inflated claims\*\*

7\. \*\*Final untouched test evaluation\*\*



