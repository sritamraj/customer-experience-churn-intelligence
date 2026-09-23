from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "data" / "model"
ARTIFACT_DIR = MODEL_DIR / "artifacts"
OUTPUT_DIR = MODEL_DIR / "monitoring"

VALIDATION_PATH = MODEL_DIR / "logistic_validation_scored.csv"
TEST_PATH = MODEL_DIR / "logistic_test_scored.csv"

MODEL_PATH = ARTIFACT_DIR / "logistic_model.joblib"
CALIBRATOR_PATH = ARTIFACT_DIR / "sigmoid_calibrator.joblib"

OUTPUT_SUMMARY = (
    OUTPUT_DIR / "outcome_calibration_summary.csv"
)

OUTPUT_TOPK = (
    OUTPUT_DIR / "outcome_calibration_topk.csv"
)

OUTPUT_DECILES = (
    OUTPUT_DIR / "outcome_calibration_deciles.csv"
)


FEATURES = [
    "recency_days",
    "customer_age_days",
    "frequency",
    "monetary",
    "historical_avg_order_value",
    "historical_avg_review_score",
    "historical_avg_delivery_days",
    "historical_avg_delivery_delay",
    "historical_freight_value",
    "historical_item_count",
    "historical_product_count",
    "historical_seller_count",
    "non_delivered_order_count",
    "avg_freight_per_order",
    "avg_items_per_order",
    "avg_sellers_per_order",
]


TARGET = "future_purchase_flag"


def load_data(path, name):

    df = pd.read_csv(path)

    print(
        f"{name} rows: "
        f"{len(df):,}"
    )

    required = FEATURES + [TARGET]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{name} missing columns: "
            f"{missing}"
        )

    return df


def score_data(df, model, calibrator, name):

    X = df[FEATURES].copy()

    y = df[TARGET].astype(int).to_numpy()

    raw_score = model.decision_function(X)

    calibrated_probability = (
        calibrator.predict_proba(
            raw_score
        )
    )

    if np.isnan(calibrated_probability).any():
        raise RuntimeError(
            f"{name}: NaN probabilities."
        )

    if (
        (calibrated_probability < 0)
        |
        (calibrated_probability > 1)
    ).any():

        raise RuntimeError(
            f"{name}: probability outside [0,1]."
        )

    return y, calibrated_probability


def top_k_metrics(y, probability, fraction):

    n = len(y)

    k = max(
        1,
        int(np.ceil(n * fraction)),
    )

    order = np.argsort(
        -probability
    )

    selected = order[:k]

    selected_y = y[selected]

    buyers = int(y.sum())

    selected_buyers = int(
        selected_y.sum()
    )

    precision = (
        selected_buyers / k
    )

    recall = (
        selected_buyers / buyers
        if buyers > 0
        else np.nan
    )

    base_rate = (
        buyers / n
    )

    lift = (
        precision / base_rate
        if base_rate > 0
        else np.nan
    )

    return {
        "selection_fraction": fraction,
        "selected_customers": k,
        "selected_buyers": selected_buyers,
        "precision": precision,
        "recall": recall,
        "lift": lift,
    }


def calibration_deciles(
    y,
    probability,
    snapshot_name,
):

    df = pd.DataFrame(
        {
            "actual": y,
            "probability": probability,
        }
    )

    # Rank-based deciles avoid problems caused by
    # repeated probabilities.
    df["decile"] = pd.qcut(
        df["probability"].rank(
            method="first"
        ),
        10,
        labels=False,
    ) + 1

    rows = []

    for decile, group in df.groupby(
        "decile",
        sort=True,
    ):

        rows.append(
            {
                "snapshot": snapshot_name,
                "decile": int(decile),
                "customers": len(group),
                "mean_predicted_probability":
                    group["probability"].mean(),
                "observed_purchase_rate":
                    group["actual"].mean(),
                "calibration_gap":
                    (
                        group["probability"].mean()
                        - group["actual"].mean()
                    ),
                "buyers":
                    int(group["actual"].sum()),
            }
        )

    return pd.DataFrame(rows)


def main():

    print("=" * 70)
    print(
        "OUTCOME / BASE-RATE DRIFT "
        "+ CALIBRATION MONITORING"
    )
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # 1. Load production artifacts
    # ---------------------------------------------------------------

    print(
        "\nLoading serialized production artifacts..."
    )

    model = joblib.load(
        MODEL_PATH
    )

    calibrator = joblib.load(
        CALIBRATOR_PATH
    )

    print(
        "Logistic model: PASS"
    )

    print(
        "Sigmoid calibrator: PASS"
    )

    # ---------------------------------------------------------------
    # 2. Load validation and test
    # ---------------------------------------------------------------

    print(
        "\nLoading validation data..."
    )

    validation = load_data(
        VALIDATION_PATH,
        "Validation",
    )

    print(
        "\nLoading test data..."
    )

    test = load_data(
        TEST_PATH,
        "Test",
    )

    # ---------------------------------------------------------------
    # 3. Score using locked production artifacts
    # ---------------------------------------------------------------

    print(
        "\nScoring validation..."
    )

    y_validation, p_validation = score_data(
        validation,
        model,
        calibrator,
        "Validation",
    )

    print(
        "Scoring test..."
    )

    y_test, p_test = score_data(
        test,
        model,
        calibrator,
        "Test",
    )

    # ---------------------------------------------------------------
    # 4. Base-rate metrics
    # ---------------------------------------------------------------

    validation_base_rate = (
        y_validation.mean()
    )

    test_base_rate = (
        y_test.mean()
    )

    relative_base_rate_change = (
        (
            test_base_rate
            - validation_base_rate
        )
        / validation_base_rate
        if validation_base_rate > 0
        else np.nan
    )

    # ---------------------------------------------------------------
    # 5. Prediction / outcome gap
    # ---------------------------------------------------------------

    validation_mean_probability = (
        p_validation.mean()
    )

    test_mean_probability = (
        p_test.mean()
    )

    validation_prediction_gap = (
        validation_mean_probability
        - validation_base_rate
    )

    test_prediction_gap = (
        test_mean_probability
        - test_base_rate
    )

    # ---------------------------------------------------------------
    # 6. Performance + calibration
    # ---------------------------------------------------------------

    validation_pr_auc = (
        average_precision_score(
            y_validation,
            p_validation,
        )
    )

    test_pr_auc = (
        average_precision_score(
            y_test,
            p_test,
        )
    )

    validation_roc_auc = (
        roc_auc_score(
            y_validation,
            p_validation,
        )
    )

    test_roc_auc = (
        roc_auc_score(
            y_test,
            p_test,
        )
    )

    validation_brier = (
        brier_score_loss(
            y_validation,
            p_validation,
        )
    )

    test_brier = (
        brier_score_loss(
            y_test,
            p_test,
        )
    )

    validation_logloss = (
        log_loss(
            y_validation,
            p_validation,
        )
    )

    test_logloss = (
        log_loss(
            y_test,
            p_test,
        )
    )

    # ---------------------------------------------------------------
    # 7. Top-K
    # ---------------------------------------------------------------

    fractions = [
        0.01,
        0.05,
        0.10,
        0.20,
        0.50,
    ]

    topk_rows = []

    for fraction in fractions:

        validation_metrics = top_k_metrics(
            y_validation,
            p_validation,
            fraction,
        )

        validation_metrics[
            "snapshot"
        ] = "validation"

        test_metrics = top_k_metrics(
            y_test,
            p_test,
            fraction,
        )

        test_metrics[
            "snapshot"
        ] = "test"

        topk_rows.append(
            validation_metrics
        )

        topk_rows.append(
            test_metrics
        )

    topk_df = pd.DataFrame(
        topk_rows
    )

    # ---------------------------------------------------------------
    # 8. Calibration deciles
    # ---------------------------------------------------------------

    validation_deciles = calibration_deciles(
        y_validation,
        p_validation,
        "validation",
    )

    test_deciles = calibration_deciles(
        y_test,
        p_test,
        "test",
    )

    deciles_df = pd.concat(
        [
            validation_deciles,
            test_deciles,
        ],
        ignore_index=True,
    )

    # ---------------------------------------------------------------
    # 9. Overall summary
    # ---------------------------------------------------------------

    summary_rows = [
        {
            "snapshot": "validation",
            "customers": len(y_validation),
            "buyers": int(y_validation.sum()),
            "actual_purchase_rate":
                validation_base_rate,
            "mean_predicted_probability":
                validation_mean_probability,
            "prediction_minus_actual_gap":
                validation_prediction_gap,
            "pr_auc":
                validation_pr_auc,
            "roc_auc":
                validation_roc_auc,
            "brier_score":
                validation_brier,
            "log_loss":
                validation_logloss,
        },
        {
            "snapshot": "test",
            "customers": len(y_test),
            "buyers": int(y_test.sum()),
            "actual_purchase_rate":
                test_base_rate,
            "mean_predicted_probability":
                test_mean_probability,
            "prediction_minus_actual_gap":
                test_prediction_gap,
            "pr_auc":
                test_pr_auc,
            "roc_auc":
                test_roc_auc,
            "brier_score":
                test_brier,
            "log_loss":
                test_logloss,
        },
    ]

    summary_df = pd.DataFrame(
        summary_rows
    )

    # ---------------------------------------------------------------
    # 10. Print base-rate drift
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "BASE-RATE DRIFT"
    )

    print(
        "=" * 70
    )

    print(
        f"Validation purchase rate: "
        f"{validation_base_rate:.6%}"
    )

    print(
        f"Test purchase rate: "
        f"{test_base_rate:.6%}"
    )

    print(
        f"Relative change: "
        f"{relative_base_rate_change:.2%}"
    )

    # ---------------------------------------------------------------
    # 11. Print calibration / performance
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "OUTCOME + CALIBRATION METRICS"
    )

    print(
        "=" * 70
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # 12. Print prediction-outcome gap
    # ---------------------------------------------------------------

    print(
        "\nPrediction vs actual:"
    )

    print(
        f"Validation gap: "
        f"{validation_prediction_gap:.6%}"
    )

    print(
        f"Test gap: "
        f"{test_prediction_gap:.6%}"
    )

    # ---------------------------------------------------------------
    # 13. Print Top-K
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP-K PERFORMANCE"
    )

    print(
        "=" * 70
    )

    print(
        topk_df[
            [
                "snapshot",
                "selection_fraction",
                "selected_customers",
                "selected_buyers",
                "precision",
                "recall",
                "lift",
            ]
        ].to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # 14. Print calibration deciles
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "CALIBRATION DECILES"
    )

    print(
        "=" * 70
    )

    print(
        deciles_df.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # 15. Save
    # ---------------------------------------------------------------

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    topk_df.to_csv(
        OUTPUT_TOPK,
        index=False,
    )

    deciles_df.to_csv(
        OUTPUT_DECILES,
        index=False,
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_SUMMARY
    )

    print(
        OUTPUT_TOPK
    )

    print(
        OUTPUT_DECILES
    )

    # ---------------------------------------------------------------
    # 16. Final monitoring interpretation
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "STEP 15C COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Test was evaluated only."
    )

    print(
        "No calibration refitting was performed."
    )

    print(
        "No model retraining was performed."
    )

    print(
        "No test-based tuning was performed."
    )


if __name__ == "__main__":
    main()