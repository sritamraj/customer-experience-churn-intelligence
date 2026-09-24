from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "model"

INPUT_FILE = DATA_DIR / "temporal_model_comparison.csv"

OUTPUT_FILE = DATA_DIR / "model_selection_report.csv"


def main():

    print("=" * 70)
    print("STEP 20 — MODEL SELECTION RECORD")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    required_models = {
        "Frequency Baseline",
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
    }

    actual_models = set(df["model"])

    missing_models = required_models - actual_models

    assert not missing_models, (
        f"Missing expected models: {missing_models}"
    )

    logistic = df[
        df["model"] == "Logistic Regression"
    ].iloc[0]

    rf = df[
        df["model"] == "Random Forest"
    ].iloc[0]

    xgb = df[
        df["model"] == "XGBoost"
    ].iloc[0]

    baseline = df[
        df["model"] == "Frequency Baseline"
    ].iloc[0]

    # ------------------------------------------------------------
    # Integrity checks
    # ------------------------------------------------------------

    assert logistic["pr_auc"] > baseline["pr_auc"]

    assert logistic["log_loss"] < baseline["log_loss"]

    assert logistic["top_1_lift"] > baseline["pr_auc"] / baseline["pr_auc"]

    # ------------------------------------------------------------
    # Model-selection record
    # ------------------------------------------------------------

    record = {

        "selected_model":
            "Logistic Regression",

        "selected_model_version":
            "logistic_16_sigmoid_v1",

        "selection_basis":
            (
                "Retained as canonical model based on development "
                "temporal comparison: lowest log loss and highest "
                "PR-AUC among compared models, with strong top-1% "
                "ranking performance."
            ),

        "development_fold":
            "2017-09-01 train -> 2017-12-01 validation",

        "frequency_baseline_log_loss":
            baseline["log_loss"],

        "logistic_log_loss":
            logistic["log_loss"],

        "logistic_roc_auc":
            logistic["roc_auc"],

        "logistic_pr_auc":
            logistic["pr_auc"],

        "logistic_top_1_lift":
            logistic["top_1_lift"],

        "logistic_top_5_lift":
            logistic["top_5_lift"],

        "logistic_top_10_lift":
            logistic["top_10_lift"],

        "logistic_top_20_lift":
            logistic["top_20_lift"],

        "random_forest_log_loss":
            rf["log_loss"],

        "random_forest_roc_auc":
            rf["roc_auc"],

        "random_forest_pr_auc":
            rf["pr_auc"],

        "xgboost_log_loss":
            xgb["log_loss"],

        "xgboost_roc_auc":
            xgb["roc_auc"],

        "xgboost_pr_auc":
            xgb["pr_auc"],

        "selection_scope":
            (
                "Development evidence only. One temporal fold; "
                "not a claim of universal model superiority."
            ),

        "final_test_status":
            "2018-06-19 final test remains protected.",

        "calibration_status":
            (
                "Canonical Logistic model retains existing "
                "sigmoid calibration workflow."
            ),
    }

    output = pd.DataFrame(
        [record]
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("SELECTED MODEL:")
    print("Logistic Regression")
    print()
    print("MODEL VERSION:")
    print("logistic_16_sigmoid_v1")
    print()
    print("DEVELOPMENT FOLD:")
    print("2017-09-01 -> 2017-12-01")
    print()
    print("FINAL TEST:")
    print("2018-06-19 remains protected")
    print()
    print("Saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()