from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DATA_DIR = Path("data/model")


def evaluate_ranking(y_true, scores, name):

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    # ------------------------------------------------------------
    # Overall metrics
    # ------------------------------------------------------------

    pr_auc = average_precision_score(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)

    print(f"PR-AUC : {pr_auc:.6f}")
    print(f"ROC-AUC: {roc_auc:.6f}")

    # ------------------------------------------------------------
    # Precision / recall at top K
    # ------------------------------------------------------------

    ranking = pd.DataFrame({
        "y": y_true,
        "score": scores
    })

    ranking = ranking.sort_values(
        "score",
        ascending=False
    ).reset_index(drop=True)

    print("\nRanking metrics:")

    for k_pct in [0.01, 0.05, 0.10]:

        k = max(
            1,
            int(len(ranking) * k_pct)
        )

        top_k = ranking.iloc[:k]

        precision = top_k["y"].mean()

        recall = (
            top_k["y"].sum()
            / ranking["y"].sum()
        )

        baseline_rate = ranking["y"].mean()

        lift = (
            precision / baseline_rate
            if baseline_rate > 0
            else np.nan
        )

        print(
            f"Top {int(k_pct * 100):2d}% "
            f"| K={k:6,} "
            f"| Precision={precision:.4f} "
            f"| Recall={recall:.4f} "
            f"| Lift={lift:.2f}x"
        )


def main():

    print("=" * 70)
    print("STAGE 1 — BASELINE PURCHASE PREDICTION")
    print("=" * 70)

    train = pd.read_csv(DATA_DIR / "train.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    # ------------------------------------------------------------
    # Frequency baseline
    # ------------------------------------------------------------

    print("\nFrequency baseline")

    evaluate_ranking(
        train["future_purchase_flag"],
        train["frequency"],
        "TRAIN — Frequency Baseline"
    )

    evaluate_ranking(
        validation["future_purchase_flag"],
        validation["frequency"],
        "VALIDATION — Frequency Baseline"
    )

    evaluate_ranking(
        test["future_purchase_flag"],
        test["frequency"],
        "TEST — Frequency Baseline"
    )


if __name__ == "__main__":
    main()