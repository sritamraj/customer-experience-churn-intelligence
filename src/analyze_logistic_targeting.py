from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# STEP 15 — LOCKED-MODEL BUSINESS TARGETING ANALYSIS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "data" / "model"

INPUT_FILE = MODEL_DIR / "logistic_validation_error_scores.csv"

OUTPUT_PERFORMANCE = MODEL_DIR / "logistic_targeting_performance.csv"
OUTPUT_SUMMARY = MODEL_DIR / "logistic_targeting_summary.csv"


TARGET_PCTS = [0.01, 0.05, 0.10, 0.20]


def validate_input(df: pd.DataFrame) -> None:
    required_columns = {
        "row_index",
        "actual",
        "predicted_probability",
        "snapshot_date",
        "rank",
        "rank_pct",
        "ranking_segment",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if len(df) != 52013:
        raise ValueError(
            f"Expected 52,013 validation rows, found {len(df):,}."
        )

    buyer_count = int(df["actual"].sum())

    if buyer_count != 414:
        raise ValueError(
            f"Expected 414 validation buyers, found {buyer_count}."
        )

    if df["row_index"].duplicated().any():
        raise ValueError("Duplicate row_index values detected.")

    if df["actual"].isna().any():
        raise ValueError("Missing actual labels detected.")

    if df["predicted_probability"].isna().any():
        raise ValueError(
            "Missing predicted probabilities detected."
        )

    if not df["predicted_probability"].between(0, 1).all():
        raise ValueError(
            "Predicted probabilities outside [0, 1] detected."
        )

    if not df["rank"].is_monotonic_increasing:
        raise ValueError(
            "Validation scores are not sorted by rank."
        )


def main() -> None:
    print("=" * 70)
    print("STEP 15 — LOCKED-MODEL BUSINESS TARGETING")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input artifact: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    validate_input(df)

    df = df.sort_values("rank").reset_index(drop=True)

    total_customers = len(df)
    total_buyers = int(df["actual"].sum())
    base_rate = total_buyers / total_customers

    snapshot_values = df["snapshot_date"].dropna().unique()

    if len(snapshot_values) != 1:
        raise ValueError(
            "Expected exactly one validation snapshot."
        )

    validation_snapshot = str(snapshot_values[0])

    print(f"Validation snapshot: {validation_snapshot}")
    print(f"Validation customers: {total_customers:,}")
    print(f"Validation buyers: {total_buyers:,}")
    print(f"Validation base rate: {base_rate:.6f}")
    print()

    rows = []

    for pct in TARGET_PCTS:
        target_count = int(np.ceil(total_customers * pct))

        targeted = df.iloc[:target_count]

        targeted_customers = len(targeted)
        targeted_buyers = int(targeted["actual"].sum())

        buyer_rate = targeted_buyers / targeted_customers

        lift = buyer_rate / base_rate

        capture = targeted_buyers / total_buyers

        random_expected_buyers = (
            targeted_customers * base_rate
        )

        incremental_buyers_vs_random = (
            targeted_buyers - random_expected_buyers
        )

        random_expected_capture = pct

        capture_vs_random = (
            capture / random_expected_capture
        )

        rows.append(
            {
                "target_pct": pct,
                "target_pct_label": f"Top {int(pct * 100)}%",
                "customers_targeted": targeted_customers,
                "actual_buyers": targeted_buyers,
                "buyer_rate": buyer_rate,
                "base_rate": base_rate,
                "lift": lift,
                "buyer_capture": capture,
                "random_expected_buyers": random_expected_buyers,
                "incremental_buyers_vs_random":
                    incremental_buyers_vs_random,
                "random_expected_capture":
                    random_expected_capture,
                "capture_vs_random":
                    capture_vs_random,
            }
        )

    performance = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Full ranking curve
    # --------------------------------------------------------

    curve_rows = []

    for pct in np.linspace(0.01, 1.00, 100):
        target_count = int(np.ceil(total_customers * pct))

        targeted = df.iloc[:target_count]

        targeted_buyers = int(targeted["actual"].sum())
        buyer_rate = targeted_buyers / target_count
        capture = targeted_buyers / total_buyers
        lift = buyer_rate / base_rate

        curve_rows.append(
            {
                "target_pct": pct,
                "customers_targeted": target_count,
                "actual_buyers": targeted_buyers,
                "buyer_rate": buyer_rate,
                "lift": lift,
                "buyer_capture": capture,
            }
        )

    curve = pd.DataFrame(curve_rows)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "validation_snapshot": validation_snapshot,
                "validation_customers": total_customers,
                "validation_buyers": total_buyers,
                "base_rate": base_rate,
                "top_1pct_lift":
                    performance.loc[
                        performance["target_pct"] == 0.01,
                        "lift"
                    ].iloc[0],
                "top_1pct_capture":
                    performance.loc[
                        performance["target_pct"] == 0.01,
                        "buyer_capture"
                    ].iloc[0],
                "top_5pct_lift":
                    performance.loc[
                        performance["target_pct"] == 0.05,
                        "lift"
                    ].iloc[0],
                "top_5pct_capture":
                    performance.loc[
                        performance["target_pct"] == 0.05,
                        "buyer_capture"
                    ].iloc[0],
                "top_10pct_lift":
                    performance.loc[
                        performance["target_pct"] == 0.10,
                        "lift"
                    ].iloc[0],
                "top_10pct_capture":
                    performance.loc[
                        performance["target_pct"] == 0.10,
                        "buyer_capture"
                    ].iloc[0],
                "top_20pct_lift":
                    performance.loc[
                        performance["target_pct"] == 0.20,
                        "lift"
                    ].iloc[0],
                "top_20pct_capture":
                    performance.loc[
                        performance["target_pct"] == 0.20,
                        "buyer_capture"
                    ].iloc[0],
            }
        ]
    )

    performance.to_csv(
        OUTPUT_PERFORMANCE,
        index=False
    )

    curve.to_csv(
        MODEL_DIR / "logistic_targeting_curve.csv",
        index=False
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print("TARGETING PERFORMANCE")
    print("-" * 70)

    display_columns = [
        "target_pct_label",
        "customers_targeted",
        "actual_buyers",
        "buyer_rate",
        "lift",
        "buyer_capture",
        "incremental_buyers_vs_random",
        "capture_vs_random",
    ]

    display_df = performance[display_columns].copy()

    display_df["buyer_rate"] = (
        display_df["buyer_rate"] * 100
    ).round(3)

    display_df["lift"] = (
        display_df["lift"]
    ).round(3)

    display_df["buyer_capture"] = (
        display_df["buyer_capture"] * 100
    ).round(3)

    display_df["incremental_buyers_vs_random"] = (
        display_df["incremental_buyers_vs_random"]
    ).round(2)

    display_df["capture_vs_random"] = (
        display_df["capture_vs_random"]
    ).round(3)

    print(
        display_df.to_string(index=False)
    )

    print()
    print("INTERPRETATION")
    print("-" * 70)

    top1 = performance.iloc[0]
    top5 = performance.iloc[1]
    top10 = performance.iloc[2]
    top20 = performance.iloc[3]

    print(
        f"Top 1%: {int(top1['actual_buyers'])} buyers captured "
        f"with {top1['lift']:.2f}x lift."
    )

    print(
        f"Top 5%: {int(top5['actual_buyers'])} buyers captured "
        f"with {top5['buyer_capture'] * 100:.2f}% buyer capture."
    )

    print(
        f"Top 10%: {int(top10['actual_buyers'])} buyers captured "
        f"with {top10['buyer_capture'] * 100:.2f}% buyer capture."
    )

    print(
        f"Top 20%: {int(top20['actual_buyers'])} buyers captured "
        f"with {top20['buyer_capture'] * 100:.2f}% buyer capture."
    )

    print()
    print("IMPORTANT:")
    print(
        "These are observational targeting-enrichment results, "
        "not causal campaign effects."
    )

    print()
    print(f"Wrote: {OUTPUT_PERFORMANCE}")
    print(f"Wrote: {MODEL_DIR / 'logistic_targeting_curve.csv'}")
    print(f"Wrote: {OUTPUT_SUMMARY}")

    print("=" * 70)
    print("STEP 15 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()