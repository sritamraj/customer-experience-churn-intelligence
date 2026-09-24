import json
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "data" / "model" / "artifacts"
OUT = ROOT / "data" / "model"

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

metadata = json.loads((ARTIFACT / "model_metadata.json").read_text())
assert metadata["model_version"] == "logistic_16_sigmoid_v1"
assert metadata["features"] == FEATURES
assert metadata["feature_count"] == 16

model = joblib.load(ARTIFACT / "logistic_model.joblib")
assert list(model.named_steps.keys()) == ["imputer", "scaler", "model"]

lr = model.named_steps["model"]
coef = lr.coef_[0]
intercept = float(lr.intercept_[0])

explain = pd.DataFrame({
    "feature": FEATURES,
    "coefficient": coef,
})
explain["abs_coefficient"] = explain["coefficient"].abs()
explain["direction"] = explain["coefficient"].apply(
    lambda x: "positive" if x > 0 else "negative" if x < 0 else "zero"
)
explain["odds_multiplier_per_1sd"] = explain["coefficient"].apply(__import__("math").exp)
explain["rank_by_abs_coefficient"] = (
    explain["abs_coefficient"].rank(method="min", ascending=False).astype(int)
)
explain = explain.sort_values("rank_by_abs_coefficient")

OUT.mkdir(parents=True, exist_ok=True)
explain.to_csv(OUT / "logistic_explainability_coefficients.csv", index=False)

summary = pd.DataFrame([{
    "model_version": metadata["model_version"],
    "intercept": intercept,
    "feature_count": len(FEATURES),
    "largest_positive_feature": explain.loc[explain["coefficient"].idxmax(), "feature"],
    "largest_positive_coefficient": float(explain["coefficient"].max()),
    "largest_negative_feature": explain.loc[explain["coefficient"].idxmin(), "feature"],
    "largest_negative_coefficient": float(explain["coefficient"].min()),
}])
summary.to_csv(OUT / "logistic_explainability_summary.csv", index=False)

print("=" * 70)
print("STEP 22 — FROZEN LOGISTIC EXPLAINABILITY")
print("=" * 70)
print(f"Model version: {metadata['model_version']}")
print(f"Intercept: {intercept:.8f}")
print()
print("COEFFICIENTS RANKED BY ABSOLUTE MAGNITUDE")
print(explain[["rank_by_abs_coefficient", "feature", "coefficient", "direction", "odds_multiplier_per_1sd"]].to_string(index=False))
print()
print("Artifacts written:")
print(OUT / "logistic_explainability_coefficients.csv")
print(OUT / "logistic_explainability_summary.csv")
