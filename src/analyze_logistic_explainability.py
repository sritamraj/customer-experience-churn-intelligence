from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ARTIFACT_DIR = ROOT / "data" / "model" / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "logistic_model.joblib"
CALIBRATOR_PATH = ARTIFACT_DIR / "sigmoid_calibrator.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

TRAIN_PATH = ROOT / "data" / "model" / "train.csv"
VALIDATION_PATH = ROOT / "data" / "model" / "validation.csv"

OUTPUT_DIR = ROOT / "data" / "model"


# ============================================================
# HELPERS
# ============================================================

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_csv(path)


def require_columns(df, columns, name):
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )


# ============================================================
# LOAD CANONICAL PRODUCTION ARTIFACTS
# ============================================================

print("=" * 70)
print("LOGISTIC MODEL EXPLAINABILITY")
print("=" * 70)

print("\nLoading canonical production artifacts...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Missing canonical model: {MODEL_PATH}"
    )

if not CALIBRATOR_PATH.exists():
    raise FileNotFoundError(
        f"Missing canonical calibrator: {CALIBRATOR_PATH}"
    )

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Missing model metadata: {METADATA_PATH}"
    )

model = joblib.load(MODEL_PATH)
calibrator = joblib.load(CALIBRATOR_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(
    f"Model version: {metadata.get('model_version')}"
)

locked_features = metadata.get("features")

if not locked_features:
    raise ValueError(
        "model_metadata.json does not contain locked feature list."
    )

print(
    f"Locked feature count: {len(locked_features)}"
)

print(
    f"Locked features: {locked_features}"
)


# ============================================================
# VERIFY CANONICAL MODEL STRUCTURE
# ============================================================

print("\nVerifying canonical model structure...")

print(
    f"Pipeline steps: {list(model.named_steps.keys())}"
)

if "imputer" not in model.named_steps:
    raise ValueError(
        "Canonical model does not contain expected imputer step."
    )

if "scaler" not in model.named_steps:
    raise ValueError(
        "Canonical model does not contain expected scaler step."
    )

if "model" not in model.named_steps:
    raise ValueError(
        "Canonical model does not contain expected model step."
    )

imputer = model.named_steps["imputer"]
scaler = model.named_steps["scaler"]
classifier = model.named_steps["model"]

print("Canonical model structure verified.")

print(
    f"Classifier: {classifier.__class__.__name__}"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading train and validation data...")

train = load_csv(TRAIN_PATH)
validation = load_csv(VALIDATION_PATH)

print(
    f"Training rows: {len(train):,}"
)

print(
    f"Validation rows: {len(validation):,}"
)


# ============================================================
# VERIFY FEATURES
# ============================================================

require_columns(
    train,
    locked_features,
    "Training data"
)

require_columns(
    validation,
    locked_features,
    "Validation data"
)

X_train = train[locked_features].copy()
X_validation = validation[locked_features].copy()

print("\nFeature schema verified.")


# ============================================================
# GLOBAL LOGISTIC EXPLAINABILITY
# ============================================================

print("\nComputing global feature importance...")

coefficients = classifier.coef_[0]

if len(coefficients) != len(locked_features):
    raise ValueError(
        "Coefficient count does not match locked feature count."
    )

global_importance = pd.DataFrame(
    {
        "feature": locked_features,
        "coefficient": coefficients,
        "odds_ratio": np.exp(coefficients),
        "abs_coefficient": np.abs(coefficients),
    }
)

global_importance["direction"] = np.where(
    global_importance["coefficient"] > 0,
    "positive",
    "negative",
)

global_importance = (
    global_importance
    .sort_values(
        "abs_coefficient",
        ascending=False,
    )
    .reset_index(drop=True)
)

global_importance["rank"] = (
    np.arange(len(global_importance)) + 1
)

global_importance = global_importance[
    [
        "rank",
        "feature",
        "coefficient",
        "odds_ratio",
        "abs_coefficient",
        "direction",
    ]
]


# ============================================================
# DRIVER SUMMARY
# ============================================================

positive_drivers = (
    global_importance[
        global_importance["coefficient"] > 0
    ]
    .sort_values(
        "coefficient",
        ascending=False,
    )
    .copy()
)

negative_drivers = (
    global_importance[
        global_importance["coefficient"] < 0
    ]
    .sort_values(
        "coefficient",
        ascending=True,
    )
    .copy()
)

driver_summary = pd.DataFrame(
    {
        "positive_driver_count": [
            len(positive_drivers)
        ],
        "negative_driver_count": [
            len(negative_drivers)
        ],
        "strongest_positive_driver": [
            (
                positive_drivers.iloc[0]["feature"]
                if len(positive_drivers) > 0
                else None
            )
        ],
        "strongest_negative_driver": [
            (
                negative_drivers.iloc[0]["feature"]
                if len(negative_drivers) > 0
                else None
            )
        ],
    }
)


# ============================================================
# VALIDATION SCORES
# ============================================================

print(
    "\nScoring validation data with canonical model..."
)

# No retraining occurs here.
# Reproduce the locked preprocessing path.

X_validation_imputed = imputer.transform(
    X_validation
)

X_validation_scaled = scaler.transform(
    X_validation_imputed
)

raw_validation_scores = classifier.decision_function(
    X_validation_scaled
)

# SigmoidCalibrator implements predict_proba(),
# not predict().

calibrated_validation_probability = (
    calibrator.predict_proba(
        raw_validation_scores
    )
)

calibrated_validation_probability = np.asarray(
    calibrated_validation_probability
).reshape(-1)

if len(calibrated_validation_probability) != len(
    validation
):
    raise ValueError(
        "Calibrated probability count does not "
        "match validation rows."
    )


# ============================================================
# BUILD EXPLAINABILITY TABLE
# ============================================================

validation_explainability = pd.DataFrame(
    {
        "row_index": np.arange(len(validation)),
        "raw_logit_score": raw_validation_scores,
        "calibrated_probability":
            calibrated_validation_probability,
    }
)


# ============================================================
# PRESERVE IDENTIFIERS / TARGET COLUMNS
# ============================================================

for column in [
    "customer_id",
    "snapshot_date",
    "target",
    "future_buyer",
]:
    if column in validation.columns:
        validation_explainability[column] = (
            validation[column].values
        )


# ============================================================
# RANK VALIDATION CUSTOMERS
# ============================================================

validation_explainability = (
    validation_explainability
    .sort_values(
        "calibrated_probability",
        ascending=False,
    )
    .reset_index(drop=True)
)

n = len(validation_explainability)

validation_explainability["rank"] = (
    np.arange(n) + 1
)

validation_explainability["rank_pct"] = (
    validation_explainability["rank"] / n
)

validation_explainability["segment"] = np.select(
    [
        validation_explainability["rank_pct"] <= 0.01,
        validation_explainability["rank_pct"] <= 0.05,
        validation_explainability["rank_pct"] <= 0.10,
        validation_explainability["rank_pct"] <= 0.20,
    ],
    [
        "Top 1%",
        "Top 1-5%",
        "Top 5-10%",
        "Top 10-20%",
    ],
    default="Bottom 80%",
)


# ============================================================
# SEGMENT FEATURE PROFILES
# ============================================================

print(
    "\nBuilding validation segment profiles..."
)

# IMPORTANT:
# validation_explainability has been sorted by probability.
# Therefore segment labels must be mapped back to the
# original validation rows using row_index.

segment_lookup = (
    validation_explainability[
        ["row_index", "segment"]
    ]
    .set_index("row_index")["segment"]
)

validation_profile = validation[
    locked_features
].copy()

validation_profile["row_index"] = np.arange(
    len(validation_profile)
)

validation_profile["segment"] = (
    validation_profile["row_index"]
    .map(segment_lookup)
)

if validation_profile["segment"].isna().any():
    raise ValueError(
        "Some validation rows could not be assigned "
        "to a prediction segment."
    )

validation_profile = validation_profile.drop(
    columns=["row_index"]
)


# ============================================================
# SEGMENT COUNTS + FEATURE PROFILES
# ============================================================

segment_profiles = (
    validation_profile
    .groupby("segment")[locked_features]
    .mean()
)

segment_counts = (
    validation_profile["segment"]
    .value_counts()
    .rename("row_count")
    .to_frame()
)

segment_profiles = segment_counts.join(
    segment_profiles,
    how="left",
)

segment_order = [
    "Top 1%",
    "Top 1-5%",
    "Top 5-10%",
    "Top 10-20%",
    "Bottom 80%",
]

segment_profiles = segment_profiles.reindex(
    segment_order
)


# ============================================================
# STANDARDIZED SEGMENT PROFILES
# ============================================================

train_means = X_train.mean()

train_stds = (
    X_train
    .std(ddof=0)
    .replace(0, np.nan)
)

standardized_profiles = (
    segment_profiles[locked_features]
    .subtract(
        train_means,
        axis=1,
    )
    .divide(
        train_stds,
        axis=1,
    )
)

standardized_profiles.insert(
    0,
    "row_count",
    segment_profiles["row_count"],
)

standardized_profiles.index.name = "segment"


# ============================================================
# WRITE OUTPUTS
# ============================================================

global_path = (
    OUTPUT_DIR
    / "logistic_global_feature_importance.csv"
)

driver_path = (
    OUTPUT_DIR
    / "logistic_feature_driver_summary.csv"
)

scores_path = (
    OUTPUT_DIR
    / "logistic_validation_explainability_scores.csv"
)

profiles_path = (
    OUTPUT_DIR
    / "logistic_validation_segment_profiles.csv"
)

standardized_path = (
    OUTPUT_DIR
    / "logistic_validation_segment_standardized_profiles.csv"
)

global_importance.to_csv(
    global_path,
    index=False,
)

driver_summary.to_csv(
    driver_path,
    index=False,
)

validation_explainability.to_csv(
    scores_path,
    index=False,
)

segment_profiles.reset_index().to_csv(
    profiles_path,
    index=False,
)

standardized_profiles.reset_index().to_csv(
    standardized_path,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXPLAINABILITY COMPLETE")
print("=" * 70)

print("\nCanonical model used:")

print(
    f"  Model version: "
    f"{metadata.get('model_version')}"
)

print(
    f"  Model: "
    f"{classifier.__class__.__name__}"
)

print("  Retrained: NO")
print("  Recalibrated: NO")
print("  Test data used: NO")

print("\nOutputs written:")

print(f"  {global_path}")
print(f"  {driver_path}")
print(f"  {scores_path}")
print(f"  {profiles_path}")
print(f"  {standardized_path}")

print("\nTop global drivers:")

print(
    global_importance[
        [
            "rank",
            "feature",
            "coefficient",
            "odds_ratio",
            "direction",
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nValidation segment counts:")

print(
    validation_explainability[
        "segment"
    ]
    .value_counts()
    .reindex(segment_order)
    .to_string()
)

print("\nDone.")