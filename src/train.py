"""Leakage-safe modeling template.

Before using:
1. Create a true prediction-time customer table.
2. Create the churn target from a FUTURE outcome window.
3. Remove identifiers and any future-derived columns.
4. Adjust feature columns below.
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

DATA_PATH = Path("data/processed/modeling_table.csv")
MODEL_DIR = Path("models")

TARGET = "churn"
ID_COLUMNS = ["customer_id"]

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["precision", "recall", "f1", "roc_auc", "average_precision"]
    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    print(f"\n{name} CV results:")
    for metric in scoring:
        key = f"test_{metric}"
        print(f"{metric}: {cv_results[key].mean():.4f} ± {cv_results[key].std():.4f}")

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print(f"\n{name} TEST:")
    print(classification_report(y_test, pred, digits=4))
    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 4))
    print("PR-AUC:", round(average_precision_score(y_test, proba), 4))
    return model

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Create data/processed/modeling_table.csv with a leakage-safe "
            "prediction-time feature table and a churn target first."
        )

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError("Target column 'churn' is missing.")

    drop_cols = [c for c in ID_COLUMNS + [TARGET] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET].astype(int)

    # Starter assumption: all remaining columns are numeric.
    # Encode categorical variables properly if your dataset contains them.
    X = X.select_dtypes(include=["number"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1
            ))
        ]),
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=42
            ))
        ])
    }

    MODEL_DIR.mkdir(exist_ok=True)
    for name, model in models.items():
        fitted = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(fitted, MODEL_DIR / filename)

if __name__ == "__main__":
    main()
