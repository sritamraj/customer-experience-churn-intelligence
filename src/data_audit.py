"""Basic data-quality audit.
Update DATA_PATH for your selected public dataset.
"""
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/data.csv")

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Put your public dataset in data/raw/ "
            "and update DATA_PATH if necessary."
        )

    df = pd.read_csv(DATA_PATH)

    print("\n=== SHAPE ===")
    print(df.shape)

    print("\n=== DATA TYPES ===")
    print(df.dtypes)

    print("\n=== MISSING VALUES ===")
    missing = df.isna().sum().sort_values(ascending=False)
    print(pd.DataFrame({
        "missing_count": missing,
        "missing_pct": (missing / len(df) * 100).round(2)
    }).head(30))

    print("\n=== DUPLICATES ===")
    print("Duplicate rows:", df.duplicated().sum())

    print("\n=== NUMERIC SUMMARY ===")
    print(df.describe(include="all").transpose().head(30))

if __name__ == "__main__":
    main()
