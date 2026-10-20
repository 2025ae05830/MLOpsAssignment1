"""Clean and preprocess the Heart Disease dataset.

- Binarize target (0 = no disease, 1 = disease present)
- Handle missing values (NaNs from '?' in the raw data)
- Encode categorical features
- Save processed dataset to data/processed/

Usage:
    python -m src.data.preprocess
"""
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/heart_cleveland.csv")
OUT_PATH = Path("data/processed/heart_clean.csv")

CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, na_values="?")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Binarize target: 0 -> no disease, 1-4 -> disease
    df["target"] = (df["target"] > 0).astype(int)

    # Drop rows with missing target (shouldn't be any, but be safe)
    df = df.dropna(subset=["target"])

    # Fill numeric missing with median
    for col in NUMERIC:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing with mode
    for col in CATEGORICAL:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def main() -> pd.DataFrame:
    df = clean(load_raw())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Cleaned dataset: {df.shape} -> {OUT_PATH}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    return df


if __name__ == "__main__":
    main()
