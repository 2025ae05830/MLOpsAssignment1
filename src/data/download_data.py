"""Download the Heart Disease dataset (UCI Cleveland subset).

Usage:
    python -m src.data.download_data
"""
from pathlib import Path

import pandas as pd
import requests

RAW_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

RAW_PATH = Path("data/raw/heart_cleveland.csv")


def download() -> Path:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(RAW_URL, timeout=60)
    resp.raise_for_status()
    RAW_PATH.write_bytes(resp.content)

    df = pd.read_csv(RAW_PATH, names=COLUMNS, na_values="?")
    # Task 1 works from a raw copy; store a tidy raw CSV with headers
    df.to_csv(RAW_PATH, index=False)
    print(f"Saved {len(df)} rows to {RAW_PATH}")
    return RAW_PATH


if __name__ == "__main__":
    download()
