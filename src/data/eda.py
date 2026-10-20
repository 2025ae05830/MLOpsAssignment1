"""Generate EDA visualizations for the Heart Disease dataset.

Saves figures to reports/figures/.

Usage:
    python -m src.data.eda
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from src.data.preprocess import clean, load_raw

FIG_DIR = Path("reports/figures")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = clean(load_raw())

    # 1. Target / class balance
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="target", data=df, ax=ax, hue="target", legend=False, palette="Set2")
    ax.set_title("Class Balance (0 = No Disease, 1 = Disease)")
    ax.set_xlabel("Target")
    fig.savefig(FIG_DIR / "class_balance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Histograms of numeric features
    numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(20, 4))
    for ax, col in zip(axes, numeric_cols):
        sns.histplot(df[col], kde=True, ax=ax, color="steelblue")
        ax.set_title(col)
    fig.suptitle("Distributions of Numeric Features")
    fig.savefig(FIG_DIR / "numeric_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Age distribution by target
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="age", hue="target", kde=True, multiple="stack", ax=ax)
    ax.set_title("Age Distribution by Target")
    fig.savefig(FIG_DIR / "age_by_target.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"EDA figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
