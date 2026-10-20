"""Render a project architecture diagram for the report.

Usage:
    python -m src.docs.architecture
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = "reports/figures/architecture.png"

STAGE = {"boxstyle": "round,pad=0.4", "linewidth": 1.2}


def box(ax, x, y, w, h, text, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=fc, edgecolor=ec, **STAGE))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.6, color="#4a5568"))


def main():
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    # Data / training row
    box(ax, 0.3, 7.0, 2.0, 0.9, "1. Data Acquisition\n(download_data.py)", "#dbeafe", "#2563eb")
    box(ax, 3.0, 7.0, 2.0, 0.9, "2. Preprocessing\n(preprocess.py)", "#dbeafe", "#2563eb")
    box(ax, 5.7, 7.0, 2.4, 0.9, "3. Model Training\nLR / RF + GridSearchCV", "#dbeafe", "#2563eb")
    box(ax, 8.7, 7.0, 2.0, 0.9, "4. MLflow Tracking\nparams, metrics, plots", "#fef3c7", "#d97706")

    arrow(ax, 2.3, 7.45, 3.0, 7.45)
    arrow(ax, 5.0, 7.45, 5.7, 7.45)
    arrow(ax, 8.1, 7.45, 8.7, 7.45)

    # Packaging row
    box(ax, 5.7, 5.2, 2.4, 0.9, "5. Model Export\n(best_model pipeline)", "#dcfce7", "#16a34a")
    arrow(ax, 6.9, 7.0, 6.9, 6.1)

    # Serving row
    box(ax, 5.7, 3.4, 2.4, 0.9, "6. FastAPI Service\n/predict  /health  /metrics", "#ede9fe", "#7c3aed")
    arrow(ax, 6.9, 5.2, 6.9, 4.3)

    box(ax, 3.0, 3.4, 2.0, 0.9, "7. Docker Image\nheart-api", "#fde68a", "#ca8a04")
    arrow(ax, 5.0, 3.85, 5.7, 3.85)

    box(ax, 8.4, 3.4, 2.3, 0.9, "8. Kubernetes (Minikube)\nDeployment + Service\n+ Ingress", "#fee2e2", "#dc2626")
    arrow(ax, 8.1, 3.85, 8.4, 3.85)

    # Monitoring row
    box(ax, 3.0, 1.6, 2.0, 0.9, "9. Prometheus\nscrapes /metrics", "#e0f2fe", "#0284c7")
    box(ax, 0.3, 1.6, 2.0, 0.9, "10. Grafana\ndashboards", "#e0f2fe", "#0284c7")
    arrow(ax, 5.0, 2.05, 5.7, 3.4)   # metrics from API
    arrow(ax, 3.0, 2.05, 2.3, 2.05)  # prometheus -> grafana

    # CI/CD band
    box(ax, 0.3, 0.3, 10.4, 0.75,
        "CI/CD (GitHub Actions):  lint (ruff)  ->  unit tests (pytest)  ->  model training  ->  artifact upload",
        "#f3f4f6", "#6b7280")
    arrow(ax, 5.5, 1.6, 5.5, 1.05)

    ax.set_title("Heart Disease Prediction — End-to-End MLOps Architecture",
                 fontsize=14, fontweight="bold", pad=14)

    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
