"""Train and evaluate classification models for heart disease prediction.

Trains Logistic Regression and Random Forest with a preprocessing pipeline
(scaling + one-hot encoding), evaluates via cross-validation and a held-out
test set, and selects the best model.

Usage:
    python -m src.models.train
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/processed/heart_clean.csv")

CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET = "target"


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL),
    ])


def build_model(name: str) -> Pipeline:
    if name == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    elif name == "random_forest":
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("preprocess", build_preprocessor()), ("clf", clf)])


PARAM_GRIDS = {
    "logistic_regression": {"clf__C": [0.1, 1.0, 10.0]},
    "random_forest": {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [None, 6],
    },
}


def evaluate(model: Pipeline, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def make_plots(model: Pipeline, X_test, y_test, name: str) -> list[str]:
    """Create confusion matrix and ROC curve plots; return file paths."""
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    paths = []

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix — {name}")
    p = f"reports/figures/cm_{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fpr, tpr, _ = roc_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, proba):.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title(f"ROC Curve — {name}")
    ax.legend()
    p = f"reports/figures/roc_{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    return paths


def main() -> dict:
    df = pd.read_csv(DATA_PATH)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Default to a repo-relative SQLite store so the pipeline is portable
    # (local runs, CI runners, containers). Override with MLFLOW_TRACKING_URI.
    repo_root = Path(__file__).resolve().parents[2]
    default_db = (repo_root / "mlflow.db").as_uri()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", default_db))

    # Pin the experiment's artifact location to a repo-relative path. Without
    # this, an existing experiment row (e.g. from a committed mlflow.db) would
    # keep its original absolute artifact_location and break on another machine
    # or CI runner.
    experiment = mlflow.get_experiment_by_name("heart_disease")
    if experiment is None:
        mlflow.create_experiment(
            "heart_disease",
            artifact_location=(repo_root / "mlruns").as_uri(),
        )
    mlflow.set_experiment("heart_disease")

    for name in ["logistic_regression", "random_forest"]:
        # Baseline run (default hyperparameters)
        with mlflow.start_run(run_name=f"{name}_baseline"):
            model = build_model(name)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
            model.fit(X_train, y_train)
            baseline_metrics = evaluate(model, X_test, y_test)
            baseline_metrics["cv_roc_auc_mean"] = cv_scores.mean()
            baseline_metrics["cv_roc_auc_std"] = cv_scores.std()

            mlflow.log_param("model_type", name)
            mlflow.log_param("tuning", "none")
            for k, v in baseline_metrics.items():
                mlflow.log_metric(k, v)
            mlflow.sklearn.log_model(model, artifact_path="model",
                                     skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"])

        # Tuned run (GridSearchCV over a small grid, scored on ROC-AUC)
        with mlflow.start_run(run_name=f"{name}_tuned"):
            search = GridSearchCV(
                build_model(name),
                PARAM_GRIDS[name],
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
            metrics = evaluate(model, X_test, y_test)
            metrics["cv_roc_auc_mean"] = cv_scores.mean()
            metrics["cv_roc_auc_std"] = cv_scores.std()
            metrics["baseline_roc_auc"] = baseline_metrics["roc_auc"]
            results[name] = {**metrics, "best_params": search.best_params_}

            mlflow.log_param("model_type", name)
            mlflow.log_param("tuning", "grid_search_cv")
            for k, v in search.best_params_.items():
                mlflow.log_param(k, v)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            for plot_path in make_plots(model, X_test, y_test, name):
                mlflow.log_artifact(plot_path)
            mlflow.sklearn.log_model(model, artifact_path="model",
                                     skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"])

            print(f"\n{name} (tuned: {search.best_params_}):")
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v:.3f}")

    best = max(results, key=lambda n: results[n]["roc_auc"])
    print(f"\nBest model by test ROC-AUC: {best}")

    Path("configs").mkdir(exist_ok=True)
    with open("configs/model_results.json", "w") as f:
        json.dump({"results": results, "best_model": best}, f, indent=2)
    return results


if __name__ == "__main__":
    main()
