import pandas as pd

from src.models.train import build_model, build_preprocessor, evaluate


def _toy_df():
    return pd.DataFrame([
        {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
         "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
         "ca": 0, "thal": 6, "target": 0},
        {"age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 0,
         "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2,
         "ca": 3, "thal": 3, "target": 1},
        {"age": 37, "sex": 1, "cp": 3, "trestbps": 130, "chol": 250, "fbs": 0,
         "restecg": 0, "thalach": 187, "exang": 0, "oldpeak": 3.5, "slope": 3,
         "ca": 0, "thal": 3, "target": 0},
        {"age": 41, "sex": 0, "cp": 2, "trestbps": 130, "chol": 204, "fbs": 0,
         "restecg": 2, "thalach": 172, "exang": 0, "oldpeak": 1.4, "slope": 1,
         "ca": 0, "thal": 3, "target": 0},
    ] * 5)


def test_preprocessor_transforms():
    df = _toy_df()
    X = df.drop(columns=["target"])
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == len(X)
    assert Xt.shape[1] > X.shape[1]  # one-hot expands categoricals


def test_build_model_pipeline():
    model = build_model("logistic_regression")
    assert "preprocess" in model.named_steps
    assert "clf" in model.named_steps


def test_build_model_invalid_name():
    try:
        build_model("not_a_model")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_evaluate_returns_expected_metrics():
    df = _toy_df()
    X, y = df.drop(columns=["target"]), df["target"]
    model = build_model("random_forest")
    model.fit(X, y)
    metrics = evaluate(model, X, y)
    for key in ["accuracy", "precision", "recall", "roc_auc"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0
