# MLOps Assignment 1: Heart Disease Prediction

End-to-end MLOps pipeline for predicting heart disease from patient health
data, following production best practices:

```
Data acquisition -> EDA -> model development -> experiment tracking (MLflow)
-> packaging -> automated testing & CI/CD -> containerization -> Kubernetes
deployment -> monitoring
```

**Dataset:** [UCI Heart Disease](https://archive.ics.uci.edu/ml/datasets/heart+disease)
(Cleveland subset, 303 records x 14 features, binary target).

**Final model:** Logistic Regression (tuned, `C=1.0`) — test **ROC-AUC 0.966**,
accuracy 0.885, recall 0.929. Served as a JSON API returning a prediction and
its confidence.

---

## Repository structure

```
.
|-- src/
|   |-- data/       download_data.py, preprocess.py, eda.py
|   |-- models/     train.py, export_model.py, predict.py
|   |-- api/        main.py          (FastAPI serving app)
|   `-- docs/       architecture.py, architecture_graphviz.py  (diagram generators)
|-- tests/          test_preprocess.py, test_train.py   (pytest)
|-- notebooks/      01_EDA.ipynb
|-- deployment/     deployment.yaml, service.yaml, ingress.yaml
|-- configs/        model_results.json
|-- data/processed/ heart_clean.csv                      (cleaned dataset)
|-- models/best_model/                                   (exported model artifact)
|-- reports/figures/                                     (EDA / evaluation figures)
|-- screenshots/                                         (evidence screenshots)
|-- mlflow.db                                            (experiment tracking store)
|-- .github/workflows/ci.yml                             (CI/CD pipeline)
|-- Dockerfile, docker-compose.yml, prometheus.yml
`-- requirements.txt
```

## 1. Setup

Requires Python 3.12+.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The Graphviz `dot` binary is optional (only for regenerating the architecture
diagram): `sudo apt install graphviz`.

## 2. Data

The raw dataset is downloaded by script; the cleaned dataset is committed.

```bash
python -m src.data.download_data    # -> data/raw/heart_cleveland.csv
python -m src.data.preprocess       # -> data/processed/heart_clean.csv
python -m src.data.eda              # -> reports/figures/*.png
```

Raw `target` is 0-4 (0 = no disease, 1-4 = disease); preprocessing binarizes it.

## 3. Train and track experiments

```bash
python -m src.models.train          # baseline + GridSearchCV runs -> MLflow
python -m src.models.export_model   # best run -> models/best_model/
python -m src.models.predict        # sample inference

# Browse experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://localhost:5000
```

Each session logs **four MLflow runs** (`{model}_baseline`, `{model}_tuned`) with
parameters, metrics, confusion-matrix/ROC plots, and the full model pipeline.

## 4. Tests and linting

```bash
ruff check src tests     # linting
pytest tests/ -v         # 8 unit tests
```

## 5. Serve the model (local)

```bash
# Option A: uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Option B: Docker
docker build -t heart-api .
docker run -p 8000:8000 heart-api

# Smoke test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

Response:

```json
{"prediction": 0, "confidence": 0.16553481726226757}
```

Interactive API docs: http://localhost:8000/docs

## 6. Deploy to Kubernetes (Minikube)

```bash
minikube start --driver=docker
minikube image load heart-api
kubectl apply -f deployment/deployment.yaml \
              -f deployment/service.yaml \
              -f deployment/ingress.yaml
kubectl get pods                    # 2/2 Running
minikube service heart-api --url    # accessible API URL
```

The Deployment runs 2 replicas with a `/health` readiness probe; the Service is
a LoadBalancer and an Ingress provides a single entry point.

## 7. Monitoring

The API exposes Prometheus metrics at `GET /metrics`
(`api_requests_total`, `api_request_latency_seconds`) and logs every request.

```bash
docker compose up -d     # Prometheus :9090, Grafana :3000 (admin/admin)
```

In Grafana, add a Prometheus data source at `http://prometheus:9090` and plot
`rate(api_requests_total[1m])` and
`rate(api_request_latency_seconds_sum[1m]) / rate(api_request_latency_seconds_count[1m])`.

## 8. CI/CD

`.github/workflows/ci.yml` runs on every push / pull request:

1. Install dependencies
2. Lint (`ruff check src tests`)
3. Unit tests (`pytest tests/ -v`)
4. Train the model
5. Upload figures and `configs/model_results.json` as build artifacts

The pipeline fails fast on any lint, test, or training error.

## API reference

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/health`  | Liveness check |
| POST | `/predict` | JSON features -> `{prediction, confidence}` |
| GET  | `/metrics` | Prometheus metrics |
| GET  | `/docs`    | Interactive OpenAPI documentation |

### Input features

`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`

## Reproducibility

Every step runs from a clean environment via `requirements.txt`. Random seeds
are fixed (`random_state=42`) and the exported model bundles its preprocessing
pipeline, so training and serving are guaranteed to match.
