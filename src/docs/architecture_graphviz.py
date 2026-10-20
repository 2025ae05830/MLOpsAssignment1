"""Render the end-to-end MLOps architecture diagram with Graphviz.

Designed to be readable by a non-technical reviewer: every arrow is labelled
with a plain-English description of what moves between the stages.

Requires the Graphviz `dot` binary (`sudo apt install graphviz`).

Usage:
    python -m src.docs.architecture_graphviz
"""
from itertools import pairwise
from pathlib import Path

from graphviz import Digraph

OUT_BASE = "reports/figures/architecture"


def build() -> Digraph:
    g = Digraph("mlops", format="png")
    g.attr(rankdir="TB", splines="spline", nodesep="0.5", ranksep="0.9",
           bgcolor="white", fontname="Helvetica", compound="true",
           size="6.7,10.1", ratio="compress")
    g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica",
           fontsize="11", penwidth="1.3", margin="0.2,0.14")
    g.attr("edge", color="#4a5568", penwidth="1.6", arrowsize="0.9",
           fontname="Helvetica", fontsize="9", fontcolor="#1f2937")

    # ---------------- Model Development ----------------
    with g.subgraph(name="cluster_training") as c:
        c.attr(label="Model Development  (done once, on the data scientist's machine)",
               style="dashed,rounded", color="#94a3b8",
               fontname="Helvetica-Bold", fontsize="12", fontcolor="#334155")
        c.node("data", "1. Data Acquisition\nUCI Heart Disease\n303 patients x 14 features",
               fillcolor="#dbeafe", color="#2563eb")
        c.node("prep", "2. Preprocessing\nfill missing values,\nencode categories, clean target",
               fillcolor="#dbeafe", color="#2563eb")
        c.node("train", "3. Model Training\nLogistic Regression / Random Forest\n+ GridSearchCV tuning",
               fillcolor="#dbeafe", color="#2563eb")
        c.node("track", "4. MLflow Tracking\nsaves every experiment:\nparameters, scores, plots",
               fillcolor="#fef3c7", color="#d97706", shape="box3d")

        c.edge("data", "prep",
               label="raw patient records\n(303 rows, CSV)")
        c.edge("prep", "train",
               label="cleaned & encoded\ntable ready for learning")
        c.edge("train", "track",
               label="every run is recorded\n(params, scores, charts)", style="dashed")

    # ---------------- Packaging ----------------
    g.node("export", "5. Model Export\nthe single best model\n(saved as one reusable file)",
           fillcolor="#dcfce7", color="#16a34a")
    g.edge("train", "export",
           label="best-scoring model\n(ROC-AUC 0.966) is chosen")

    # ---------------- Serving ----------------
    g.node("api", "6. FastAPI Service\nweb endpoint that answers\n/predict, /health, /metrics",
           fillcolor="#ede9fe", color="#7c3aed")
    g.node("docker", "7. Docker Image\n'heart-api' - the app bundled\nwith its exact environment",
           fillcolor="#fde68a", color="#ca8a04", shape="box3d")
    g.node("k8s", "8. Kubernetes (Minikube)\nruns 2 copies for reliability,\nplus a public web address",
           fillcolor="#fee2e2", color="#dc2626")

    g.edge("export", "api",
           label="saved model is loaded\nand served to users")
    g.edge("api", "docker",
           label="app is packaged into\na portable container")
    g.edge("docker", "k8s",
           label="container is deployed\nand load-balanced")

    # ---------------- Monitoring ----------------
    g.node("prom", "9. Prometheus\ncollects live numbers\n(how many requests, how fast)",
           fillcolor="#e0f2fe", color="#0284c7")
    g.node("grafana", "10. Grafana\nturns those numbers\ninto readable charts",
           fillcolor="#e0f2fe", color="#0284c7")
    g.edge("api", "prom",
           label="API reports its own stats\nevery 5 seconds", style="dashed")
    g.edge("prom", "grafana",
           label="numbers displayed\nas live dashboards")

    # ---------------- CI/CD ----------------
    g.node("ci", "CI/CD - GitHub Actions\nruns automatically on every code change:\n"
                 "lint (ruff) -> unit tests (pytest) -> retrain model -> save results",
           fillcolor="#f3f4f6", color="#6b7280", shape="box")
    g.edge("ci", "train",
           label="checks the code and\nre-trains automatically", style="dotted",
           constraint="false", color="#6b7280")

    # Align monitoring row
    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("prom")
        s.node("grafana")

    # ---------------- Legend ----------------
    # Placed in its own rank so all entries stack vertically in one column.
    with g.subgraph(name="cluster_legend") as lg:
        lg.attr(label="Legend - what the colours mean", style="dashed,rounded",
                color="#94a3b8", fontname="Helvetica-Bold", fontsize="11",
                fontcolor="#334155", rank="sink")

        # Chain the entries with invisible edges => guaranteed single vertical column
        entries = [
            ("lg1", "Blue  -  data preparation and model training", "#dbeafe", "#2563eb"),
            ("lg2", "Amber  -  experiment tracking and app packaging", "#fef3c7", "#d97706"),
            ("lg3", "Green  -  the finished model, ready to use", "#dcfce7", "#16a34a"),
            ("lg4", "Purple  -  the live web service", "#ede9fe", "#7c3aed"),
            ("lg5", "Red  -  the running deployment", "#fee2e2", "#dc2626"),
            ("lg6", "Light blue  -  monitoring and dashboards", "#e0f2fe", "#0284c7"),
            ("lg7", "Grey  -  automation (CI/CD) checks", "#f3f4f6", "#6b7280"),
            ("lg8", "Solid arrow = data/model moves forward", "#ffffff", "#cbd5e1"),
            ("lg9", "Dashed arrow = metrics or automatic checks", "#ffffff", "#cbd5e1"),
        ]
        for nid, label, fc, ec in entries:
            lg.node(nid, label, fillcolor=fc, color=ec, fontsize="9",
                    margin="0.12,0.06", width="2.7", fixedsize="shape")
        for (a, *_), (b, *_) in pairwise(entries):
            lg.edge(a, b, style="invis")

    return g


def main() -> None:
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    g = build()
    g.render(OUT_BASE, cleanup=True)          # PNG for the report
    g.format = "svg"
    g.render(OUT_BASE, cleanup=True)          # vector copy
    print(f"Saved {OUT_BASE}.png and {OUT_BASE}.svg")


if __name__ == "__main__":
    main()
