"""Support vector machine: card-transaction fraud classifier on synthetic data.

Pipeline: generate data -> train sklearn pipeline -> evaluate -> export ONNX -> verify with
onnxruntime -> write the artefacts the ASP.NET site consumes:

  <web>/App_Data/models/svm/{model.onnx, schema.json}
  <web>/wwwroot/projects/svm/charts/*.png
  <tests>/fixtures/svm.json   (reference inputs/outputs for the .NET parity test)

Run:  uv run python train.py
This project is self-contained: it imports nothing from the web app or other projects.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from sklearn.compose import ColumnTransformer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

SLUG = "svm"
MODEL_VERSION = "1.0.0"
SEED = 42
LABELS = ["Legitimate", "Fraud"]  # index = class id; probability[1] is P(fraud)
ONNX_TOLERANCE = 2e-3   # SVC probabilities come from Platt scaling; float32 ONNX differs slightly
THUMBNAIL_SOURCE = "decision-boundary.png"

NUMERIC_INPUTS = [
    {"name": "amount", "label": "Transaction amount", "min": 1, "max": 5000, "step": 1, "default": 80},
    {"name": "hour_of_day", "label": "Hour of day (0-23)", "min": 0, "max": 23, "step": 1, "default": 14},
    {"name": "distance_from_home_km", "label": "Distance from home (km)", "min": 0, "max": 3000, "step": 1, "default": 5,
     "help": "Distance between the card's home area and the merchant"},
    {"name": "txns_last_hour", "label": "Transactions in the last hour", "min": 0, "max": 15, "step": 1, "default": 1},
]
SELECT_INPUTS = [
    {"name": "channel", "label": "Channel", "options": ["in-store", "online", "atm"], "default": "in-store"},
]
PRESETS = [
    {"name": "Everyday purchase", "values": {"amount": 42, "hour_of_day": 13, "distance_from_home_km": 3,
                                             "txns_last_hour": 1, "channel": "in-store"}},
    {"name": "Suspicious burst", "values": {"amount": 1900, "hour_of_day": 3, "distance_from_home_km": 1400,
                                            "txns_last_hour": 7, "channel": "online"}},
]


def make_data(n: int = 5000, seed: int = SEED) -> pd.DataFrame:
    """Synthetic transactions; fraud risk rises with amount, night hours, distance, bursts and online use."""
    rng = np.random.default_rng(seed)
    amount = np.clip(rng.lognormal(np.log(60), 1.1, n), 1, 5000).round()
    hour = rng.integers(0, 24, n)
    distance = np.clip(rng.exponential(25, n) + (rng.random(n) < 0.05) * rng.uniform(200, 2500, n), 0, 3000).round()
    burst = np.clip(rng.poisson(1.0, n), 0, 15)
    channel = rng.choice(["in-store", "online", "atm"], size=n, p=[0.55, 0.35, 0.10])

    night = (hour <= 5).astype(float)
    ch = pd.Series(channel).map({"in-store": 0.0, "online": 0.9, "atm": 0.3}).to_numpy()
    z = -4.6 + 1.0 * np.log(amount / 60) + 1.9 * night + 0.9 * np.log1p(distance / 20) + 0.5 * burst + ch
    fraud = rng.binomial(1, 1 / (1 + np.exp(-z)))

    return pd.DataFrame({
        "amount": amount, "hour_of_day": hour, "distance_from_home_km": distance, "txns_last_hour": burst,
        "channel": channel, "target": fraud,
    })


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLUMNS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
    ])
    return Pipeline([("pre", pre), ("clf", SVC(kernel="rbf", C=2.0, gamma="scale", probability=True,
                                                random_state=SEED))])


def extra_charts(result, save) -> None:
    proba = result.pipeline.predict_proba(result.x_test)[:, 1]

    # Decision surface over amount x distance, other inputs held at typical values.
    base = {c: float(result.x_train[c].median()) for c in NUMERIC_COLUMNS}
    base["channel"] = "online"
    amounts = np.geomspace(1, 1000, 80)   # stay inside the range the model has seen plenty of
    dists = np.linspace(0, 1500, 80)
    aa, dd = np.meshgrid(amounts, dists)
    grid = pd.DataFrame({c: np.full(aa.size, base[c]) for c in ALL_COLUMNS})
    grid["amount"] = aa.ravel()
    grid["distance_from_home_km"] = dd.ravel()
    grid["hour_of_day"] = 3.0
    surface = result.pipeline.predict_proba(grid[ALL_COLUMNS])[:, 1].reshape(aa.shape)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    cs = ax.contourf(aa, dd, surface, levels=np.linspace(0, 1, 11), cmap="Purples")
    ax.contour(aa, dd, surface, levels=[0.5], colors=[ACCENT], linewidths=2)
    ax.set_xscale("log")
    ax.set_xlabel("Transaction amount (log scale)")
    ax.set_ylabel("Distance from home (km)")
    ax.set_title("Fraud probability surface")
    ax.grid(False)
    fig.colorbar(cs, ax=ax, label="P(fraud)")
    save(fig, "decision-boundary.png",
         "Fraud probability across amount and distance for a 3am online purchase; the line marks 50%.")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    fpr, tpr, _ = roc_curve(result.y_test, proba)
    ax.plot(fpr, tpr, color=ACCENT, linewidth=2, label="SVM")
    ax.plot([0, 1], [0, 1], "--", color=MUTED, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curve (AUC {result.metrics['roc_auc']:.2f})")
    ax.legend(loc="lower right")
    save(fig, "roc-curve.png", "ROC curve on held-out transactions.")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bins = np.linspace(0, 1, 21)
    ax.hist(proba[result.y_test == 0], bins=bins, color=ACCENT_LIGHT, label="Legitimate", log=True)
    ax.hist(proba[result.y_test == 1], bins=bins, color=ACCENT, alpha=0.85, label="Fraud", log=True)
    ax.set_xlabel("Predicted probability of fraud")
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title("Score separation")
    ax.legend()
    save(fig, "score-distribution.png", "How cleanly predicted fraud probabilities separate the two groups.")

# ---------------------------------------------------------------------------------------------------
# Common pipeline (identical shape across the classification projects)
# ---------------------------------------------------------------------------------------------------

NUMERIC_COLUMNS = [i["name"] for i in NUMERIC_INPUTS]
CATEGORICAL_COLUMNS = [i["name"] for i in SELECT_INPUTS]
ALL_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
ACCENT = "#4f46e5"
ACCENT_LIGHT = "#a5b4fc"
MUTED = "#78716c"


def to_onnx(pipeline: Pipeline) -> bytes:
    """One ONNX input per column (float [N,1] for numbers, string [N,1] for selects)."""
    initial_types = [(c, FloatTensorType([None, 1])) for c in NUMERIC_COLUMNS] + \
                    [(c, StringTensorType([None, 1])) for c in CATEGORICAL_COLUMNS]
    clf = pipeline.steps[-1][1]
    model = convert_sklearn(pipeline, initial_types=initial_types, options={id(clf): {"zipmap": False}},
                            target_opset={"": 17, "ai.onnx.ml": 3})
    return model.SerializeToString()


def onnx_predict(model_bytes: bytes, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Returns (predicted class ids, class probabilities) from onnxruntime."""
    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    feed = {c: df[c].to_numpy(dtype=np.float32).reshape(-1, 1) for c in NUMERIC_COLUMNS}
    feed |= {c: df[c].to_numpy(dtype=object).reshape(-1, 1) for c in CATEGORICAL_COLUMNS}
    labels, probs = session.run(None, feed)
    return labels, probs


@dataclass
class Result:
    pipeline: Pipeline
    model_bytes: bytes
    x_train: pd.DataFrame
    y_train: np.ndarray
    x_test: pd.DataFrame
    y_test: np.ndarray
    metrics: dict


def train_and_evaluate() -> Result:
    data = make_data()
    x = data[ALL_COLUMNS]
    y = data["target"].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=SEED, stratify=y)

    pipeline = build_pipeline().fit(x_train, y_train)
    proba = pipeline.predict_proba(x_test)
    pred = proba.argmax(axis=1)
    if len(LABELS) == 2:
        roc_auc = float(roc_auc_score(y_test, proba[:, 1]))
    else:
        roc_auc = float(roc_auc_score(y_test, proba, multi_class="ovr"))
    baseline = float(np.bincount(y_train).max() / len(y_train))   # always guess the majority class
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "majority_baseline": baseline,
        "roc_auc": roc_auc,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "class_share": {LABELS[i]: float((y == i).mean()) for i in range(len(LABELS))},
    }

    model_bytes = to_onnx(pipeline)

    # Verify the ONNX model reproduces sklearn on the whole test set before anything is written.
    # The site picks the class by arg-max of the probabilities (SVC's own label can differ near 50%).
    _, onnx_proba = onnx_predict(model_bytes, x_test)
    max_diff = float(np.abs(onnx_proba - proba).max())
    agree = float((onnx_proba.argmax(axis=1) == pred).mean())
    assert max_diff < ONNX_TOLERANCE, f"ONNX and sklearn disagree (max diff {max_diff})"
    assert agree > 0.999, f"ONNX and sklearn predict different classes for {1 - agree:.2%} of rows"
    metrics["onnx_max_abs_diff"] = max_diff
    return Result(pipeline, model_bytes, x_train, y_train, x_test, y_test, metrics)


def build_schema() -> dict:
    inputs = [{"type": "number", **i} for i in NUMERIC_INPUTS] + [{"type": "select", **i} for i in SELECT_INPUTS]
    return {
        "modelFile": "model.onnx",
        "modelVersion": MODEL_VERSION,
        "inputs": inputs,
        "outputs": {"type": "classification", "labels": LABELS},
        "presets": PRESETS,
    }


def build_fixtures(result: Result, n_random: int = 6) -> list[dict]:
    """Reference inputs with sklearn's answers, so the .NET runner can be checked against Python."""
    rng = np.random.default_rng(SEED + 1)
    cases = [dict(p["values"]) for p in PRESETS]
    for _ in range(n_random):
        case = {i["name"]: round(float(rng.uniform(i["min"], i["max"])) / i["step"]) * i["step"]
                for i in NUMERIC_INPUTS}
        for s in SELECT_INPUTS:
            case[s["name"]] = str(rng.choice(s["options"]))
        cases.append(case)

    df = pd.DataFrame(cases)[ALL_COLUMNS]
    proba = result.pipeline.predict_proba(df)
    return [
        {
            "input": {k: (float(v) if k in NUMERIC_COLUMNS else str(v)) for k, v in row.items()},
            "label": LABELS[int(np.argmax(p))],
            "probabilities": {LABELS[i]: float(p[i]) for i in range(len(LABELS))},
        }
        for row, p in zip(df.to_dict("records"), proba)
    ]


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "sans-serif", "font.size": 11, "axes.edgecolor": "#d6d3d1", "axes.labelcolor": "#44403c",
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 13, "axes.grid": True, "grid.color": "#f0efed",
    })


def make_charts(result: Result, out_dir: Path) -> list[dict]:
    """Writes PNG charts and returns the captions for projects.json."""
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    charts: list[dict] = []

    def save(fig, name: str, caption: str) -> None:
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=200)
        plt.close(fig)
        charts.append({"file": name, "caption": caption})

    extra_charts(result, save)

    pred = result.pipeline.predict_proba(result.x_test).argmax(axis=1)
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    ConfusionMatrixDisplay.from_predictions(result.y_test, pred, display_labels=LABELS, ax=ax, cmap="Purples",
                                            colorbar=False)
    ax.grid(False)
    ax.set_title("Confusion matrix")
    save(fig, "confusion-matrix.png", "Confusion matrix on held-out data.")

    shutil.copyfile(out_dir / THUMBNAIL_SOURCE, out_dir / "thumbnail.png")
    return charts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo-root", type=Path, default=default_root,
                        help="repository root containing src/ and tests/ (default: two levels up)")
    args = parser.parse_args()
    web = args.repo_root / "src" / "MLPortfolio.Web"

    result = train_and_evaluate()
    print(json.dumps(result.metrics, indent=2))

    model_dir = web / "App_Data" / "models" / SLUG
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.onnx").write_bytes(result.model_bytes)
    (model_dir / "schema.json").write_text(json.dumps(build_schema(), indent=2) + "\n", encoding="utf-8")

    fixtures_dir = args.repo_root / "tests" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    (fixtures_dir / f"{SLUG}.json").write_text(json.dumps(build_fixtures(result), indent=2) + "\n", encoding="utf-8")

    charts = make_charts(result, web / "wwwroot" / "projects" / SLUG / "charts")
    print("charts (paste into projects.json):")
    print(json.dumps(charts, indent=2))
    print(f"wrote model, schema, fixtures and charts for '{SLUG}'")


if __name__ == "__main__":
    main()
