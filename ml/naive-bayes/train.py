"""Gaussian naive Bayes: bank customer churn on synthetic data.

Pipeline: generate data -> train sklearn pipeline -> evaluate -> export ONNX -> verify with
onnxruntime -> write the artefacts the ASP.NET site consumes:

  <web>/App_Data/models/naive-bayes/{model.onnx, schema.json}
  <web>/wwwroot/projects/naive-bayes/charts/*.png
  <tests>/fixtures/naive-bayes.json   (reference inputs/outputs for the .NET parity test)

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
from sklearn.naive_bayes import GaussianNB

SLUG = "naive-bayes"
MODEL_VERSION = "1.0.0"
SEED = 42
LABELS = ["Stays", "Churns"]  # index = class id; probability[1] is P(churn)
ONNX_TOLERANCE = 1e-3
THUMBNAIL_SOURCE = "feature-distributions.png"

NUMERIC_INPUTS = [
    {"name": "age", "label": "Age", "min": 18, "max": 90, "step": 1, "default": 40},
    {"name": "balance_k", "label": "Account balance (thousands)", "min": 0, "max": 250, "step": 1, "default": 40},
    {"name": "products", "label": "Products held", "min": 1, "max": 4, "step": 1, "default": 2,
     "help": "Accounts, cards, loans, deposits"},
    {"name": "tenure_years", "label": "Years as a customer", "min": 0, "max": 30, "step": 1, "default": 5},
]
SELECT_INPUTS = [
    {"name": "active_member", "label": "Active in the last 3 months", "options": ["yes", "no"], "default": "yes"},
]
PRESETS = [
    {"name": "Loyal customer", "values": {"age": 45, "balance_k": 60, "products": 3, "tenure_years": 14,
                                          "active_member": "yes"}},
    {"name": "At risk", "values": {"age": 58, "balance_k": 140, "products": 1, "tenure_years": 1,
                                   "active_member": "no"}},
]


def make_data(n: int = 6000, seed: int = SEED) -> pd.DataFrame:
    """Synthetic customers; inactivity, a single product and short tenure raise churn risk."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(42, 13, n), 18, 90).round()
    balance = np.clip(rng.gamma(2.0, 28.0, n), 0, 250).round()
    products = np.clip(rng.choice([1, 2, 3, 4], size=n, p=[0.42, 0.42, 0.12, 0.04]), 1, 4)
    tenure = np.clip(rng.gamma(2.2, 3.0, n), 0, 30).round()
    active = rng.choice(["yes", "no"], size=n, p=[0.6, 0.4])

    z = (-1.5 + 0.06 * (age - 42) + 0.008 * (balance - 56) - 0.8 * (products - 1.8) - 0.12 * (tenure - 6)
         + 1.6 * (active == "no"))
    churn = rng.binomial(1, 1 / (1 + np.exp(-z)))
    return pd.DataFrame({"age": age, "balance_k": balance, "products": products, "tenure_years": tenure,
                         "active_member": active, "target": churn})


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC_COLUMNS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
    ])
    return Pipeline([("pre", pre), ("clf", GaussianNB())])


def extra_charts(result, save) -> None:
    proba = result.pipeline.predict_proba(result.x_test)[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, (col, title) in zip(axes, [("age", "Age"), ("tenure_years", "Years as a customer"),
                                       ("balance_k", "Balance (thousands)")]):
        lo, hi = result.x_train[col].min(), result.x_train[col].max()
        bins = np.linspace(lo, hi, 25)
        for cls, colour in [(0, "#78716c"), (1, ACCENT)]:
            ax.hist(result.x_train[col][result.y_train == cls], bins=bins, density=True, histtype="stepfilled",
                    alpha=0.45, edgecolor=colour, linewidth=1.6, color=colour, label=LABELS[cls])
        ax.set_title(title, fontsize=11)
        ax.set_yticks([])
    axes[0].legend()
    save(fig, "feature-distributions.png",
         "Naive Bayes models each feature's distribution per class; these are the shapes it compares.")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    fpr, tpr, _ = roc_curve(result.y_test, proba)
    ax.plot(fpr, tpr, color=ACCENT, linewidth=2, label="Naive Bayes")
    ax.plot([0, 1], [0, 1], "--", color=MUTED, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curve (AUC {result.metrics['roc_auc']:.2f})")
    ax.legend(loc="lower right")
    save(fig, "roc-curve.png", "ROC curve on held-out customers.")

    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(proba, bins) - 1, 0, 9)
    mean_pred = np.array([proba[idx == b].mean() for b in range(10) if (idx == b).any()])
    frac_pos = np.array([result.y_test[idx == b].mean() for b in range(10) if (idx == b).any()])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color=MUTED, label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", color=ACCENT, label="Model")
    ax.set_xlabel("Predicted probability of churn")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration")
    ax.legend()
    save(fig, "calibration.png", "Naive Bayes is often over-confident: probabilities drift from observed rates.")

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
