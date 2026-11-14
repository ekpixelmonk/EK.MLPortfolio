"""Decision tree: loan approval on synthetic applications.

Pipeline: generate data -> train sklearn pipeline -> evaluate -> export ONNX -> verify with
onnxruntime -> write the artefacts the ASP.NET site consumes:

  <web>/App_Data/models/decision-tree/{model.onnx, schema.json}
  <web>/wwwroot/projects/decision-tree/charts/*.png
  <tests>/fixtures/decision-tree.json   (reference inputs/outputs for the .NET parity test)

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
from sklearn.tree import DecisionTreeClassifier, plot_tree

SLUG = "decision-tree"
MODEL_VERSION = "1.0.0"
SEED = 42
LABELS = ["Rejected", "Approved"]  # index = class id
ONNX_TOLERANCE = 1e-4
THUMBNAIL_SOURCE = "depth-vs-accuracy.png"
MAX_DEPTH = 4

NUMERIC_INPUTS = [
    {"name": "credit_score", "label": "Credit score", "min": 300, "max": 850, "step": 1, "default": 680,
     "help": "A 300-850 credit score"},
    {"name": "annual_income_k", "label": "Annual income (thousands)", "min": 15, "max": 250, "step": 1, "default": 70},
    {"name": "loan_to_income_pct", "label": "Loan as % of annual income", "min": 5, "max": 120, "step": 1, "default": 40,
     "help": "Requested loan amount divided by yearly income"},
    {"name": "employment_years", "label": "Years in current job", "min": 0, "max": 30, "step": 1, "default": 4},
]
SELECT_INPUTS = [
    {"name": "collateral", "label": "Collateral offered", "options": ["none", "vehicle", "property"], "default": "none"},
]
PRESETS = [
    {"name": "Strong applicant", "values": {"credit_score": 780, "annual_income_k": 120, "loan_to_income_pct": 20,
                                            "employment_years": 12, "collateral": "property"}},
    {"name": "Weak applicant", "values": {"credit_score": 560, "annual_income_k": 40, "loan_to_income_pct": 85,
                                          "employment_years": 1, "collateral": "none"}},
    {"name": "Borderline", "values": {"credit_score": 640, "annual_income_k": 60, "loan_to_income_pct": 45,
                                      "employment_years": 3, "collateral": "vehicle"}},
]


def make_data(n: int = 5000, seed: int = SEED) -> pd.DataFrame:
    """Synthetic applications. A banker's rule of thumb decides, then 6% of decisions are flipped as noise."""
    rng = np.random.default_rng(seed)
    score = np.clip(rng.normal(660, 85, n), 300, 850).round()
    income = np.clip(rng.lognormal(np.log(65), 0.5, n), 15, 250).round()
    lti = np.clip(rng.gamma(3.0, 15.0, n), 5, 120).round()
    years = np.clip(rng.gamma(2.0, 3.0, n), 0, 30).round()
    collateral = rng.choice(["none", "vehicle", "property"], size=n, p=[0.5, 0.25, 0.25])

    approve = ((score >= 650) & (lti <= 50)) \
        | ((score >= 600) & (lti <= 35) & (collateral == "property")) \
        | ((score >= 700) & (years >= 2) & (lti <= 75))
    flip = rng.random(n) < 0.06
    approve = np.where(flip, ~approve, approve)

    return pd.DataFrame({
        "credit_score": score, "annual_income_k": income, "loan_to_income_pct": lti, "employment_years": years,
        "collateral": collateral, "target": approve.astype(int),
    })


def build_pipeline(max_depth: int = MAX_DEPTH) -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC_COLUMNS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
    ])
    return Pipeline([("pre", pre), ("clf", DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=20,
                                                                  random_state=SEED))])


def extra_charts(result, save) -> None:
    names = [str(n).split("__")[-1] for n in result.pipeline.named_steps["pre"].get_feature_names_out()]
    tree = result.pipeline.named_steps["clf"]

    fig, ax = plt.subplots(figsize=(15, 7))
    plot_tree(tree, feature_names=names, class_names=LABELS, filled=True, rounded=True, impurity=False,
              proportion=True, max_depth=2, fontsize=10, ax=ax)
    save(fig, "tree.png", "The learned tree (top two levels): each box is a yes/no question about the applicant.")

    order = np.argsort(tree.feature_importances_)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([names[i] for i in order], tree.feature_importances_[order], color=ACCENT)
    ax.set_title("Feature importance")
    ax.set_xlabel("Share of impurity reduction")
    save(fig, "feature-importance.png", "Which questions the tree relies on most when deciding.")

    depths = list(range(1, 15))
    train_acc, test_acc = [], []
    for d in depths:
        m = build_pipeline(d).fit(result.x_train, result.y_train)
        train_acc.append(accuracy_score(result.y_train, m.predict(result.x_train)))
        test_acc.append(accuracy_score(result.y_test, m.predict(result.x_test)))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(depths, train_acc, "o-", color=MUTED, label="Training")
    ax.plot(depths, test_acc, "o-", color=ACCENT, label="Held-out")
    ax.axvline(MAX_DEPTH, color=ACCENT_LIGHT, linestyle="--", label=f"Chosen depth ({MAX_DEPTH})")
    ax.set_xlabel("Maximum tree depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Depth vs accuracy")
    ax.legend()
    save(fig, "depth-vs-accuracy.png", "Deeper trees memorise the training data; held-out accuracy stops improving early.")

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
