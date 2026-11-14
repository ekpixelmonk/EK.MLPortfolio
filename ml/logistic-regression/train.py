"""Logistic regression pilot: loan default classifier on synthetic data.

Pipeline: generate data -> train sklearn pipeline -> evaluate -> export ONNX -> verify with
onnxruntime -> write the artefacts the ASP.NET site consumes:

  <web>/App_Data/models/logistic-regression/{model.onnx, schema.json}
  <web>/wwwroot/projects/logistic-regression/charts/*.png
  <tests>/fixtures/logistic-regression.json   (reference inputs/outputs for the .NET parity test)

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SLUG = "logistic-regression"
MODEL_VERSION = "1.0.0"
SEED = 42
LABELS = ["Repaid", "Default"]  # index = class id; probability[1] is P(default)

# Input definition. Single source of truth for the generated schema.json and the ONNX input names.
NUMERIC_INPUTS = [
    {"name": "annual_income_k", "label": "Annual income (thousands)", "min": 10, "max": 300, "step": 1, "default": 60,
     "help": "Gross yearly income, in thousands"},
    {"name": "debt_to_income_pct", "label": "Debt-to-income ratio (%)", "min": 0, "max": 80, "step": 1, "default": 30,
     "help": "Monthly debt payments as a share of monthly income"},
    {"name": "credit_history_years", "label": "Credit history (years)", "min": 0, "max": 40, "step": 1, "default": 8},
    {"name": "late_payments_12m", "label": "Late payments (last 12 months)", "min": 0, "max": 12, "step": 1, "default": 0},
]
EMPLOYMENT_OPTIONS = ["salaried", "self-employed", "unemployed"]
NUMERIC_COLUMNS = [i["name"] for i in NUMERIC_INPUTS]
CATEGORICAL_COLUMNS = ["employment"]
ALL_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

PRESETS = [
    {"name": "Low risk", "values": {"annual_income_k": 120, "debt_to_income_pct": 15, "credit_history_years": 15,
                                    "late_payments_12m": 0, "employment": "salaried"}},
    {"name": "High risk", "values": {"annual_income_k": 28, "debt_to_income_pct": 62, "credit_history_years": 2,
                                     "late_payments_12m": 5, "employment": "unemployed"}},
]


def make_data(n: int = 6000, seed: int = SEED) -> pd.DataFrame:
    """Synthetic borrowers. The true default process is known, so the fit can be sanity-checked."""
    rng = np.random.default_rng(seed)
    income = np.clip(rng.lognormal(mean=np.log(60), sigma=0.5, size=n), 10, 300)
    dti = np.clip(rng.beta(2.2, 4.5, size=n) * 80, 0, 80)
    history = np.clip(rng.gamma(shape=2.5, scale=3.5, size=n), 0, 40)
    late = np.clip(rng.poisson(0.8, size=n), 0, 12)
    employment = rng.choice(EMPLOYMENT_OPTIONS, size=n, p=[0.65, 0.25, 0.10])

    emp_effect = pd.Series(employment).map({"salaried": 0.0, "self-employed": 0.4, "unemployed": 1.2}).to_numpy()
    z = -1.6 + 0.06 * (dti - 30) - 0.01 * (income - 60) - 0.08 * history + 0.45 * late + emp_effect
    p_default = 1 / (1 + np.exp(-z))
    default = rng.binomial(1, p_default)

    return pd.DataFrame({
        "annual_income_k": income.round(0),
        "debt_to_income_pct": dti.round(0),
        "credit_history_years": history.round(0),
        "late_payments_12m": late,
        "employment": employment,
        "default": default,
    })


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLUMNS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])


def to_onnx(pipeline: Pipeline) -> bytes:
    """One ONNX input per column (float [N,1] for numbers, string [N,1] for the select)."""
    initial_types = [(c, FloatTensorType([None, 1])) for c in NUMERIC_COLUMNS] + \
                    [(c, StringTensorType([None, 1])) for c in CATEGORICAL_COLUMNS]
    clf = pipeline.named_steps["clf"]
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
    x_test: pd.DataFrame
    y_test: np.ndarray
    metrics: dict


def train_and_evaluate() -> Result:
    data = make_data()
    x = data[ALL_COLUMNS]
    y = data["default"].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=SEED, stratify=y)

    pipeline = build_pipeline().fit(x_train, y_train)
    proba = pipeline.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, proba > 0.5)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "default_rate": float(y.mean()),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }

    model_bytes = to_onnx(pipeline)

    # Verify the ONNX model reproduces sklearn on the whole test set before anything is written.
    _, onnx_proba = onnx_predict(model_bytes, x_test)
    max_diff = float(np.abs(onnx_proba[:, 1] - proba).max())
    assert max_diff < 1e-4, f"ONNX and sklearn disagree (max diff {max_diff})"
    metrics["onnx_max_abs_diff"] = max_diff
    return Result(pipeline, model_bytes, x_test, y_test, metrics)


# ---------------------------------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------------------------------

def build_schema() -> dict:
    inputs = [{"type": "number", **i} for i in NUMERIC_INPUTS]
    inputs.append({"name": "employment", "label": "Employment", "type": "select",
                   "options": EMPLOYMENT_OPTIONS, "default": "salaried"})
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
    cases = [p["values"] for p in PRESETS]
    for _ in range(n_random):
        case = {i["name"]: float(rng.integers(i["min"], i["max"] + 1)) for i in NUMERIC_INPUTS}
        case["employment"] = str(rng.choice(EMPLOYMENT_OPTIONS))
        cases.append(case)

    df = pd.DataFrame(cases)[ALL_COLUMNS]
    proba = result.pipeline.predict_proba(df)
    return [
        {
            "input": {k: (float(v) if k in NUMERIC_COLUMNS else str(v)) for k, v in row.items()},
            "label": LABELS[int(np.argmax(p))],
            "probabilities": {LABELS[0]: float(p[0]), LABELS[1]: float(p[1])},
        }
        for row, p in zip(df.to_dict("records"), proba)
    ]


ACCENT = "#4f46e5"
MUTED = "#78716c"


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "sans-serif", "font.size": 11, "axes.edgecolor": "#d6d3d1", "axes.labelcolor": "#44403c",
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 13, "axes.grid": True, "grid.color": "#f0efed",
    })


def make_charts(result: Result, out_dir: Path) -> list[dict]:
    """Writes 2x PNG charts and returns the captions for projects.json."""
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    x_test, y_test = result.x_test, result.y_test
    proba = result.pipeline.predict_proba(x_test)[:, 1]
    charts: list[dict] = []

    def save(fig, name: str, caption: str) -> None:
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=200)
        plt.close(fig)
        charts.append({"file": name, "caption": caption})

    fig, ax = plt.subplots(figsize=(6, 4.5))
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, color=ACCENT, linewidth=2, label="Logistic regression")
    ax.plot([0, 1], [0, 1], "--", color=MUTED, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curve (AUC {result.metrics['roc_auc']:.2f})")
    ax.legend(loc="lower right")
    save(fig, "roc-curve.png", "ROC curve on held-out data: how well the model ranks risky borrowers above safe ones.")

    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, proba > 0.5, display_labels=LABELS, ax=ax, cmap="Purples",
                                            colorbar=False)
    ax.grid(False)
    ax.set_title("Confusion matrix (threshold 0.5)")
    save(fig, "confusion-matrix.png", "Confusion matrix on held-out data at a 0.5 probability threshold.")

    names = result.pipeline.named_steps["pre"].get_feature_names_out()
    coefs = result.pipeline.named_steps["clf"].coef_[0]
    order = np.argsort(coefs)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([str(names[i]).split("__")[-1] for i in order], coefs[order],
            color=[ACCENT if coefs[i] > 0 else "#a5b4fc" for i in order])
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_title("Coefficients (standardised features)")
    ax.set_xlabel("Effect on log-odds of default")
    save(fig, "coefficients.png", "Learned coefficients: positive values push the prediction toward default.")

    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(proba, bins) - 1, 0, 9)
    mean_pred = np.array([proba[idx == b].mean() for b in range(10) if (idx == b).any()])
    frac_pos = np.array([y_test[idx == b].mean() for b in range(10) if (idx == b).any()])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color=MUTED, label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", color=ACCENT, label="Model")
    ax.set_xlabel("Predicted probability of default")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration")
    ax.legend()
    save(fig, "calibration.png", "Calibration: predicted probabilities against the default rate actually observed.")

    shutil.copyfile(out_dir / "roc-curve.png", out_dir / "thumbnail.png")
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
