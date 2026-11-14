"""Linear regression pilot: predict a stock's daily return from market, sector and volume signals.

Pipeline: generate data -> train sklearn pipeline -> evaluate -> export ONNX -> verify with
onnxruntime -> write the artefacts the ASP.NET site consumes:

  <web>/App_Data/models/linear-regression/{model.onnx, schema.json}
  <web>/wwwroot/projects/linear-regression/charts/*.png
  <tests>/fixtures/linear-regression.json   (reference inputs/outputs for the .NET parity test)

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
from skl2onnx.common.data_types import FloatTensorType
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SLUG = "linear-regression"
MODEL_VERSION = "1.0.0"
SEED = 42
TARGET = "stock_return_pct"

# Input definition. Single source of truth for the generated schema.json and the ONNX input names.
INPUTS = [
    {"name": "market_return_pct", "label": "Market return today (%)", "min": -5, "max": 5, "step": 0.1, "default": 0.5,
     "help": "Daily return of the broad market index"},
    {"name": "sector_return_pct", "label": "Sector return today (%)", "min": -5, "max": 5, "step": 0.1, "default": 0.4,
     "help": "Daily return of the stock's sector index"},
    {"name": "volume_ratio", "label": "Volume vs 20-day average (x)", "min": 0.2, "max": 4, "step": 0.1, "default": 1.0,
     "help": "1.0 means normal trading volume"},
]
COLUMNS = [i["name"] for i in INPUTS]

PRESETS = [
    {"name": "Strong market day", "values": {"market_return_pct": 2.1, "sector_return_pct": 1.8, "volume_ratio": 1.6}},
    {"name": "Sell-off", "values": {"market_return_pct": -3.2, "sector_return_pct": -2.6, "volume_ratio": 2.8}},
    {"name": "Quiet day", "values": {"market_return_pct": 0.1, "sector_return_pct": 0.0, "volume_ratio": 0.7}},
]


def make_data(n: int = 3000, seed: int = SEED) -> pd.DataFrame:
    """Synthetic daily returns with a known relationship (market beta ~1.15) plus noise."""
    rng = np.random.default_rng(seed)
    market = np.clip(rng.normal(0.05, 1.1, n), -5, 5)
    sector = np.clip(0.7 * market + rng.normal(0, 0.7, n), -5, 5)
    volume = np.clip(rng.lognormal(mean=0.0, sigma=0.35, size=n), 0.2, 4)
    stock = 0.02 + 1.15 * market + 0.35 * sector + 0.10 * (volume - 1) + rng.normal(0, 0.8, n)
    return pd.DataFrame({
        "market_return_pct": market.round(2),
        "sector_return_pct": sector.round(2),
        "volume_ratio": volume.round(2),
        TARGET: stock.round(3),
    })


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([("num", StandardScaler(), COLUMNS)])
    return Pipeline([("pre", pre), ("reg", LinearRegression())])


def to_onnx(pipeline: Pipeline) -> bytes:
    """One float [N,1] ONNX input per column."""
    initial_types = [(c, FloatTensorType([None, 1])) for c in COLUMNS]
    model = convert_sklearn(pipeline, initial_types=initial_types, target_opset={"": 17, "ai.onnx.ml": 3})
    return model.SerializeToString()


def onnx_predict(model_bytes: bytes, df: pd.DataFrame) -> np.ndarray:
    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    feed = {c: df[c].to_numpy(dtype=np.float32).reshape(-1, 1) for c in COLUMNS}
    (out,) = session.run(None, feed)
    return out.reshape(-1)


@dataclass
class Result:
    pipeline: Pipeline
    model_bytes: bytes
    x_test: pd.DataFrame
    y_test: np.ndarray
    metrics: dict


def train_and_evaluate() -> Result:
    data = make_data()
    x, y = data[COLUMNS], data[TARGET].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=SEED)

    pipeline = build_pipeline().fit(x_train, y_train)
    pred = pipeline.predict(x_test)
    metrics = {
        "r2": float(r2_score(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }

    model_bytes = to_onnx(pipeline)

    # Verify the ONNX model reproduces sklearn on the whole test set before anything is written.
    max_diff = float(np.abs(onnx_predict(model_bytes, x_test) - pred).max())
    assert max_diff < 1e-3, f"ONNX and sklearn disagree (max diff {max_diff})"
    metrics["onnx_max_abs_diff"] = max_diff
    return Result(pipeline, model_bytes, x_test, y_test, metrics)


# ---------------------------------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------------------------------

def build_schema() -> dict:
    return {
        "modelFile": "model.onnx",
        "modelVersion": MODEL_VERSION,
        "inputs": [{"type": "number", **i} for i in INPUTS],
        "outputs": {"type": "regression", "unit": "%"},
        "presets": PRESETS,
    }


def build_fixtures(result: Result, n_random: int = 6) -> list[dict]:
    """Reference inputs with sklearn's answers, so the .NET runner can be checked against Python."""
    rng = np.random.default_rng(SEED + 1)
    cases = [p["values"] for p in PRESETS]
    for _ in range(n_random):
        cases.append({i["name"]: round(float(rng.uniform(i["min"], i["max"])), 1) for i in INPUTS})

    df = pd.DataFrame(cases)[COLUMNS]
    pred = result.pipeline.predict(df)
    return [
        {"input": {k: float(v) for k, v in row.items()}, "prediction": float(p)}
        for row, p in zip(df.to_dict("records"), pred)
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
    pred = result.pipeline.predict(x_test)
    resid = y_test - pred
    charts: list[dict] = []

    def save(fig, name: str, caption: str) -> None:
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=200)
        plt.close(fig)
        charts.append({"file": name, "caption": caption})

    lim = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(pred, y_test, s=10, alpha=0.45, color=ACCENT)
    ax.plot(lim, lim, "--", color=MUTED, label="Perfect prediction")
    ax.set_xlabel("Predicted return (%)")
    ax.set_ylabel("Actual return (%)")
    ax.set_title(f"Predicted vs actual (R² {result.metrics['r2']:.2f})")
    ax.legend(loc="upper left")
    save(fig, "predicted-vs-actual.png", "Predicted against actual daily returns on held-out data.")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(pred, resid, s=10, alpha=0.45, color=ACCENT)
    ax.axhline(0, color=MUTED, linestyle="--")
    ax.set_xlabel("Predicted return (%)")
    ax.set_ylabel("Residual (%)")
    ax.set_title("Residuals vs predicted")
    save(fig, "residuals.png", "Residuals show no pattern, which suggests a straight line is a reasonable fit.")

    scaler = result.pipeline.named_steps["pre"].named_transformers_["num"]
    per_unit = result.pipeline.named_steps["reg"].coef_ / scaler.scale_
    labels = ["Market return (%)", "Sector return (%)", "Volume ratio (x)"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh(labels, per_unit, color=ACCENT)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("Change in predicted return (%) per 1-unit increase")
    ax.set_title("Coefficients (original units)")
    save(fig, "coefficients.png", "Learned coefficients: the market return coefficient is the stock's beta.")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.hist(resid, bins=30, color=ACCENT, alpha=0.8)
    ax.set_xlabel("Residual (%)")
    ax.set_ylabel("Days")
    ax.set_title(f"Residual distribution (RMSE {result.metrics['rmse']:.2f}%)")
    save(fig, "residual-distribution.png", "Residuals are roughly bell-shaped and centred on zero.")

    shutil.copyfile(out_dir / "predicted-vs-actual.png", out_dir / "thumbnail.png")
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
