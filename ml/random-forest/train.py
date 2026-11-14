"""Random forest regression: 30-day stock volatility on synthetic data.

Pipeline: generate data -> train sklearn pipeline -> evaluate (against a linear baseline) -> export ONNX ->
verify with onnxruntime -> write the artefacts the ASP.NET site consumes:

  <web>/App_Data/models/random-forest/{model.onnx, schema.json}
  <web>/wwwroot/projects/random-forest/charts/*.png
  <tests>/fixtures/random-forest.json   (reference inputs/outputs for the .NET parity test)

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SLUG = "random-forest"
MODEL_VERSION = "1.0.0"
SEED = 42
ONNX_TOLERANCE = 1e-3
SECTORS = ["technology", "banks", "utilities", "energy", "healthcare"]

NUMERIC_INPUTS = [
    {"name": "beta", "label": "Beta (market sensitivity)", "min": 0.2, "max": 2.5, "step": 0.05, "default": 1.0,
     "help": "How strongly the stock moves with the market"},
    {"name": "debt_to_equity", "label": "Debt-to-equity ratio", "min": 0, "max": 4, "step": 0.1, "default": 0.8},
    {"name": "market_cap_bn", "label": "Market cap (billions)", "min": 1, "max": 500, "step": 1, "default": 40},
]
SELECT_INPUTS = [
    {"name": "sector", "label": "Sector", "options": SECTORS, "default": "technology"},
]
NUMERIC_COLUMNS = [i["name"] for i in NUMERIC_INPUTS]
CATEGORICAL_COLUMNS = [i["name"] for i in SELECT_INPUTS]
ALL_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

PRESETS = [
    {"name": "Steady utility", "values": {"beta": 0.5, "debt_to_equity": 1.2, "market_cap_bn": 60, "sector": "utilities"}},
    {"name": "Volatile small-cap tech", "values": {"beta": 1.8, "debt_to_equity": 2.5, "market_cap_bn": 4, "sector": "technology"}},
    {"name": "Large-cap bank", "values": {"beta": 1.1, "debt_to_equity": 0.9, "market_cap_bn": 220, "sector": "banks"}},
]


def make_data(n: int = 4000, seed: int = SEED) -> pd.DataFrame:
    """Synthetic stocks. Volatility has threshold and interaction effects that a straight line cannot capture."""
    rng = np.random.default_rng(seed)
    beta = np.clip(rng.normal(1.0, 0.45, n), 0.2, 2.5).round(2)
    debt = np.clip(rng.gamma(2.0, 0.55, n), 0, 4).round(1)
    cap = np.clip(rng.lognormal(np.log(40), 1.1, n), 1, 500).round()
    sector = rng.choice(SECTORS, size=n, p=[0.28, 0.2, 0.14, 0.18, 0.2])

    base = pd.Series(sector).map({"technology": 20, "banks": 17, "utilities": 9, "energy": 24, "healthcare": 16}).to_numpy()
    leverage_penalty = np.where(debt > 2.0, 5.0 * (debt - 2.0), 0.0)               # kicks in above 2x
    sector_beta = np.where(np.isin(sector, ["technology", "energy"]), 1.4, 0.8)     # beta matters more in some sectors
    small_cap = 6.0 * np.exp(-cap / 25.0)                                           # small caps are jumpier
    vol = base + 9.0 * sector_beta * (beta - 1.0) + leverage_penalty + small_cap + rng.normal(0, 2.2, n)
    return pd.DataFrame({"beta": beta, "debt_to_equity": debt, "market_cap_bn": cap, "sector": sector,
                         "target": np.clip(vol, 4, None).round(2)})


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC_COLUMNS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
    ])
    forest = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    return Pipeline([("pre", pre), ("reg", forest)])


def to_onnx(pipeline: Pipeline) -> bytes:
    """One ONNX input per column (float [N,1] for numbers, string [N,1] for selects)."""
    initial_types = [(c, FloatTensorType([None, 1])) for c in NUMERIC_COLUMNS] + \
                    [(c, StringTensorType([None, 1])) for c in CATEGORICAL_COLUMNS]
    model = convert_sklearn(pipeline, initial_types=initial_types, target_opset={"": 17, "ai.onnx.ml": 3})
    return model.SerializeToString()


def onnx_predict(model_bytes: bytes, df: pd.DataFrame) -> np.ndarray:
    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    feed = {c: df[c].to_numpy(dtype=np.float32).reshape(-1, 1) for c in NUMERIC_COLUMNS}
    feed |= {c: df[c].to_numpy(dtype=object).reshape(-1, 1) for c in CATEGORICAL_COLUMNS}
    return session.run(None, feed)[0].ravel()


@dataclass
class Result:
    pipeline: Pipeline
    model_bytes: bytes
    x_train: pd.DataFrame
    y_train: np.ndarray
    x_test: pd.DataFrame
    y_test: np.ndarray
    linear_pred: np.ndarray
    metrics: dict


def train_and_evaluate() -> Result:
    data = make_data()
    x = data[ALL_COLUMNS]
    y = data["target"].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=SEED)

    pipeline = build_pipeline().fit(x_train, y_train)
    pred = pipeline.predict(x_test)

    # Baseline: plain linear regression on the same columns, to show what the forest adds.
    linear = Pipeline([("pre", ColumnTransformer([
        ("num", "passthrough", NUMERIC_COLUMNS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
    ])), ("reg", LinearRegression())]).fit(x_train, y_train)
    linear_pred = linear.predict(x_test)

    metrics = {
        "r2": float(r2_score(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
        "linear_baseline_r2": float(r2_score(y_test, linear_pred)),
        "linear_baseline_rmse": float(np.sqrt(mean_squared_error(y_test, linear_pred))),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
    }

    model_bytes = to_onnx(pipeline)
    max_diff = float(np.abs(onnx_predict(model_bytes, x_test) - pred).max())
    assert max_diff < ONNX_TOLERANCE, f"ONNX and sklearn disagree (max diff {max_diff})"
    metrics["onnx_max_abs_diff"] = max_diff
    metrics["onnx_size_kb"] = round(len(model_bytes) / 1024)
    return Result(pipeline, model_bytes, x_train, y_train, x_test, y_test, linear_pred, metrics)


def build_schema() -> dict:
    inputs = [{"type": "number", **i} for i in NUMERIC_INPUTS] + [{"type": "select", **i} for i in SELECT_INPUTS]
    return {
        "modelFile": "model.onnx",
        "modelVersion": MODEL_VERSION,
        "inputs": inputs,
        "outputs": {"type": "regression", "unit": "%"},
        "presets": PRESETS,
    }


def build_fixtures(result: Result, n_random: int = 6) -> list[dict]:
    """Reference inputs with sklearn's answers, so the .NET runner can be checked against Python."""
    rng = np.random.default_rng(SEED + 1)
    cases = [dict(p["values"]) for p in PRESETS]
    for _ in range(n_random):
        case = {i["name"]: round(float(rng.uniform(i["min"], i["max"])) / i["step"]) * i["step"]
                for i in NUMERIC_INPUTS}
        case["sector"] = str(rng.choice(SECTORS))
        cases.append(case)

    df = pd.DataFrame(cases)[ALL_COLUMNS]
    pred = result.pipeline.predict(df)
    return [
        {"input": {k: (float(v) if k in NUMERIC_COLUMNS else str(v)) for k, v in row.items()}, "prediction": float(p)}
        for row, p in zip(df.to_dict("records"), pred)
    ]


ACCENT = "#4f46e5"
ACCENT_LIGHT = "#a5b4fc"
MUTED = "#78716c"


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
    pred = result.pipeline.predict(result.x_test)

    def save(fig, name: str, caption: str) -> None:
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=200)
        plt.close(fig)
        charts.append({"file": name, "caption": caption})

    fig, ax = plt.subplots(figsize=(5.8, 5))
    ax.scatter(result.y_test, pred, s=10, alpha=0.4, color=ACCENT)
    lim = [min(result.y_test.min(), pred.min()), max(result.y_test.max(), pred.max())]
    ax.plot(lim, lim, "--", color=MUTED, label="Perfect prediction")
    ax.set_xlabel("Actual 30-day volatility (%)")
    ax.set_ylabel("Predicted volatility (%)")
    ax.set_title(f"Predicted vs actual (R² {result.metrics['r2']:.2f})")
    ax.legend()
    save(fig, "predicted-vs-actual.png", "Predicted against actual volatility on held-out stocks.")

    names = [str(n).split("__")[-1] for n in result.pipeline.named_steps["pre"].get_feature_names_out()]
    imp = result.pipeline.named_steps["reg"].feature_importances_
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([names[i] for i in order], imp[order], color=ACCENT)
    ax.set_title("Feature importance")
    ax.set_xlabel("Share of variance explained")
    save(fig, "feature-importance.png", "Which inputs the forest relies on most to predict volatility.")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    scores = [result.metrics["linear_baseline_r2"], result.metrics["r2"]]
    bars = ax.bar(["Linear regression", "Random forest"], scores, color=[ACCENT_LIGHT, ACCENT], width=0.55)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, s + 0.01, f"{s:.2f}", ha="center", color="#44403c")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R² on held-out data")
    ax.set_title("Forest vs straight line")
    ax.grid(axis="x", visible=False)
    save(fig, "forest-vs-linear.png", "Same data, same inputs: the forest captures effects a straight line misses.")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    base = {c: float(result.x_train[c].median()) for c in NUMERIC_COLUMNS}
    betas = np.linspace(0.2, 2.5, 60)
    for sector, colour, style in [("technology", ACCENT, "-"), ("utilities", MUTED, "-"), ("energy", ACCENT_LIGHT, "--")]:
        grid = pd.DataFrame({**{c: np.full(len(betas), base[c]) for c in NUMERIC_COLUMNS}, "beta": betas,
                             "sector": sector})[ALL_COLUMNS]
        ax.plot(betas, result.pipeline.predict(grid), style, color=colour, linewidth=2, label=sector)
    ax.set_xlabel("Beta")
    ax.set_ylabel("Predicted volatility (%)")
    ax.set_title("Effect of beta by sector")
    ax.legend()
    save(fig, "beta-by-sector.png", "Predicted volatility as beta rises, other inputs held typical. The slope differs by sector.")

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
