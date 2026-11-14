"""Random forest regression: forecast Reliance Industries' next-20-day volatility from real prices.

Pipeline: download real prices from Yahoo Finance (cached in data/) -> audit and clean -> build features known
today and the realised volatility of the next 20 trading days -> chronological split with a 20-day gap ->
tune and train a forest -> compare with simple baselines -> export ONNX -> verify with onnxruntime ->
write the artefacts the ASP.NET site consumes:

  <web>/App_Data/models/random-forest/{model.onnx, schema.json}
  <web>/wwwroot/projects/random-forest/charts/*.png
  <tests>/fixtures/random-forest.json   (reference inputs/outputs for the .NET parity test)

The forest predicts log(future volatility / 60-day volatility), so it learns "how much will volatility move
away from its recent level" instead of the level itself, which changes between market regimes. The exported
ONNX graph appends exp() and a multiply by the 60-day volatility input, so the site gets a plain percentage.

Run:  uv run python train.py              (uses the cached data/raw_yahoo.csv)
      uv run python train.py --refresh    (downloads fresh prices from Yahoo Finance first)
This project is self-contained: it imports nothing from the web app or other projects (the download and
audit code is deliberately a copy of the one in linear-regression).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import ssl
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import yfinance as yf
from onnx import TensorProto, helper
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

SLUG = "random-forest"
MODEL_VERSION = "1.0.0"
SEED = 42
ONNX_TOLERANCE = 1e-3
TARGET = "next_20d_volatility_pct"
SCALE_COLUMN = "vol_60d_pct"  # the forest predicts log(target / this column)

TICKERS = {"stock": "RELIANCE.NS", "market": "^NSEI", "vix": "^INDIAVIX"}  # Reliance, Nifty 50, India VIX
START_DATE = "2015-01-01"
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_CSV = DATA_DIR / "raw_yahoo.csv"
HORIZON = 20           # trading days the volatility forecast covers
HIGH_WINDOW = 126      # trading days (about six months) for the drawdown reference high
OUTLIER_Z = 5.0        # robust z-score of a daily return above which a day is flagged
TRADING_DAYS = 252
PARKINSON = 4 * np.log(2)

# Input definition. Single source of truth for the generated schema.json and the ONNX input names.
INPUTS = [
    {"name": "vol_5d_pct", "label": "5-day volatility (%)", "min": 0, "max": 100, "step": 1, "default": 20,
     "help": "Annualised daily high-low range volatility of Reliance over the last 5 days"},
    {"name": "vol_20d_pct", "label": "20-day volatility (%)", "min": 5, "max": 80, "step": 1, "default": 22,
     "help": "The same measure over the last 20 days"},
    {"name": "vol_60d_pct", "label": "60-day volatility (%)", "min": 10, "max": 80, "step": 1, "default": 22,
     "help": "The same measure over the last 60 days; the forecast is built around this level"},
    {"name": "return_20d_pct", "label": "20-day return (%)", "min": -25, "max": 30, "step": 0.5, "default": 1,
     "help": "Price change over the last 20 trading days"},
    {"name": "drawdown_pct", "label": "Below 6-month high (%)", "min": -40, "max": 0, "step": 0.5, "default": -8,
     "help": "0 means at the highest close of the last six months; -10 means 10% below it"},
    {"name": "volume_ratio", "label": "Recent volume vs normal (x)", "min": 0.2, "max": 4, "step": 0.1, "default": 1.0,
     "help": "Average volume of the last 5 days against the 60 days before them"},
    {"name": "market_vol_20d_pct", "label": "Market 20-day volatility (%)", "min": 4, "max": 60, "step": 1, "default": 11,
     "help": "Annualised 20-day volatility of the Nifty 50"},
    {"name": "india_vix", "label": "India VIX", "min": 8, "max": 60, "step": 0.5, "default": 15,
     "help": "The market's expected volatility over the next 30 days (index points)"},
]
COLUMNS = [i["name"] for i in INPUTS]

PRESETS = [
    {"name": "Calm market", "values": {"vol_5d_pct": 12, "vol_20d_pct": 15, "vol_60d_pct": 18, "return_20d_pct": 2,
                                       "drawdown_pct": -3, "volume_ratio": 0.8, "market_vol_20d_pct": 8, "india_vix": 11}},
    {"name": "After a sharp sell-off", "values": {"vol_5d_pct": 45, "vol_20d_pct": 38, "vol_60d_pct": 28, "return_20d_pct": -15,
                                                  "drawdown_pct": -25, "volume_ratio": 2.5, "market_vol_20d_pct": 35, "india_vix": 38}},
    {"name": "Stock-specific shock", "values": {"vol_5d_pct": 60, "vol_20d_pct": 25, "vol_60d_pct": 22, "return_20d_pct": 8,
                                                 "drawdown_pct": -4, "volume_ratio": 3.5, "market_vol_20d_pct": 11, "india_vix": 14}},
]


# ---------------------------------------------------------------------------------------------------
# Data: download, audit, clean
# ---------------------------------------------------------------------------------------------------

def _use_system_certificates() -> None:
    """yfinance's HTTP client ignores the Windows certificate store; hand it a bundle built from it.
    Verification stays on. Does nothing if CURL_CA_BUNDLE is already set or on other platforms."""
    if os.environ.get("CURL_CA_BUNDLE") or sys.platform != "win32":
        return
    import certifi

    pems = [ssl.DER_cert_to_PEM_cert(cert) for store in ("ROOT", "CA")
            for cert, encoding, _ in ssl.enum_certificates(store) if encoding == "x509_asn"]
    bundle = Path(tempfile.gettempdir()) / "ml-portfolio-cabundle.pem"
    bundle.write_text(Path(certifi.where()).read_text() + "\n" + "\n".join(pems), encoding="utf-8")
    os.environ["CURL_CA_BUNDLE"] = str(bundle)


def download_raw() -> pd.DataFrame:
    """Daily OHLCV for the stock, the market index and India VIX. `Close` is already adjusted by Yahoo for
    splits and bonus issues; auto_adjust is off so dividends are not baked into the prices."""
    _use_system_certificates()
    raw = yf.download(list(TICKERS.values()), start=START_DATE, auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data (offline, rate limited, or a certificate problem).")
    raw = raw.dropna(how="all")
    DATA_DIR.mkdir(exist_ok=True)
    raw.to_csv(RAW_CSV)
    return raw


def load_raw(refresh: bool = False) -> pd.DataFrame:
    if refresh or not RAW_CSV.exists():
        return download_raw()
    return pd.read_csv(RAW_CSV, header=[0, 1], index_col=0, parse_dates=[0])


@dataclass
class Audit:
    """What the data-quality checks found and what was done about it (printed, charted and documented)."""
    counts: dict
    notes: list[str]
    flagged: pd.DataFrame   # daily returns that are extreme for their day's market move; kept
    returns: pd.DataFrame   # aligned daily returns (%)
    vix: pd.Series          # aligned India VIX closes


def prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, Audit]:
    """Checks the raw download for missing values, bad ranges and outliers, and builds the model table."""
    col = lambda field, key: raw[(field, TICKERS[key])]  # noqa: E731
    close = pd.DataFrame({k: col("Close", k) for k in TICKERS})
    high, low, volume = col("High", "stock"), col("Low", "stock"), col("Volume", "stock")
    counts: dict = {"raw_rows": int(len(close)), "first_date": str(close.index.min().date()),
                    "last_date": str(close.index.max().date()),
                    "duplicate_dates": int(close.index.duplicated().sum()),
                    "non_positive_prices": int((close <= 0).sum().sum())}
    counts["missing_close"] = {k: int(close[k].isna().sum()) for k in close}
    notes = []

    # 1. Missing closes: the index or VIX has no print on days Reliance still traded. Filling would invent
    #    zero returns (and understate volatility), so those days are dropped and returns span aligned dates only.
    keep = close.notna().all(axis=1) & high.notna() & low.notna()
    counts["dropped_missing"] = int((~keep).sum())
    notes.append(f"{counts['dropped_missing']} days dropped because a series had no value")

    # 2. Bad values: zero volume (missing, not "nobody traded"), and a high-low range that is zero or inverted
    #    (a stale or broken quote), which would corrupt the range-based volatility.
    bad_volume = volume.fillna(0) <= 0
    bad_range = (high <= low) | (high <= 0)
    counts["dropped_zero_volume"] = int((keep & bad_volume).sum())
    counts["dropped_bad_range"] = int((keep & ~bad_volume & bad_range).sum())
    keep &= ~bad_volume & ~bad_range
    notes.append(f"{counts['dropped_zero_volume']} days dropped for zero volume and {counts['dropped_bad_range']} for a zero or inverted high-low range")
    close, high, low, volume = close[keep], high[keep], low[keep], volume[keep]
    returns = close[["stock", "market"]].pct_change() * 100

    # 3. Outliers. Volatility IS the extreme days, so large moves are never removed. A day is flagged when the
    #    stock's return is far from what that day's market move explains (robust z-score of the residual).
    paired = returns.dropna()
    slope, intercept = np.polyfit(paired["market"], paired["stock"], 1)
    resid = paired["stock"] - (intercept + slope * paired["market"])
    centred = resid - resid.median()
    z = centred / (1.4826 * centred.abs().median())
    flagged = paired[z.abs() > OUTLIER_Z].assign(residual_z=z[z.abs() > OUTLIER_Z].round(1))
    counts["flagged_outliers"] = int(len(flagged))
    counts["largest_abs_return_pct"] = round(float(returns["stock"].abs().max()), 1)
    notes.append(f"{len(flagged)} days flagged as stock-specific outliers (|robust z| > {OUTLIER_Z:g}) and kept")

    # 4. Features known at the close of each day, and the realised volatility of the following HORIZON days.
    #    Reliance volatility uses the Parkinson high-low estimator (much less noisy than close-to-close).
    range_sq = np.log(high / low) ** 2

    def parkinson(window: int) -> pd.Series:
        return np.sqrt(range_sq.rolling(window).mean() / PARKINSON) * np.sqrt(TRADING_DAYS) * 100

    table = pd.DataFrame({
        "vol_5d_pct": parkinson(5),
        "vol_20d_pct": parkinson(20),
        "vol_60d_pct": parkinson(60),
        "return_20d_pct": (close["stock"] / close["stock"].shift(20) - 1) * 100,
        "drawdown_pct": (close["stock"] / close["stock"].rolling(HIGH_WINDOW).max() - 1) * 100,
        "volume_ratio": volume.rolling(5).mean() / volume.shift(5).rolling(60).mean(),
        "market_vol_20d_pct": returns["market"].rolling(20).std() * np.sqrt(TRADING_DAYS),
        "india_vix": close["vix"],
    })
    table[TARGET] = np.sqrt(range_sq[::-1].rolling(HORIZON).mean()[::-1].shift(-1) / PARKINSON) * np.sqrt(TRADING_DAYS) * 100
    before = len(table)
    table = table.dropna()
    counts["dropped_warmup_or_horizon"] = int(before - len(table))

    # 5. Match the form's domain by capping each feature at the range the form accepts (the rows stay).
    capped = {}
    for i in INPUTS:
        outside = int(((table[i["name"]] < i["min"]) | (table[i["name"]] > i["max"])).sum())
        if outside:
            capped[i["name"]] = outside
        table[i["name"]] = table[i["name"]].clip(i["min"], i["max"])
    counts["capped_to_form_range"] = capped
    counts["rows_used"] = int(len(table))
    notes.append(f"{sum(capped.values())} feature values capped to the form's ranges ({len(capped)} features)")
    return table, Audit(counts, notes, flagged, returns, close["vix"])


# ---------------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------------

def to_onnx(pipeline: Pipeline) -> bytes:
    """One float [N,1] ONNX input per column, with exp() and a multiply by the 60-day volatility appended so the
    graph outputs volatility in percent (the forest itself outputs log of the ratio to that level)."""
    initial_types = [(c, FloatTensorType([None, 1])) for c in COLUMNS]
    model = convert_sklearn(pipeline, initial_types=initial_types, target_opset={"": 17, "ai.onnx.ml": 3})
    graph = model.graph
    raw_out = graph.output[0].name
    graph.node.append(helper.make_node("Exp", [raw_out], ["ratio"], name="to_ratio"))
    graph.node.append(helper.make_node("Reshape", ["ratio", "flat_shape"], ["ratio_flat"], name="flatten_ratio"))
    graph.node.append(helper.make_node("Mul", ["ratio_flat", SCALE_COLUMN], ["volatility"], name="scale_by_60d"))
    graph.initializer.append(helper.make_tensor("flat_shape", TensorProto.INT64, [2], [-1, 1]))
    del graph.output[:]
    graph.output.append(helper.make_tensor_value_info("volatility", TensorProto.FLOAT, [None, 1]))
    onnx.checker.check_model(model)
    return model.SerializeToString()


def onnx_predict(model_bytes: bytes, df: pd.DataFrame) -> np.ndarray:
    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    feed = {c: df[c].to_numpy(dtype=np.float32).reshape(-1, 1) for c in COLUMNS}
    return session.run(None, feed)[0].ravel()


def forecast(model, df: pd.DataFrame) -> np.ndarray:
    """Volatility in percent from a model that predicts log(volatility / 60-day volatility)."""
    return df[SCALE_COLUMN].to_numpy() * np.exp(model.predict(df[COLUMNS]))


@dataclass
class Result:
    model: Pipeline
    model_bytes: bytes
    x_train: pd.DataFrame
    y_train: np.ndarray
    x_test: pd.DataFrame
    y_test: np.ndarray
    linear: LinearRegression      # plain linear regression on the volatility level
    metrics: dict
    importance: pd.Series
    data: pd.DataFrame
    audit: Audit


def _scores(y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {"r2": float(r2_score(y_true, pred)), "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
            "mae": float(mean_absolute_error(y_true, pred))}


def _preprocess() -> ColumnTransformer:
    return ColumnTransformer([("num", "passthrough", COLUMNS)])


def train_and_evaluate(refresh: bool = False) -> Result:
    data, audit = prepare(load_raw(refresh))
    x, y = data[COLUMNS], data[TARGET].to_numpy()
    log_ratio = np.log(y / data[SCALE_COLUMN].to_numpy())

    # Chronological split. Each target covers the next HORIZON days, so the last HORIZON training rows overlap
    # the first test days; skip that many rows between the two so the test set is genuinely unseen.
    n_train = int(len(data) * 0.75)
    x_train, y_train, t_train = x.iloc[:n_train], y[:n_train], log_ratio[:n_train]
    x_test, y_test = x.iloc[n_train + HORIZON:], y[n_train + HORIZON:]

    # Small time-series-aware search on the training data only. Targets overlap heavily and there are only a
    # handful of independent volatility episodes, so the forest is kept shallow with large leaves.
    pipeline = Pipeline([("pre", _preprocess()), ("reg", RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1))])
    grid = GridSearchCV(
        pipeline, {"reg__max_depth": [3, 4, 5], "reg__min_samples_leaf": [40, 60, 80]},
        cv=TimeSeriesSplit(n_splits=4, gap=HORIZON), scoring="neg_mean_squared_error",
    ).fit(x_train, t_train)
    model = grid.best_estimator_
    pred = forecast(model, x_test)

    # Baselines on the same test days. "Linear (same setup)" shares the forest's target; "linear" is the plain
    # level regression most people would try first.
    linear = LinearRegression().fit(x_train, y_train)
    linear_ratio = Pipeline([("pre", _preprocess()), ("reg", LinearRegression())]).fit(x_train, t_train)
    metrics = {
        **_scores(y_test, pred),
        "linear": _scores(y_test, linear.predict(x_test)),
        "linear_same_setup": _scores(y_test, forecast(linear_ratio, x_test)),
        "persistence_20d": _scores(y_test, x_test["vol_20d_pct"].to_numpy()),
        "persistence_60d": _scores(y_test, x_test["vol_60d_pct"].to_numpy()),
        "train_mean": _scores(y_test, np.full_like(y_test, y_train.mean())),
        "best_params": {k.replace("reg__", ""): v for k, v in grid.best_params_.items()},
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "train_period": f"{x_train.index.min().date()} to {x_train.index.max().date()}",
        "test_period": f"{x_test.index.min().date()} to {x_test.index.max().date()}",
        "train_mean_vol": float(y_train.mean()),
        "test_mean_vol": float(y_test.mean()),
    }

    model_bytes = to_onnx(model)
    max_diff = float(np.abs(onnx_predict(model_bytes, x_test) - pred).max())
    assert max_diff < ONNX_TOLERANCE, f"ONNX and sklearn disagree (max diff {max_diff})"
    metrics["onnx_max_abs_diff"] = max_diff
    metrics["onnx_size_kb"] = round(len(model_bytes) / 1024)

    # Permutation importance on the test period: how much R2 drops when a column is shuffled.
    rng = np.random.default_rng(SEED)
    base_r2 = r2_score(y_test, pred)
    drops = {}
    for c in COLUMNS:
        scores = []
        for _ in range(20):
            shuffled = x_test.copy()
            shuffled[c] = rng.permutation(shuffled[c].to_numpy())
            scores.append(base_r2 - r2_score(y_test, forecast(model, shuffled)))
        drops[c] = float(np.mean(scores))
    return Result(model, model_bytes, x_train, y_train, x_test, y_test, linear, metrics, pd.Series(drops), data, audit)


# ---------------------------------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------------------------------

def build_schema() -> dict:
    return {
        "modelFile": "model.onnx",
        "modelVersion": MODEL_VERSION,
        "inputs": [{"type": "number", **i} for i in INPUTS],
        "outputs": {"type": "regression", "unit": "%", "signed": False},
        "presets": PRESETS,
    }


def build_fixtures(result: Result, n_random: int = 6) -> list[dict]:
    """Reference inputs with sklearn's answers, so the .NET runner can be checked against Python."""
    rng = np.random.default_rng(SEED + 1)
    cases = [dict(p["values"]) for p in PRESETS]
    for _ in range(n_random):
        cases.append({i["name"]: round(float(rng.uniform(i["min"], i["max"])) / i["step"]) * i["step"] for i in INPUTS})

    df = pd.DataFrame(cases)[COLUMNS]
    pred = forecast(result.model, df)
    return [
        {"input": {k: float(v) for k, v in row.items()}, "prediction": float(p)}
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
    m = result.metrics
    pred = forecast(result.model, result.x_test)

    def save(fig, name: str, caption: str) -> None:
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=200)
        plt.close(fig)
        charts.append({"file": name, "caption": caption})

    # Data chart: Reliance's 20-day volatility and India VIX, with the flagged days.
    data = result.data
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(data.index, data["vol_20d_pct"], color=ACCENT, linewidth=1.3, label="Reliance 20-day volatility")
    ax.plot(data.index, data["india_vix"], color=MUTED, linewidth=1.0, label="India VIX")
    flagged = result.audit.flagged.index.intersection(data.index)
    ax.scatter(flagged, data.loc[flagged, "vol_20d_pct"], s=22, facecolors="none", edgecolors="#dc2626", linewidths=1.1,
               label=f"Flagged stock-specific day ({len(result.audit.flagged)})", zorder=3)
    ax.set_ylabel("Annualised volatility (%) / VIX points")
    ax.set_title("Reliance volatility and India VIX")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "volatility-history.png", "Reliance's 20-day range-based volatility next to India VIX. The spikes are real (March 2020, the 2024 election result, single-day shocks); red circles mark stock-specific outlier days, which were kept.")

    # Forecast timeline over the test period.
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    idx = result.x_test.index
    ax.plot(idx, result.y_test, color="#1c1917", linewidth=1.2, label="Actual next-20-day volatility")
    ax.plot(idx, result.x_test["vol_20d_pct"], color=MUTED, linewidth=1.0, linestyle="--", label="Persistence (last 20 days)")
    ax.plot(idx, pred, color=ACCENT, linewidth=1.8, label="Random forest")
    ax.set_ylabel("Annualised volatility (%)")
    ax.set_title("Forecast vs outcome, test period")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "forecast-timeline.png", "The forest's forecast against what happened over the next 20 days, on the most recent days held out from training. It tracks the broad level but cannot foresee sudden shocks.")

    # Scatter (thumbnail).
    fig, ax = plt.subplots(figsize=(5.8, 5))
    ax.scatter(result.y_test, pred, s=10, alpha=0.4, color=ACCENT)
    lim = [min(result.y_test.min(), pred.min()), max(result.y_test.max(), pred.max())]
    ax.plot(lim, lim, "--", color=MUTED, label="Perfect prediction")
    ax.set_xlabel("Actual next-20-day volatility (%)")
    ax.set_ylabel("Predicted volatility (%)")
    ax.set_title(f"Predicted vs actual (RMSE {m['rmse']:.1f} points)")
    ax.legend()
    save(fig, "predicted-vs-actual.png", "Predicted against actual 20-day volatility on the test period. Forecasts are compressed towards the middle: the extremes are not predicted.")

    # Model comparison.
    items = [("Train\naverage", m["train_mean"]["rmse"]), ("60-day\nvolatility", m["persistence_60d"]["rmse"]),
             ("20-day\nvolatility", m["persistence_20d"]["rmse"]), ("Linear\nregression", m["linear"]["rmse"]),
             ("Random\nforest", m["rmse"])]
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    colours = ["#d6d3d1", "#d6d3d1", "#d6d3d1", ACCENT_LIGHT, ACCENT]
    bars = ax.bar([n for n, _ in items], [v for _, v in items], color=colours, width=0.6)
    for bar, (_, v) in zip(bars, items):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center", color="#44403c")
    ax.set_ylabel("Error (RMSE, volatility points) - lower is better")
    ax.set_title("Forest vs simpler forecasts")
    ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), fontsize=9)
    save(fig, "model-comparison.png", "Test-period error for five forecasts. Simply carrying today's volatility forward is a hard baseline to beat; the forest and a straight line both do a little better.")

    # Permutation importance.
    order = result.importance.sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([n.replace("_pct", "").replace("_", " ") for n in order.index], order.values, color=ACCENT)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_title("Feature importance (permutation)")
    ax.set_xlabel("Drop in test R² when the column is shuffled")
    save(fig, "feature-importance.png", "Which inputs the forest relies on, measured by how much accuracy falls when each is scrambled. The 60-day level anchors the forecast; bars below zero are inputs that add noise on the test period.")

    # Recent-spike effect, forest vs straight line: how the forecast moves as 20-day volatility rises while the
    # 60-day level stays typical (the forest predicts a ratio to that level, so this shows mean reversion).
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    base = {c: float(result.x_train[c].median()) for c in COLUMNS}
    spike = np.linspace(10, 55, 60)
    grid = pd.DataFrame({**{c: np.full(len(spike), base[c]) for c in COLUMNS}, "vol_20d_pct": spike})[COLUMNS]
    ax.plot(spike, forecast(result.model, grid), color=ACCENT, linewidth=2, label="Random forest")
    ax.plot(spike, result.linear.predict(grid), color=MUTED, linewidth=2, linestyle="--", label="Linear regression")
    ax.set_xlabel("20-day volatility (%), 60-day level held typical")
    ax.set_ylabel("Predicted next-20-day volatility (%)")
    ax.set_title("Effect of a recent volatility spike")
    ax.legend(fontsize=9)
    save(fig, "spike-effect.png", "Predicted volatility as the last 20 days get more volatile, other inputs held at typical values. Both models stay close to the 60-day level instead of carrying a spike forward, which is why simply repeating the last 20 days does worse.")

    shutil.copyfile(out_dir / "predicted-vs-actual.png", out_dir / "thumbnail.png")
    return charts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo-root", type=Path, default=default_root,
                        help="repository root containing src/ and tests/ (default: two levels up)")
    parser.add_argument("--refresh", action="store_true", help="download fresh prices from Yahoo Finance first")
    args = parser.parse_args()
    web = args.repo_root / "src" / "MLPortfolio.Web"

    result = train_and_evaluate(refresh=args.refresh)
    print("data audit:", json.dumps(result.audit.counts, indent=2))
    print(json.dumps(result.metrics, indent=2))
    print("importance:", result.importance.round(3).to_dict())

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
