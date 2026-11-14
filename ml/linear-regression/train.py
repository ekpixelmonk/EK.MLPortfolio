"""Linear regression: predict Reliance Industries' daily return from the Nifty 50 and volume.

Pipeline: download real prices from Yahoo Finance (cached in data/) -> audit and clean -> chronological split ->
train sklearn pipeline -> evaluate -> export ONNX -> verify with
onnxruntime -> write the artefacts the ASP.NET site consumes:

  <web>/App_Data/models/linear-regression/{model.onnx, schema.json}
  <web>/wwwroot/projects/linear-regression/charts/*.png
  <tests>/fixtures/linear-regression.json   (reference inputs/outputs for the .NET parity test)

Run:  uv run python train.py              (uses the cached data/raw_yahoo.csv)
      uv run python train.py --refresh    (downloads fresh prices from Yahoo Finance first)
This project is self-contained: it imports nothing from the web app or other projects.
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
import onnxruntime as ort
import pandas as pd
import yfinance as yf
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SLUG = "linear-regression"
MODEL_VERSION = "1.0.0"
SEED = 42
TARGET = "stock_return_pct"

TICKERS = {"stock": "RELIANCE.NS", "market": "^NSEI"}  # Reliance, Nifty 50
START_DATE = "2015-01-01"
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_CSV = DATA_DIR / "raw_yahoo.csv"
VOLUME_WINDOW = 20
OUTLIER_Z = 5.0  # robust z-score of the residual above which a day is flagged

# Input definition. Single source of truth for the generated schema.json and the ONNX input names.
INPUTS = [
    {"name": "market_return_pct", "label": "Market return today (%)", "min": -5, "max": 5, "step": 0.1, "default": 0.5,
     "help": "Daily return of the Nifty 50"},
    {"name": "volume_ratio", "label": "Volume vs 20-day average (x)", "min": 0.2, "max": 4, "step": 0.1, "default": 1.0,
     "help": "1.0 means normal trading volume"},
]
COLUMNS = [i["name"] for i in INPUTS]

PRESETS = [
    {"name": "Strong market day", "values": {"market_return_pct": 2.1, "volume_ratio": 1.6}},
    {"name": "Sell-off", "values": {"market_return_pct": -3.2, "volume_ratio": 2.8}},
    {"name": "Quiet day", "values": {"market_return_pct": 0.1, "volume_ratio": 0.7}},
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
    """Daily OHLCV for the stock and the market index. `Close` is already adjusted by
    Yahoo for splits and bonus issues; auto_adjust is off so dividends are not baked into the prices."""
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
    flagged: pd.DataFrame       # kept in the data, but extreme for their inputs
    out_of_range: pd.DataFrame  # removed: market move beyond the form's +/-5% range
    prices: pd.DataFrame        # aligned closing prices, for the price-history chart


def prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, Audit]:
    """Checks the raw download for missing values, bad volumes and outliers, and builds the model table."""
    close = pd.DataFrame({k: raw[("Close", t)] for k, t in TICKERS.items()})
    volume = raw[("Volume", TICKERS["stock"])]
    counts: dict = {"raw_rows": int(len(close)), "first_date": str(close.index.min().date()),
                    "last_date": str(close.index.max().date()),
                    "duplicate_dates": int(close.index.duplicated().sum()),
                    "non_positive_prices": int((close <= 0).sum().sum())}
    counts["missing_close"] = {k: int(close[k].isna().sum()) for k in close}
    notes = []

    # 1. Missing closes. Indices have no print on days Reliance still traded (holidays, special sessions).
    #    Filling would invent zero returns, so those days are dropped and returns span aligned dates only.
    aligned = close.dropna()
    counts["dropped_missing_close"] = int(len(close) - len(aligned))
    notes.append(f"{counts['dropped_missing_close']} days dropped because an index or the stock had no close")

    # 2. Zero volume on a trading day is missing volume, not zero trading.
    zero_volume = volume.reindex(aligned.index).fillna(0) <= 0
    counts["dropped_zero_volume"] = int(zero_volume.sum())
    aligned = aligned[~zero_volume]
    volume = volume.reindex(aligned.index)
    notes.append(f"{counts['dropped_zero_volume']} days dropped for zero stock volume")

    # 3. Features. Returns are computed after alignment so stock and indices cover the same interval.
    returns = aligned.pct_change() * 100
    baseline = volume.shift(1).rolling(VOLUME_WINDOW).mean()  # the 20 previous days, excluding today
    table = pd.DataFrame({
        "market_return_pct": returns["market"],
        "volume_ratio": volume / baseline, TARGET: returns["stock"],
    }).dropna()
    counts["dropped_warmup"] = int(len(aligned) - len(table))

    # 4. Stale prints: a flat day for the stock while the market moved is suspicious.
    stale = (table[TARGET] == 0) & (table["market_return_pct"].abs() > 0.5)
    counts["stale_stock_prices"] = int(stale.sum())
    table = table[~stale]

    # 5. Match the form's domain. Volume is capped at the form's range (a 9x day still means "very busy"),
    #    but a market move beyond +/-5% is dropped: it is a crash or election-day regime the form
    #    does not accept, and one such day would dominate a least-squares fit.
    vol = next(i for i in INPUTS if i["name"] == "volume_ratio")
    counts["volume_capped"] = int(((table["volume_ratio"] < vol["min"]) | (table["volume_ratio"] > vol["max"])).sum())
    table["volume_ratio"] = table["volume_ratio"].clip(vol["min"], vol["max"])
    outside = pd.Series(False, index=table.index)
    for i in INPUTS:
        if i["name"] != "volume_ratio":
            outside |= (table[i["name"]] < i["min"]) | (table[i["name"]] > i["max"])
    out_of_range = table[outside].copy()
    table = table[~outside]
    counts["dropped_out_of_range"] = int(len(out_of_range))
    notes.append(f"{counts['volume_capped']} days had volume outside 0.2-4x the average and were capped")
    notes.append(f"{len(out_of_range)} days dropped for a market move beyond +/-5%")

    # 6. Outliers that stay: a robust z-score of the residual from a plain market-only fit. A big residual
    #    is a real stock-specific event, so it is flagged, reported and kept rather than silently deleted.
    slope, intercept = np.polyfit(table["market_return_pct"], table[TARGET], 1)
    resid = table[TARGET] - (intercept + slope * table["market_return_pct"])
    centred = resid - resid.median()
    z = centred / (1.4826 * centred.abs().median())
    flagged = table[z.abs() > OUTLIER_Z].assign(residual_z=z[z.abs() > OUTLIER_Z].round(1))
    counts["flagged_outliers"] = int(len(flagged))
    counts["rows_used"] = int(len(table))
    notes.append(f"{len(flagged)} days flagged as stock-specific outliers (|robust z| > {OUTLIER_Z:g}) and kept")
    return table, Audit(counts, notes, flagged, out_of_range, aligned)


def make_data(refresh: bool = False) -> pd.DataFrame:
    return prepare(load_raw(refresh))[0]


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
    data: pd.DataFrame
    audit: Audit
    n_train: int


def train_and_evaluate(refresh: bool = False) -> Result:
    data, audit = prepare(load_raw(refresh))
    x, y = data[COLUMNS], data[TARGET].to_numpy()

    # Chronological split: train on the earlier 75% of days, test on the most recent 25%.
    n_train = int(len(data) * 0.75)
    x_train, x_test, y_train, y_test = x.iloc[:n_train], x.iloc[n_train:], y[:n_train], y[n_train:]

    pipeline = build_pipeline().fit(x_train, y_train)
    pred = pipeline.predict(x_test)

    # Reference points: always predicting the training mean, and a beta-only model with just the market return.
    mean_pred = np.full_like(y_test, y_train.mean())
    beta_only = LinearRegression().fit(x_train[["market_return_pct"]], y_train)
    metrics = {
        "r2": float(r2_score(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
        "mean_baseline_rmse": float(np.sqrt(mean_squared_error(y_test, mean_pred))),
        "market_only_r2": float(r2_score(y_test, beta_only.predict(x_test[["market_return_pct"]]))),
        "beta": float(beta_only.coef_[0]),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "train_period": f"{x_train.index.min().date()} to {x_train.index.max().date()}",
        "test_period": f"{x_test.index.min().date()} to {x_test.index.max().date()}",
    }

    model_bytes = to_onnx(pipeline)

    # Verify the ONNX model reproduces sklearn on the whole test set before anything is written.
    max_diff = float(np.abs(onnx_predict(model_bytes, x_test) - pred).max())
    assert max_diff < 1e-3, f"ONNX and sklearn disagree (max diff {max_diff})"
    metrics["onnx_max_abs_diff"] = max_diff
    return Result(pipeline, model_bytes, x_test, y_test, metrics, data, audit, n_train)


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

    # Data charts: what was downloaded, and what the audit flagged.
    prices = result.audit.prices
    rebased = prices / prices.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for col, label, colour in [("stock", "Reliance", ACCENT), ("market", "Nifty 50", MUTED)]:
        ax.plot(rebased.index, rebased[col], label=label, color=colour, linewidth=1.4)
    ax.set_yscale("log")
    ax.set_yticks([100, 200, 400, 800])
    ax.set_yticklabels(["100", "200", "400", "800"])
    ax.minorticks_off()
    ax.set_ylabel("Rebased to 100 (log scale)")
    ax.set_title(f"Closing prices, {prices.index.min().year} to {prices.index.max().year}")
    ax.legend(loc="upper left")
    save(fig, "price-history.png", "Reliance and Nifty 50 closing prices from Yahoo Finance, rebased to 100. Reliance has tracked the Nifty 50 with more swing.")

    data, audit = result.data, result.audit
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(data["market_return_pct"], data[TARGET], s=8, alpha=0.35, color=ACCENT, label="Kept")
    if len(audit.flagged):
        ax.scatter(audit.flagged["market_return_pct"], audit.flagged[TARGET], s=34, facecolors="none",
                   edgecolors="#dc2626", linewidths=1.3, label=f"Flagged outlier, kept ({len(audit.flagged)})")
    if len(audit.out_of_range):
        ax.scatter(audit.out_of_range["market_return_pct"], audit.out_of_range[TARGET], s=26, marker="x",
                   color="#d97706", label=f"Dropped, beyond +/-5% ({len(audit.out_of_range)})")
    ax.axhline(0, color="#d6d3d1", linewidth=0.8)
    ax.axvline(0, color="#d6d3d1", linewidth=0.8)
    ax.set_xlabel("Nifty 50 daily return (%)")
    ax.set_ylabel("Reliance daily return (%)")
    ax.set_title("Outliers and dropped days")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "outliers.png", "Daily Reliance return against the Nifty 50 return. Red circles are stock-specific outliers that were kept; orange crosses are crash and election-day moves outside the form's range.")

    lim = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(pred, y_test, s=10, alpha=0.45, color=ACCENT)
    ax.plot(lim, lim, "--", color=MUTED, label="Perfect prediction")
    ax.set_xlabel("Predicted return (%)")
    ax.set_ylabel("Actual return (%)")
    ax.set_title(f"Predicted vs actual (R² {result.metrics['r2']:.2f})")
    ax.legend(loc="upper left")
    save(fig, "predicted-vs-actual.png", "Predicted against actual daily returns on the most recent 25% of days.")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(pred, resid, s=10, alpha=0.45, color=ACCENT)
    ax.axhline(0, color=MUTED, linestyle="--")
    ax.set_xlabel("Predicted return (%)")
    ax.set_ylabel("Residual (%)")
    ax.set_title("Residuals vs predicted")
    save(fig, "residuals.png", "Residuals show no trend against the prediction, but a few days are missed by 4 to 8 points.")

    scaler = result.pipeline.named_steps["pre"].named_transformers_["num"]
    per_unit = result.pipeline.named_steps["reg"].coef_ / scaler.scale_
    labels = ["Market return (%)", "Volume ratio (x)"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh(labels, per_unit, color=ACCENT)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("Change in predicted return (%) per 1-unit increase")
    ax.set_title("Coefficients (original units)")
    save(fig, "coefficients.png", "Learned coefficients: the market coefficient is the stock's beta; volume adds a small positive effect.")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.hist(resid, bins=30, color=ACCENT, alpha=0.8)
    ax.set_xlabel("Residual (%)")
    ax.set_ylabel("Days")
    ax.set_title(f"Residual distribution (RMSE {result.metrics['rmse']:.2f}%)")
    save(fig, "residual-distribution.png", "Residuals are centred near zero with heavy tails: occasional stock-specific days the model cannot see.")

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
