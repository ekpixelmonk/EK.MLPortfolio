# Random forest: Reliance volatility forecast

Part of the ML Portfolio Model Zoo.

## Problem
Forecast Reliance Industries' annualised volatility over the next 20 trading days from what is known at today's close.

## Data
Real daily prices from Yahoo Finance via `yfinance`: `RELIANCE.NS`, `^NSEI` (Nifty 50) and `^INDIAVIX` (India VIX), from 2015-01-01. The raw download is cached in `data/raw_yahoo.csv`, which is git-ignored because Yahoo's terms do not allow redistributing their data. `train.py` downloads it on first run; use `--refresh` to fetch a fresh copy. (The download and audit code is a deliberate copy of the one in `ml/linear-regression`; projects here do not import from each other.)

`prepare()` in `train.py` audits and cleans the data, and prints what it found:
- days where the index or VIX has no close are dropped, never forward-filled;
- zero-volume days and zero or inverted high-low ranges are dropped;
- stock-specific outliers (robust z-score of the daily return's residual above 5) are flagged and kept, because for volatility the extreme days are the signal;
- features are capped at the site form's ranges (rows are kept).

Volatility is the Parkinson high-low estimator. Inputs: 5, 20 and 60-day Reliance volatility, 20-day return, drawdown from the six-month high, recent volume ratio, the Nifty 50's 20-day volatility and India VIX.

## Approach
`RandomForestRegressor` (300 trees) on a log target: `log(next-20-day volatility / 60-day volatility)`. The ONNX graph appends `exp()` and a multiply by the 60-day volatility input, so the model outputs a percentage. Chronological split: earlier 75% train, a 20-day gap, then test. Depth and leaf size are chosen by time-series cross-validation on the training data only (depth 3, 80 days per leaf won). Baselines: training average, last 20-day and 60-day volatility, and a linear regression on the same inputs.

## How to run
Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```powershell
cd ml/random-forest
uv sync
uv run python train.py              # trains on the cached data, exports, writes site artefacts
uv run python train.py --refresh    # downloads fresh prices first
uv run jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb   # re-run the notebook
```

If `uv sync` fails with a certificate error, set `$env:UV_NATIVE_TLS = "1"` first. On Windows, `train.py` builds a CA bundle from the system certificate store for the Yahoo download (verification stays on); set `CURL_CA_BUNDLE` yourself to override.

`train.py` writes, relative to the repository root (override with `--repo-root`):
- `src/MLPortfolio.Web/App_Data/models/random-forest/{model.onnx, schema.json}`
- `src/MLPortfolio.Web/wwwroot/projects/random-forest/charts/*.png`
- `tests/fixtures/random-forest.json` (reference inputs/outputs used by the .NET parity test)

## Findings
Test RMSE 3.93 volatility points (MAE 3.12), against 4.26 for repeating the last 20 days and 3.86 for a linear regression. The forest beats persistence by about 8% but only ties linear regression: volatility is close to linear and one stock has few independent episodes. R2 is negative for every method (calm test period). Predicting the raw level instead of the ratio scored worse than the training average.

## Limitations
One stock, one decade. Not a trading signal. Educational purposes only, not investment advice.
