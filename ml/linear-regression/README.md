# Linear regression: Reliance daily return

Part of the ML Portfolio Model Zoo.

## Problem
Predict Reliance Industries' daily return (%) from the Nifty 50's return and unusual trading volume.

## Data
Real daily prices from Yahoo Finance via `yfinance`: `RELIANCE.NS` and `^NSEI` (Nifty 50), from 2015-01-01. The raw download is cached in `data/raw_yahoo.csv`, which is git-ignored because Yahoo's terms do not allow redistributing their data. `train.py` downloads it on first run; use `--refresh` to fetch a fresh copy.

`prepare()` in `train.py` audits and cleans the data, and prints what it found:
- indices with no close on days Reliance traded are dropped, never forward-filled (that would invent zero returns);
- zero-volume days and stale flat prints are dropped;
- market moves beyond +/-5% (crash and election days) are dropped to match the site form's range, and volume is capped at 0.2-4x;
- stock-specific outliers (robust z-score of the residual above 5) are flagged and kept, because they are real events.

## Approach
scikit-learn pipeline: `StandardScaler` then ordinary least squares `LinearRegression`. Chronological split: the earlier 75% of days train, the latest 25% test. The pipeline is exported to ONNX (one float input per feature) and checked against sklearn on the full test set before anything is written.

## How to run
Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```powershell
cd ml/linear-regression
uv sync
uv run python train.py              # trains on the cached data, exports, writes site artefacts
uv run python train.py --refresh    # downloads fresh prices first
uv run jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb   # re-run the notebook
```
On Windows, `train.py` builds a CA bundle from the system certificate store for the download (verification stays on). Set `CURL_CA_BUNDLE` yourself to override.

`train.py` writes, relative to the repository root (override with `--repo-root`):
- `src/MLPortfolio.Web/App_Data/models/linear-regression/{model.onnx, schema.json}`
- `src/MLPortfolio.Web/wwwroot/projects/linear-regression/charts/*.png`
- `tests/fixtures/linear-regression.json` (reference inputs/outputs used by the .NET parity test)

## Findings
Held-out R2 is 0.42 (beta about 1.07); volume adds almost nothing over a market-only fit (0.42). A first version also used the Nifty Energy index and scored 0.32: that index is dominated by Reliance, so it partly contains the target, and its link weakened as index weights were capped. It was dropped.

## Limitations
One stock, one period, a linear model. Same-day returns are only known after the day, so this describes a relationship rather than forecasting. Educational purposes only, not investment advice.
