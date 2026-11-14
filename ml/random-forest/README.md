# Random forest: stock volatility

Part of the ML Portfolio Model Zoo.

## Problem
Given a stock's beta, debt-to-equity ratio, market cap and sector, predict its 30-day annualised volatility (%).

## Data
Synthetic. `train.py` generates 4,000 stocks (seed 42). Volatility has threshold effects (leverage above 2x), interactions (beta matters more in some sectors) and a small-cap premium, so a straight line misses structure the forest can find. No market data.

## Approach
scikit-learn pipeline: numeric columns passed through, one-hot encoding for sector, then `RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=5)`. A plain linear regression on the same inputs is the baseline. Exported to ONNX and checked against sklearn on the full test set.

## How to run
Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```powershell
cd ml/random-forest
uv sync
uv run python train.py          # trains, exports, writes site artefacts
uv run jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb   # re-run the notebook
```

If `uv sync` fails with a certificate error, set `$env:UV_NATIVE_TLS = "1"` first.

`train.py` writes, relative to the repository root (override with `--repo-root`):
- `src/MLPortfolio.Web/App_Data/models/random-forest/{model.onnx, schema.json}`
- `src/MLPortfolio.Web/wwwroot/projects/random-forest/charts/*.png`
- `tests/fixtures/random-forest.json` (reference inputs/outputs used by the .NET parity test)

## Limitations
The data is synthetic and the volatility formula is invented, so this illustrates the technique, not real risk. A forest cannot extrapolate beyond the ranges it saw. Educational purposes only.
