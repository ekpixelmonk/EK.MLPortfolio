# Linear regression: stock daily return from market signals

Part of the ML Portfolio Model Zoo.

## Problem
Predict a stock's daily return (%) from the market's return, its sector's return and unusual trading volume. The market-return coefficient is the stock's beta.

## Data
Synthetic. `train.py` generates 3,000 trading days from a known relationship (market beta about 1.15, plus noise; seed 42), so the learned coefficients can be checked against the truth. No real prices are used.

## Approach
scikit-learn pipeline: `StandardScaler` then ordinary least squares `LinearRegression`. The pipeline is exported to ONNX (one float input per feature) and checked against sklearn on the full test set before anything is written.

## How to run
Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```powershell
cd ml/linear-regression
uv sync
uv run python train.py          # trains, exports, writes site artefacts
uv run jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb   # re-run the notebook
```

`train.py` writes, relative to the repository root (override with `--repo-root`):
- `src/MLPortfolio.Web/App_Data/models/linear-regression/{model.onnx, schema.json}`
- `src/MLPortfolio.Web/wwwroot/projects/linear-regression/charts/*.png`
- `tests/fixtures/linear-regression.json` (reference inputs/outputs used by the .NET parity test)

## Limitations
The data is synthetic and the relationship is exactly linear by construction. Real returns are noisier, change over time and are not predictable from same-day information. Educational purposes only, not investment advice.
