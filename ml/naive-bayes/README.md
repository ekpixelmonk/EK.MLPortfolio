# Gaussian naive Bayes: bank customer churn

Part of the ML Portfolio Model Zoo.

## Problem
Given a customer's age, balance, product count, tenure and recent activity, estimate the probability that they leave the bank.

## Data
Synthetic. `train.py` generates 6,000 customers (seed 42) from a known churn process. No customer data.

## Approach
scikit-learn pipeline: numeric columns passed through, one-hot encoding for activity, then `GaussianNB`. Exported to ONNX and checked against sklearn.

## How to run
Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```powershell
cd ml/naive-bayes
uv sync
uv run python train.py          # trains, exports, writes site artefacts
uv run jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb   # re-run the notebook
```

If `uv sync` fails with a certificate error, set `$env:UV_NATIVE_TLS = "1"` first.

`train.py` writes, relative to the repository root (override with `--repo-root`):
- `src/MLPortfolio.Web/App_Data/models/naive-bayes/{model.onnx, schema.json}`
- `src/MLPortfolio.Web/wwwroot/projects/naive-bayes/charts/*.png`
- `tests/fixtures/naive-bayes.json` (reference inputs/outputs used by the .NET parity test)

## Limitations
Naive Bayes assumes features are independent and bell-shaped; here they are neither, so its probabilities are over-confident (see the calibration chart). Educational purposes only.
