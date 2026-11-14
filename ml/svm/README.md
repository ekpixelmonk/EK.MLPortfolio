# Support vector machine: card fraud detection

Part of the ML Portfolio Model Zoo.

## Problem
Given a card transaction's amount, hour, distance from home, recent transaction count and channel, estimate the probability that it is fraud.

## Data
Synthetic. `train.py` generates 5,000 transactions (seed 42) from a known risk process with roughly 15% fraud. No real card data.

## Approach
scikit-learn pipeline: `StandardScaler` + one-hot encoding, then an RBF `SVC` with Platt-scaled probabilities. Exported to ONNX and checked against sklearn (probabilities within 2e-3, same predicted class).

## How to run
Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```powershell
cd ml/svm
uv sync
uv run python train.py          # trains, exports, writes site artefacts
uv run jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb   # re-run the notebook
```

If `uv sync` fails with a certificate error, set `$env:UV_NATIVE_TLS = "1"` first.

`train.py` writes, relative to the repository root (override with `--repo-root`):
- `src/MLPortfolio.Web/App_Data/models/svm/{model.onnx, schema.json}`
- `src/MLPortfolio.Web/wwwroot/projects/svm/charts/*.png`
- `tests/fixtures/svm.json` (reference inputs/outputs used by the .NET parity test)

## Limitations
SVM probabilities are a fitted add-on, not native outputs, and fraud data is far more imbalanced in real life. Synthetic data only. Educational purposes only.
