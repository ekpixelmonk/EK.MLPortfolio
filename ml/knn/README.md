# k-nearest neighbours: stock style classifier

Part of the ML Portfolio Model Zoo.

## Problem
Given a stock's price/earnings ratio, dividend yield and revenue growth, classify it as Value, Growth or Income by looking at the most similar stocks.

## Data
Synthetic. `train.py` draws 3,000 stocks from three overlapping clusters (seed 42). No market data.

## Approach
scikit-learn pipeline: `StandardScaler` (essential for distance-based models) then `KNeighborsClassifier(n_neighbors=15)`. Exported to ONNX and checked against sklearn.

## How to run
Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```powershell
cd ml/knn
uv sync
uv run python train.py          # trains, exports, writes site artefacts
uv run jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb   # re-run the notebook
```

If `uv sync` fails with a certificate error, set `$env:UV_NATIVE_TLS = "1"` first.

`train.py` writes, relative to the repository root (override with `--repo-root`):
- `src/MLPortfolio.Web/App_Data/models/knn/{model.onnx, schema.json}`
- `src/MLPortfolio.Web/wwwroot/projects/knn/charts/*.png`
- `tests/fixtures/knn.json` (reference inputs/outputs used by the .NET parity test)

## Limitations
The clusters are invented; real stocks do not fall into three tidy groups. The whole training set is part of the model, so large datasets make KNN slow. Educational purposes only.
