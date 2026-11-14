# Logistic regression: loan default classifier

Part of the ML Portfolio Model Zoo.

## Problem
Given a borrower's income, debt-to-income ratio, credit history, recent late payments and employment type, estimate the probability that they default, and classify them as `Repaid` or `Default`.

## Data
Synthetic. `train.py` generates 6,000 borrowers from a known default process (seed 42), so there is nothing sensitive and the learned coefficients can be checked against the truth. No real borrower data is used.

## Approach
scikit-learn pipeline: `StandardScaler` on the numeric columns, one-hot encoding for employment, then `LogisticRegression`. The whole pipeline is exported to ONNX (one input per column) and checked against sklearn on the full test set before anything is written.

## How to run
Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```powershell
cd ml/logistic-regression
uv sync
uv run python train.py          # trains, exports, writes site artefacts
uv run jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb   # re-run the notebook
```

`train.py` writes, relative to the repository root (override with `--repo-root`):
- `src/MLPortfolio.Web/App_Data/models/logistic-regression/{model.onnx, schema.json}`
- `src/MLPortfolio.Web/wwwroot/projects/logistic-regression/charts/*.png`
- `tests/fixtures/logistic-regression.json` (reference inputs/outputs used by the .NET parity test)

## Limitations
The data is synthetic, so the model illustrates the technique and says nothing about real borrowers. Educational purposes only.
