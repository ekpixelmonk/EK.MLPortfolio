# Decision tree: loan approval rules

Part of the ML Portfolio Model Zoo.

## Problem
Given a loan applicant's credit score, income, loan size, job tenure and collateral, predict whether the application is approved or rejected, using a tree whose rules can be read.

## Data
Synthetic. `train.py` generates 5,000 applications from a banker's rule of thumb with 6% of decisions flipped as noise (seed 42). No real applicant data.

## Approach
scikit-learn pipeline: numeric columns passed through (trees need no scaling), one-hot encoding for collateral, then `DecisionTreeClassifier(max_depth=4, min_samples_leaf=20)`. Exported to ONNX and checked against sklearn on the full test set.

## How to run
Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```powershell
cd ml/decision-tree
uv sync
uv run python train.py          # trains, exports, writes site artefacts
uv run jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook.ipynb   # re-run the notebook
```

If `uv sync` fails with a certificate error, set `$env:UV_NATIVE_TLS = "1"` first.

`train.py` writes, relative to the repository root (override with `--repo-root`):
- `src/MLPortfolio.Web/App_Data/models/decision-tree/{model.onnx, schema.json}`
- `src/MLPortfolio.Web/wwwroot/projects/decision-tree/charts/*.png`
- `tests/fixtures/decision-tree.json` (reference inputs/outputs used by the .NET parity test)

## Limitations
The data is synthetic and the 'rules' are invented, so the tree illustrates the technique, not real lending policy. Educational purposes only.
