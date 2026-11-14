# NSE market regime classifier

Part of the ML Portfolio (flagship). Work in progress: data, labels, features, EDA and model are built, and the project is live on the site without a form: the page states the model's reading of the latest day. The catalog entry stays `coming-soon`.

## Data
Daily history from Yahoo Finance via `yfinance`: Nifty 200 (`^CNX200`, from 2007-09-03) and India VIX (`^INDIAVIX`, from 2008-03-03). The raw download is cached in `data/raw_yahoo.csv`, which is git-ignored because Yahoo's terms do not allow redistributing their data.

## How to run
Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```powershell
cd ml/nse-regime
uv sync
uv run python data.py            # uses the cached download, or fetches it the first time; prints a quality summary
uv run python data.py --refresh  # downloads a fresh copy
```

If `uv sync` fails with a certificate error, set `$env:UV_NATIVE_TLS = "1"`. On Windows, `data.py` builds a CA bundle from the system certificate store for the Yahoo download (verification stays on).

## Data check (2026-11-14 download)
4,724 rows, 2007-09-03 to 2026-11-13. No duplicate dates and no non-positive closes. India VIX has 143 missing closes, almost all before it starts on 2008-03-03; 4,581 days have both series. Nifty 200 volume is present (up to 8 billion shares a day) but not yet checked for gaps or zero days; India VIX has no volume (all zeros), as expected.

## Regime labels
`labels.py` defines four regimes from the Nifty 200, trend x volatility: trailing 63-day return above or below zero, and 20-day realised volatility (annualised) above or below 18%.

| Regime | Trend | Volatility | Share of days |
|---|---|---|---|
| Calm rally | up | low | 51.9% |
| Volatile rally | up | high | 10.6% |
| Grind down | down | low | 19.5% |
| Stress | down | high | 18.0% |

The thresholds are fixed numbers (18% is about the 70th percentile), so a label never depends on the future. A switch is confirmed only after the new regime holds for 5 trading days (also causal): this cuts 280 switches to 103 (average run 17 to 45 days). Sanity check: 2008 is 92% Stress, 2014 and 2017 are 94% and 90% Calm rally, March 2020 is Stress, and the 2021 recovery shows as Volatile rally. The label is the current regime, so the model must not use the two measurements above as inputs (it would only re-derive its label); VIX, drawdown, distance from the 200-day average and short-term returns are candidates.

```powershell
uv run python labels.py   # prints the summary, writes data/labels.csv and data/regime_timeline.png
```

## Features
`features.py` audits the data and builds seven features (all use only data up to the day, none is the 63-day return or 20-day volatility behind the labels): `india_vix`, `vix_vs_avg_pct` (VIX against its 60-day average), `vix_change_5d_pct`, `drawdown_pct` (below the 252-day high), `dist_200dma_pct`, `ma50_vs_ma200_pct` and `return_5d_pct`. The site write-up explains each one in detail.

Audit findings: the Nifty 200 high equals the low on 1,006 days (1,005 before 2012) and volume is zero on 1,868 days (1,862 before mid-2015), so the high-low range and volume features first proposed were dropped (keeping them would have removed 2008 to 2011, including the 2008 crash); `vix_change_5d_pct` replaced them. VIX has no close for 126 days before its launch and 17 after (dropped, never filled); 18 moves beyond 6% are real events and kept; 72 warm-up days dropped. Result: 4,509 days from 2008-06-23 to 2026-11-13 (Calm rally 2,408, Volatile rally 452, Grind down 908, Stress 741).

```powershell
uv run python features.py   # prints the audit, feature summary, per-regime medians and correlations
```

## Model
`train.py` evaluates four candidates walk-forward (6 expanding folds, 63-day gap, 3,864 out-of-sample days from 2011-02-10) and exports the class-weighted logistic regression, fitted on all 4,509 days, to ONNX (seven float [1,1] inputs, one per feature; class probabilities out). It writes `model.onnx`, `schema.json`, six charts and `tests/fixtures/nse-regime.json` into the web app folders; it also writes `snapshot.json` (the latest day's seven feature values) and `tests/fixtures/nse-regime-snapshot.json`. There is no prediction form: the site runs the exported ONNX model on the snapshot and shows "The model predicts the current regime as: ... (illustrative)". Re-run `train.py` after refreshing the data to update the reading.

| Model | Accuracy | Balanced accuracy | ROC AUC |
|---|---|---|---|
| Always the most common regime | 55.4% | 30.1% | n/a |
| Logistic regression | 68.2% | 48.7% | 0.844 |
| **Logistic regression, balanced classes** | **69.5%** | **56.2%** | **0.861** |
| Random forest | 68.9% | 48.3% | 0.811 |
| Gradient boosting | 65.0% | 46.8% | 0.774 |

Per-regime recall: Grind down 86%, Calm rally 74%, Stress 50%, Volatile rally 15%. Beats the simple guess in 5 of 6 test periods (not 2016-2018). A model given yesterday's regime would score 98%, a different and easier task. `uv run python train.py [--refresh]` re-runs everything.

## EDA and training stats
`eda.py` writes 12 exploratory charts (`eda-*.png`) into the web app's chart folder: index and VIX history with regimes, missing data by year, daily-return distribution, the label rule scatter, regimes by year and spell length, regime transitions, feature history, feature distributions (with the form ranges), features by regime, feature pairs, correlation and mutual information. The regime colours (green, amber, blue, red) pass the dataviz colour-blind check; the earlier grey failed the chroma floor. `uv run python eda.py` regenerates them.

`uv run python train.py --report` prints the training-stats tables used on the project page (folds, settings, per-regime precision and recall, coefficients, log loss and Brier score), so the page numbers come from the code. Both are embedded inline in `content/nse-regime.md` (Data, Features, Model training and Results); `projects.json` is unchanged while the entry is `coming-soon`.

## Limitations
Educational purposes only, not investment advice.
