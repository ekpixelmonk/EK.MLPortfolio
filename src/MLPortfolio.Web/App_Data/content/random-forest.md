## Problem

How volatile will Reliance Industries be over the next month? Given what is known at today's close, this model forecasts the stock's annualised volatility over the next 20 trading days. Volatility tends to cluster (calm periods follow calm periods, turbulent follow turbulent), so recent volatility is informative, but it also fades and jumps, which makes a forecast harder than it looks.

## Data

Real prices, downloaded from Yahoo Finance: daily Reliance (`RELIANCE.NS`) prices and volume, the Nifty 50 (`^NSEI`) and India VIX (`^INDIAVIX`), from January 2015 to November 2026 (2,930 trading days). Volatility is measured with the Parkinson estimator, which uses each day's high-to-low range and is much less noisy than the usual close-to-close measure.

Eight inputs are computed at each close: Reliance's 5, 20 and 60-day volatility, its 20-day return, how far it sits below its six-month high, recent volume against normal, the Nifty 50's 20-day volatility and India VIX. The target is the volatility of the 20 days that follow.

Before modelling, the data was audited:

- **Missing data:** the Nifty 50 has no close on 9 days and India VIX on 20 days when Reliance traded (holidays and special sessions). Filling them would invent zero returns and understate volatility, so those days were dropped (26 in total).
- **Bad values:** no duplicate dates, no non-positive prices, and no zero or inverted high-low ranges. One day with zero volume was dropped.
- **Outliers:** 20 days were flagged as stock-specific outliers (a robust z-score of the daily return's residual above 5), including the 11.6% jump on 2 November 2026. For a volatility model the extreme days are the signal, so none were removed.
- **Form range:** 75 feature values (out of about 22,000) fell outside the ranges the form accepts and were capped, not dropped.

2,758 days remain. The raw download is not stored in the repository; `uv run python train.py --refresh` fetches it again.

## Approach

A random forest, trained to predict the *ratio* of future volatility to the 60-day level (as a logarithm) instead of the volatility itself. Volatility levels change between market regimes, and a forest cannot predict values outside what it has seen; a ratio is much more stable. The exported ONNX model converts the ratio back, so the site shows an ordinary percentage.

The split is chronological: trained on the earlier 75% of days (July 2015 to December 2023) and tested on the later days (January 2024 to October 2026), with 20 days skipped between the two because each target looks 20 days ahead. Tree depth and leaf size were chosen on the training data only, with a time-aware cross-validation, and came out shallow (depth 3, 80 days per leaf): there are only a handful of independent volatility episodes in ten years, and deeper trees memorise them.

## Results

On the test period the forest's average error is **3.9 volatility points** (RMSE; MAE 3.1). For comparison:

| Forecast | RMSE | MAE |
|---|---|---|
| Training-period average | 5.92 | 5.21 |
| Last 60 days' volatility | 4.31 | 3.28 |
| Last 20 days' volatility | 4.26 | 3.18 |
| Linear regression, same inputs | 3.86 | 3.18 |
| **Random forest** | **3.93** | **3.12** |

The forest beats simply carrying today's volatility forward by about 8%, and the training average by a third. It does not beat a plain linear regression: the two are a statistical tie, with the forest slightly better on MAE and slightly worse on RMSE. The 60-day level matters most; drawdown, India VIX and volume add a little. R² is negative for every method here (the forest's is -0.16), because the test period was unusually calm (average 18% against 23% in training) and varied little, so any error in the level is heavily penalised; the RMSE comparison is the fairer one.

## Limitations

Volatility here is near-linear, so a forest finds little a straight line misses. It also cannot foresee sudden shocks, such as an election result or a company announcement, which drive the biggest errors. One stock, about ten years, and only a few independent episodes of high volatility. The inputs must be computed the same way as in training (high-low range volatility over 5, 20 and 60 days). This is not a trading signal. Educational purposes only, not investment advice.

## Conclusion

A random forest can forecast Reliance's next-month volatility roughly 8% better than repeating today's level, but on this data it is no better than linear regression, and it is much harder to explain. The most useful lesson was in the setup rather than the model: switching to a range-based volatility measure and forecasting a ratio to the 60-day level did more for the result than any forest setting, and the first attempt, which predicted the raw level, did worse than guessing the training average.

**What would improve it:**

- **More stocks.** One stock gives only a few independent volatility episodes. Training on many NSE stocks would give a forest more to learn from and is where it should start to beat a straight line.
- **A volatility model built for the job,** such as GARCH or HAR, as a stronger baseline than a plain regression, and as inputs to the forest.
- **Event information.** Earnings dates, index rebalancing and election dates explain many of the largest misses.
- **Intraday data.** Realised volatility from intraday prices is more accurate than from daily ranges.
- **A different horizon or target.** Predicting whether volatility will rise or fall (a classification) may be more learnable than predicting its level.
