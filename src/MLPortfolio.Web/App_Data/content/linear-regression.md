## Problem

How much does Reliance Industries move when the market moves? This model predicts Reliance's daily return (%) from the Nifty 50's return and how unusual the day's trading volume is. The coefficient on the market return is the stock's beta.

## Data

Real prices, downloaded from Yahoo Finance: daily closes for `RELIANCE.NS` and the Nifty 50 (`^NSEI`), plus Reliance's volume, from January 2015 to November 2026 (2,930 trading days). Yahoo's closes are already adjusted for Reliance's bonus issues. Volume is measured against the average of the previous 20 days.

Before modelling, the data was audited:

- **Missing data:** the Nifty 50 has no close on 9 days when Reliance traded (holidays and special sessions such as Diwali Muhurat trading). Filling them would invent zero returns, so those days were dropped, and returns are computed between aligned dates only.
- **Bad values:** no duplicate dates and no non-positive prices. One day with zero volume was dropped (missing, not "nobody traded"), and one flat print on a day the market moved was dropped as stale.
- **Out-of-range inputs:** 11 days with a market move beyond ±5% (the March 2020 crash, the June 2024 election result) were dropped to match the form's range. 21 days with volume outside 0.2 to 4 times normal were capped, not dropped.
- **Outliers:** 19 days were flagged as stock-specific outliers (a robust z-score of the residual above 5), such as a +11.6% jump on 2 November 2026 on heavy volume while the market rose 1.7%. They are real events, so they were kept and shown on the outlier chart.

2,888 days remain. The raw download is not stored in the repository; `uv run python train.py --refresh` fetches it again.

## Approach

Ordinary least squares in a scikit-learn pipeline with feature standardisation. The split is chronological: trained on the earlier 75% of days (February 2015 to December 2023) and tested on the most recent 25% (December 2023 to November 2026), because a random split would let the model peek at the future. The pipeline is exported to ONNX, checked against scikit-learn on the test days, and run inside this site.

An earlier version also used the Nifty Energy index as a sector input. It was dropped: that index is dominated by Reliance, so its return partly echoes the target, and its link to Reliance weakened as index weights were capped. It scored worse on the test period (R² 0.32) than this simpler model.

## Results

On the test period, R² is **0.42**, with an RMSE of 1.06 percentage points against 1.39 for always predicting the average. Beta is about 1.07: Reliance has moved a little more than the market. Volume adds almost nothing on its own (a market-only fit scores 0.42 too), though it gets a small positive coefficient. The relationship has been fairly stable: fitting two-year windows gives betas between 0.9 and 1.2.

The residuals show no trend against the prediction, but they have heavy tails: a few days are missed by 4 to 8 points because of news about Reliance itself, which no market input can capture.

## Limitations

Same-day market returns are only known once the day is over, so this describes a relationship and cannot forecast tomorrow. One stock over one period, and a linear model that leaves most of the day-to-day variation (about 58%) unexplained. Educational purposes only, not investment advice.

## Conclusion

Reliance moves roughly one for one with the Nifty 50, a little more than the market (beta about 1.07), and that relationship has held up for a decade. A single straight line explains about 42% of its daily moves, which is a respectable result for one input, and the rest is mostly news about Reliance itself. The model describes how Reliance and the market move together; it is not a tool for predicting the stock.

The audit mattered as much as the model. Dropping unaligned days, treating crash days separately and keeping real outliers visible changed what the numbers meant, and the first attempt, which added a sector index that contained the stock itself, scored worse than the simpler model that replaced it.

**What would improve it:**

- **Lagged inputs.** Today's market return is not known in advance. Using yesterday's market, sector and volume would make it a real forecast, though a much weaker one.
- **A sector index that excludes Reliance,** or a peer group of energy stocks, so the sector signal is independent of the target.
- **A model that allows for changing volatility.** Residuals are heavy-tailed and calmer in quiet periods, so a rolling beta or a volatility model such as GARCH would fit better than one fixed line.
- **Event days.** Results days and company announcements cause the largest misses. A flag for them, or a robust fit that gives them less weight, would help.
- **More stocks and periods,** to check that beta here is typical and not specific to Reliance or to 2015 to 2026.
