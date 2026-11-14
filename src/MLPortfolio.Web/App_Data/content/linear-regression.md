## Problem

How much does Reliance Industries move when the market moves? This model predicts Reliance's daily return (%) from the Nifty 50's return, the Nifty Energy index's return, and how unusual the day's trading volume is.

## Data

Real prices, downloaded from Yahoo Finance: daily closes for `RELIANCE.NS`, the Nifty 50 (`^NSEI`) and the Nifty Energy index (`^CNXENERGY`), from January 2015 to November 2026 (2,930 trading days). Yahoo's closes are already adjusted for Reliance's bonus issues. Volume is measured against the average of the previous 20 days.

Before modelling, the data was audited:

- **Missing data:** the indices have no close on 25 days when Reliance traded (holidays and special sessions such as Diwali Muhurat trading). Filling them would invent zero returns, so those days were dropped, and returns are computed between aligned dates only.
- **Bad values:** no duplicate dates and no non-positive prices. One day with zero volume was dropped (missing, not "nobody traded"), and one flat print on a day the market moved was dropped as stale.
- **Out-of-range inputs:** 21 days with a market or sector move beyond ±5% (the March 2020 crash, the June 2024 election result) were dropped to match the form's range. 21 days with volume outside 0.2 to 4 times normal were capped, not dropped.
- **Outliers:** 17 days were flagged as stock-specific outliers (a robust z-score of the residual above 5), such as a +11.6% jump on 2 November 2026 on heavy volume while the market rose 1.7%. They are real events, so they were kept and shown on the outlier chart.

2,862 days remain. The raw download is not stored in the repository; `uv run python train.py --refresh` fetches it again.

## Approach

Ordinary least squares in a scikit-learn pipeline with feature standardisation. The split is chronological: trained on the earlier 75% of days (February 2015 to November 2023) and tested on the most recent 25% (December 2023 to November 2026), because a random split would let the model peek at the future. The pipeline is exported to ONNX, checked against scikit-learn on the test days, and run inside this site.

## Results

On the test period, R² is **0.32**, with an RMSE of 1.12 percentage points against 1.36 for always predicting the average. A simpler model using only the Nifty 50 return does better, at R² 0.41, with a beta of about 1.02.

Two things explain it. The Nifty Energy index is heavily weighted towards Reliance itself, so its return partly echoes the target: it takes most of the weight in the model and the market coefficient shrinks to nearly zero. And that link weakens over time as index weights are capped, which is why the model trained on earlier years over-predicts large moves recently (the residual chart slopes downward). More inputs did not mean a better model.

## Limitations

Same-day market and sector returns are only known once the day is over, so this describes a relationship and cannot forecast tomorrow. One stock over one period, a linear model, and a sector input that overlaps the target. Educational purposes only, not investment advice.
