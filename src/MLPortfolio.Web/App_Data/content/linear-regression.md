## Problem

How much does a stock move when the market moves? This model predicts a stock's daily return (%) from the market's return, its sector's return, and how unusual the day's trading volume is. The coefficient on the market return is the stock's beta.

## Data

Synthetic. 3,000 trading days are generated from a known relationship (market beta about 1.15, plus a sector effect, a small volume effect and noise; seed 42), so the learned coefficients can be checked against the truth. No real prices are used.

## Approach

Ordinary least squares in a scikit-learn pipeline with feature standardisation, trained on 75% of the days. The pipeline is exported to ONNX, checked against scikit-learn on the held-out days (predictions agree to within 1e-6), and run inside this site.

## Results

On the held-out 25%, R² is **0.78**, with an RMSE of 0.81 percentage points and an MAE of 0.65. The residuals have no visible pattern and are roughly bell-shaped, which is what a well-specified linear model should show. The learned coefficients recover the market beta and the smaller sector and volume effects used to generate the data.

## Limitations

The relationship is exactly linear by construction. Real returns are noisier, their relationship to the market changes over time, and same-day information is not available before the day happens. Educational purposes only, not investment advice.
