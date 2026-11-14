## Problem

An analyst wants a quick estimate of how jumpy a stock will be over the next month. Given the stock's beta, debt-to-equity ratio, market cap and sector, the model predicts its 30-day annualised volatility in percent.

## Data

Synthetic. 4,000 stocks are generated with effects a straight line cannot capture: volatility jumps once debt-to-equity passes 2, small caps are jumpier, and beta matters more in technology and energy than elsewhere. Noise is added on top. No market data is used.

## Approach

A scikit-learn pipeline: numeric features pass through, sector is one-hot encoded, and a `RandomForestRegressor` (100 trees, depth at most 8, at least 5 samples per leaf) is fitted on 75% of the data. A plain linear regression on the same inputs is the baseline. The forest is exported to ONNX (about 0.9 MB), checked against scikit-learn on the held-out set, and run inside this site.

## Results

On the held-out 25%, the forest reaches an R² of **0.85** with an RMSE of 2.6 percentage points. The linear baseline gets 0.80 and 3.0, so the forest earns its extra complexity by picking up the leverage threshold and the sector-specific beta effect. The beta chart shows the flattening at the extremes: a forest cannot extrapolate beyond what it saw in training.

## Limitations

The volatility formula is invented, so this illustrates the technique, not real risk. Forests are harder to explain than a single tree and predict a step function, not a smooth curve. Not investment advice. Educational purposes only.
