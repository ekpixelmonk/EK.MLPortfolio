## Problem

Markets move through different regimes: calm uptrends, choppy sideways periods and high-volatility sell-offs. Knowing which regime the market is in helps frame risk. This project classifies NIFTY market regimes from trend and volatility features.

## Data

Historical NSE/NIFTY index data. The exact source and data cutoff date will be stated here once the project is built.

## Approach

Engineer trend-strength and rolling-volatility features, label regimes, and train a classifier. The model is exported to ONNX and run inside this site.

## Results

Coming with the project: a regime timeline, a confusion matrix and feature importance.

## Limitations

Regimes are a modelling construct, not a forecast. Past behaviour does not predict future returns.
