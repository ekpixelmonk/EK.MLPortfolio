## Problem

Predict a continuous number from numeric features by fitting the best straight line (or plane) through the data.

## Data

A synthetic dataset generated for this project, so the true relationship is known and the fit can be checked against it. The data source will be stated here once the project is built.

## Approach

Ordinary least squares with scikit-learn, with residual diagnostics to check the assumptions. The fitted model is exported to ONNX so the site runs the same model that was trained.

## Results

Coming with the project: R², error metrics, a fitted-line chart and a residual plot.

## Limitations

A linear model cannot capture curved relationships, and it is sensitive to outliers.
