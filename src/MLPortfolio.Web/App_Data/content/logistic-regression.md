## Problem

Predict a yes/no outcome, here whether a student passes an exam, from a handful of numeric and categorical features, and report how confident the model is.

## Data

A synthetic dataset generated for this project, so there is nothing sensitive and the pattern is known. The data source will be stated here once the project is built.

## Approach

A scikit-learn pipeline: scale the numeric features, encode the categorical one, and fit a logistic regression. The pipeline is exported to ONNX so the site runs the same model that was trained.

## Results

Coming with the project: accuracy, a confusion matrix, a calibration curve and the learned coefficients.

## Limitations

The data is synthetic, so the model illustrates the technique and says nothing about real students.
