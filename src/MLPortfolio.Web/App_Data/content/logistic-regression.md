## Problem

A lender wants to know how likely a borrower is to default. Given income, debt-to-income ratio, credit history, recent late payments and employment type, the model estimates the probability of default and classifies the borrower as `Repaid` or `Default`.

## Data

Synthetic. 6,000 borrowers are generated from a known default process (seed 42), so there is nothing sensitive and the learned coefficients can be checked against the truth. About 17% of the borrowers default. No real borrower data is used.

## Approach

A scikit-learn pipeline: standardise the numeric features, one-hot encode employment, and fit a logistic regression on 75% of the data. The whole pipeline is exported to ONNX, checked against scikit-learn on the held-out set (probabilities agree to within 1e-7), and run inside this site.

## Results

On the held-out 25%, the model reaches an ROC AUC of **0.81** and 85.5% accuracy. Accuracy needs context: always predicting "Repaid" would already be right about 83% of the time, so AUC and the calibration chart say more. Debt-to-income ratio and recent late payments push the prediction toward default most strongly; a longer credit history and higher income pull it the other way.

## Limitations

The data is synthetic, so the model illustrates the technique and says nothing about real borrowers. A real credit model would also need fairness checks and careful treatment of the base rate. Educational purposes only.
