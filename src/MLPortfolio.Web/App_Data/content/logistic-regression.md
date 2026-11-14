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

## Conclusion

Logistic regression gives a usable, explainable default estimate: an ROC AUC of 0.81 from a handful of inputs, with coefficients that point the right way (more debt and recent late payments raise the risk, longer history and higher income lower it). Its main strength is that every prediction can be explained, and the probability it outputs can be used directly. Its main weakness is visible in the accuracy figure: at 85.5% it is only slightly above the 83% you get by always predicting "Repaid", so the value is in ranking borrowers, not in the headline accuracy.

**What would improve it:**

- **Interactions and non-linear effects.** A straight-line score cannot capture that high debt is far worse when combined with recent late payments. Interaction terms, or a tree ensemble as a comparison, would show how much that costs.
- **Choosing the cut-off by cost.** A missed default usually costs far more than a declined good borrower, so the 50% threshold should be set from those costs, not left at the default.
- **Checking calibration.** The calibration chart is the test of whether "30%" really means 30%; recalibrating would tighten it if it drifts.
- **Real data and fairness checks.** The data here is invented. A real credit model would need real outcomes, a look at which groups are approved at different rates, and monitoring as the economy shifts.
