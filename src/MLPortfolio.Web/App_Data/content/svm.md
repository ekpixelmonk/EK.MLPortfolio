## Problem

A card issuer wants to flag suspicious transactions. Given the amount, the hour, the distance from the cardholder's home, how many transactions happened in the last hour and the channel (in-store, online or ATM), the model estimates the probability that a transaction is fraud.

## Data

Synthetic. 5,000 transactions are generated from a known risk process: fraud is more likely for large amounts, night hours, far-from-home purchases, bursts of activity and online use. About 15% of transactions are fraud, far higher than in real life, to keep the example learnable. No real card data is used.

## Approach

A scikit-learn pipeline: standardise the numeric features (essential for SVMs), one-hot encode the channel, and fit an RBF-kernel support vector machine with Platt-scaled probabilities on 75% of the data. The pipeline is exported to ONNX, checked against scikit-learn on the held-out set (probabilities agree to within 2e-3), and run inside this site.

## Results

On the held-out 25%, the model reaches an ROC AUC of **0.75** and 85.8% accuracy. Accuracy needs context: always predicting "Legitimate" would already be right about 85% of the time, so AUC is the fairer measure, and this is a modest result because the synthetic fraud signal is noisy. The probability surface shows the RBF kernel's flexible boundary; it is least trustworthy at the edges where little training data exists, such as very large amounts.

## Limitations

The data is synthetic and heavily simplified, and real fraud is rarer and shifts over time. SVM probabilities are a fitted add-on rather than a native output. Educational purposes only.

## Conclusion

The support vector machine's fraud detector is only modestly useful here: an ROC AUC of 0.75, and 85.8% accuracy against 85.0% for always predicting "Legitimate". The lesson is that accuracy is misleading when one class dominates, and that AUC is the fairer measure. On this noisy synthetic signal the RBF kernel's flexibility buys little, and the probabilities are a fitted add-on (Platt scaling) rather than something the model produces natively.

**What would improve it:**

- **Tuning.** The kernel width and penalty were not searched. Cross-validating them could improve the AUC, and they matter a great deal for an SVM.
- **Metrics that fit the problem.** Fraud is rare, so precision and recall at a chosen threshold, and the cost of a missed fraud against a false alarm, say more than accuracy.
- **Handling the imbalance.** Class weights, or resampling, would make the model pay more attention to the rare fraud cases.
- **Better inputs and a comparison.** Features like spending patterns over time and merchant type usually matter more than the model, and gradient boosting is the usual strong baseline for tabular fraud data; SVMs also scale poorly to millions of transactions.
- **Realistic data.** Real fraud is far rarer than 15% and its patterns shift as fraudsters adapt, so any real model needs ongoing monitoring.
