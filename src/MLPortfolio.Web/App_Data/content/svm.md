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
