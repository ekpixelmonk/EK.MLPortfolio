## Problem

A bank wants to know which customers are likely to leave. Given age, account balance, number of products, years as a customer and whether the customer has been active recently, the model estimates the probability of churn.

## Data

Synthetic. 6,000 customers are generated from a known churn process: inactivity, a single product, short tenure and older age raise the risk. About 35% of customers churn. No real customer data is used.

## Approach

A scikit-learn pipeline: numeric features pass through, activity is one-hot encoded, and a `GaussianNB` classifier is fitted on 75% of the data. Naive Bayes models each feature separately for each class as a bell curve and multiplies the evidence together, assuming the features are independent. The pipeline is exported to ONNX, checked against scikit-learn on the held-out set, and run inside this site.

## Results

On the held-out 25%, the model reaches **71.9% accuracy** and an ROC AUC of 0.79, against 65.4% for always predicting "Stays". It is fast and a respectable baseline, but its bell-curve assumption does not suit counts like "products held" or skewed balances. The calibration chart shows the typical consequence: the probabilities are over-confident compared with the churn rates actually observed.

## Limitations

The data is synthetic and the churn rule is invented. Naive Bayes ranks customers better than it estimates probabilities, so treat the percentage as a score. Educational purposes only.

## Conclusion

Naive Bayes is a fast, simple baseline for customer churn, and here it is a modest one: 71.9% accuracy against 65.4% for always predicting "Stays", with an ROC AUC of 0.79. It ranks customers better than it estimates probabilities. The calibration chart shows the typical result: the percentages are over-confident, because the model treats age, balance, products, tenure and activity as independent when they are not. Treat its output as a score for ordering customers, not as a literal chance of leaving.

**What would improve it:**

- **Calibrating the probabilities.** Platt or isotonic calibration would fix the over-confidence without changing the ranking.
- **Better-suited distributions.** Gaussian bell curves fit "products held" (a small count) and a skewed balance poorly. Transforming those inputs (for example the log of the balance) or treating counts as categories would help.
- **Reducing the independence problem.** Combining or dropping strongly related inputs, or moving to a model that handles correlation, such as logistic regression, would remove its main assumption.
- **A stronger comparison.** Logistic regression and gradient boosting are the natural benchmarks; naive Bayes is worth keeping only if its speed or simplicity matters.
- **Real data.** The churn rule here is invented, and real churn depends on events (a price change, a service failure) that these inputs cannot see.
