## Problem

A lender wants loan decisions that can be explained to the applicant. Given credit score, income, loan size relative to income, years in the current job and any collateral, the model predicts whether an application is `Approved` or `Rejected`, using a tree of yes/no questions.

## Data

Synthetic. 5,000 applications are generated from a banker's rule of thumb (a good score with a modest loan, or a very good score with a stable job, or property collateral on a smaller loan). Then 6% of decisions are flipped at random to mimic inconsistent human judgement. About 44% of applications are approved. No real applicant data is used.

## Approach

A scikit-learn pipeline: numeric features pass straight through (trees do not need scaling), collateral is one-hot encoded, and a `DecisionTreeClassifier` limited to depth 4 with at least 20 applications per leaf is fitted on 75% of the data. The pipeline is exported to ONNX, checked against scikit-learn on the held-out set, and run inside this site.

## Results

On the held-out 25%, the tree reaches **93.2% accuracy** and an ROC AUC of 0.93, against 55.7% for always guessing "Rejected". Credit score and loan-to-income ratio carry most of the decisions, which matches how the data was built. The depth chart shows the trade-off: deeper trees fit the training data better and the noise with it, while held-out accuracy stops improving after a few levels.

## Limitations

The "rules" are invented, so the tree rediscovers a rule of thumb, not real lending policy. Single trees are unstable: a small change in the data can reshuffle the questions. A real credit decision would also need fairness checks. Educational purposes only.

## Conclusion

The decision tree rediscovered the lending rule of thumb that generated the data, scoring 93.2% against 55.7% for always guessing "Rejected", and it did so with a model anyone can read and explain to an applicant. That is its real selling point: the questions it asks (credit score, loan size relative to income) are exactly the ones a loan officer would ask. The depth chart is the main caution: a deeper tree fits the training data better but only memorises its noise, so a shallow tree was the right choice.

A single tree is also unstable. A small change in the data can reshuffle the questions, so the neat rules should be treated as one reasonable reading of the data, not the truth.

**What would improve it:**

- **An ensemble.** A random forest or gradient boosting model averages many trees and is far steadier, at the cost of the readable rules. It is the usual next step when accuracy matters more than explanation.
- **Pruning and cross-validation.** Choosing the depth and leaf size by cross-validation, or by cost-complexity pruning, is more reliable than fixing them by hand.
- **Constraints that match policy.** For lending, monotonic constraints (a higher credit score can never lower approval) keep the rules sensible.
- **Real data and fairness checks.** The rules here are invented; a real tree would need real decisions and a review of who it approves and rejects.
