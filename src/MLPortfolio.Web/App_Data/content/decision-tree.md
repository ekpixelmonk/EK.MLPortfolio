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
