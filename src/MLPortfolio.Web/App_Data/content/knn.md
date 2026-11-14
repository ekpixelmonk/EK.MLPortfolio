## Problem

An investor wants to know what kind of stock they are looking at. Given the price/earnings ratio, dividend yield and yearly revenue growth, the model classifies it as `Value`, `Growth` or `Income` by finding the most similar stocks it has seen.

## Data

Synthetic. 3,000 stocks are drawn from three overlapping clusters: Value (modest P/E, steady), Growth (high P/E and fast revenue growth) and Income (high dividend yield). The overlap is deliberate, so the model cannot be perfect. No market data is used.

## Approach

A scikit-learn pipeline: standardise the three features (essential, because distance is dominated by whichever feature has the biggest scale), then a `KNeighborsClassifier` with k = 15 fitted on 75% of the data. The pipeline is exported to ONNX, checked against scikit-learn on the held-out set, and run inside this site. The probabilities are simply the share of the 15 neighbours in each class.

## Results

On the held-out 25%, the model reaches **85.3% accuracy** and a one-vs-rest ROC AUC of 0.95, against 37.9% for always guessing the biggest class. Growth is easy to separate; most mistakes are between Value and Income. The k chart shows the usual trade-off: with k = 1 the model memorises noise, and a moderate k generalises better.

## Limitations

The clusters are invented, and real stocks do not fall into three tidy groups. KNN stores the whole training set, so it slows down on large data. Not investment advice. Educational purposes only.
