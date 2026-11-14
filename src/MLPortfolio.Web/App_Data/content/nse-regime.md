## Problem

Markets move through different regimes: calm uptrends, nervous rallies, slow declines and outright stress. Knowing which regime the market is in helps frame risk: the same 1% daily move means something different in a calm rally than in a sell-off. This project classifies the Indian market's current regime from a handful of indicators that can be read off the Nifty 200 index and India VIX on any given day.

This project is still being built. The data and the features are described below; the model, charts, results and conclusion will follow.

## Data

### Where it comes from

Two daily series, downloaded from Yahoo Finance with the `yfinance` library:

- **Nifty 200** (`^CNX200`): an index of the 200 largest listed Indian companies by market value. It is broader than the Nifty 50, so it is less dominated by a handful of stocks. We use its daily close (and, for checking, its high, low and volume).
- **India VIX** (`^INDIAVIX`): the market's expected volatility over the next 30 days, calculated by NSE from the prices of Nifty 50 options. Higher means traders are paying more for protection and expect bigger moves. It is quoted in index points that read like an annualised percentage (a VIX of 20 means the market expects roughly 20% annualised volatility). It is an input to the model; it is not part of the regime labels.

The download covers 3 September 2007 to 13 November 2026 (4,724 rows). The Nifty 200 has a close for every one of them. India VIX only starts on 3 March 2008. The raw download is not stored in the repository, because Yahoo's terms do not allow redistributing their data; `uv run python data.py` fetches it again.

![Nifty 200 and India VIX with regimes shaded](/projects/nse-regime/charts/eda-history.png)

*The Nifty 200 (log scale) and India VIX from 2007 to 2026, shaded by regime. Stress sits on the 2008 crash, March 2020 and 2022; VIX spikes line up with the red bands.*

The two series tell the same story. India VIX jumps whenever the index falls fast (late 2008, March 2020), and calm stretches of a rising index come with a VIX that drifts down to 10 to 15. The index more than quadrupled over the period, which is why it is drawn on a log scale.

### What was checked

Before building anything the data was audited, and every check is run by `features.py`:

- **Duplicates and impossible values:** no duplicate dates and no zero or negative prices in either series.
- **Missing data:** India VIX has no close on 126 days before it launched, and on 17 days afterwards (1 January and other holidays that Yahoo lists for the index but on which the market was shut). Those days are dropped. They are never filled in, because carrying a stale VIX forward would invent a calm reading for a day when nothing was measured.
- **High, low and volume:** Yahoo's history for the Nifty 200 is incomplete here. On 1,006 days the high equals the low (1,005 of them before 2012), so there is no real intraday range, and the index volume is zero on 1,868 days (1,862 of them before mid-2015). Features based on the daily range or on volume were therefore left out: using them would have forced us to throw away 2008 to 2011, which includes the 2008 crash, the most important stress period in the data.
- **Outliers:** 18 days had an index move beyond 6%. They are the 2008 crash days (including a fall of 11% on 24 October 2008), the 16.8% jump on 18 May 2009 (the election result), the March 2020 crash (a fall of 12.8% on 23 March) and a few others. These are real events, and they are exactly what a regime model has to learn from, so none were removed.
- **Warm-up:** the indicators below look back up to 252 trading days, so the first 72 days with a VIX reading had too little history and were dropped.

![Missing data by year](/projects/nse-regime/charts/eda-data-quality.png)

*Where the download has gaps. India VIX is missing before March 2008; the Nifty 200's high-low range is missing until 2011 and its volume until mid-2015, which is why those two features were not used.*

![Distribution of daily index moves and the largest moves](/projects/nse-regime/charts/eda-returns.png)

*Most days move less than 2%, but the tails are fat. Dashed lines mark the 6% flag; 18 days beyond it (the 2008 crash, the 2009 election rally, March 2020) are real events and were kept.*

Both charts support the choices above: the gaps explain why two features were left out, and the extreme days are few, genuine and exactly the moments a regime model exists for.

After the checks **4,509 trading days remain, from 23 June 2008 to 13 November 2026**.

### The regime labels

The model needs a correct answer to learn from, so each day is given a regime from a fixed rule applied to two measurements of the Nifty 200:

- **Trend:** is the 63-day (about three months) return above or below zero?
- **Volatility:** is the 20-day realised volatility (annualised) above or below 18%? The 18% line is roughly the 70th percentile of 2008 to 2026.

| Regime | Trend | Volatility | Days |
|---|---|---|---|
| Calm rally | up | low | 2,408 (53%) |
| Volatile rally | up | high | 452 (10%) |
| Grind down | down | low | 908 (20%) |
| Stress | down | high | 741 (16%) |

![Scatter of 63-day return against 20-day volatility coloured by regime](/projects/nse-regime/charts/eda-label-rule.png)

*Every day plotted by the two measurements that define the regimes. The lines are the thresholds: zero return and 18% volatility. The colour is the confirmed regime, so points near a line can carry the neighbouring colour.*

The thresholds are fixed numbers rather than percentiles of the whole history, and a regime change is only accepted once the new regime has held for 5 trading days. Both choices use only past data, so no label depends on the future, and the 5-day rule cuts the number of switches from 280 to 103 (an average regime lasts about 45 days). The labels match history: 2008 is almost entirely Stress, March 2020 is Stress, 2014 and 2017 are mostly Calm rally, and the 2021 recovery is a Volatile rally.

![Regimes by year and how long each regime lasts](/projects/nse-regime/charts/eda-regime-durations.png)

*Left: the mix of regimes in each year (2008 is almost all Stress, 2014 and 2017 almost all Calm rally). Right: Calm rally spells last longest (median about 60 days); the other three typically last under a month, which is part of why they are harder to learn.*

![Which regime follows which](/projects/nse-regime/charts/eda-transitions.png)

*Across the 103 regime changes, what follows what. Stress usually gives way to Grind down (58%) or a Volatile rally (32%); a Calm rally mostly slips into Grind down (56%) or turns into a Volatile rally (32%) and rarely jumps straight to Stress (12%).*

Two consequences for the model. The classes are unbalanced: Calm rally is more than half of all days, so a lazy model could look accurate by always saying Calm rally, which is why the results below also report balanced accuracy and compare against that guess. And the regimes are not equally learnable: Volatile rally spells are short and rare, so the model has few examples to learn from.

Two things to keep in mind. The regimes are a definition, not a discovered truth: the model learns to reproduce this rule from other indicators. And the model is deliberately not given the 63-day return or the 20-day volatility, because that would only let it recompute its own answer.

## Features

A feature is one number computed at the close of a day that describes the market. Seven are used. All of them depend only on data up to and including that day, and none of them is the trend or volatility measurement used to define the labels. They fall into three groups: how nervous the market is (VIX), how far prices have fallen, and where prices sit against their long-term trend.

![The seven features over time](/projects/nse-regime/charts/eda-feature-history.png)

*Each feature from 2008 to 2026. VIX, drawdown and the distance from the 200-day average all swing hard in the shaded Stress periods; the 5-day return and the VIX changes are noisy and look much the same in every regime.*

![Histogram of each feature](/projects/nse-regime/charts/eda-feature-distributions.png)

*How each feature is distributed. VIX and the VIX measures are skewed to the right (calm most of the time, occasional spikes); drawdown is bunched near zero. Dashed lines show the form ranges; few days fall outside them.*

The table shows the middle half of the values (25th to 75th percentile) for each regime, so you can see what "typical" looks like.

| Feature | Calm rally | Volatile rally | Grind down | Stress |
|---|---|---|---|---|
| India VIX | 13.0 to 18.1 | 18.9 to 30.7 | 14.4 to 18.8 | 20.0 to 35.5 |
| VIX vs 60-day average (%) | -13.7 to 5.1 | -13.1 to 7.1 | -10.2 to 8.9 | -7.4 to 19.4 |
| VIX 5-day change (%) | -5.1 to 6.2 | -8.7 to 4.4 | -6.1 to 7.0 | -8.4 to 6.3 |
| Drawdown from 1-year high (%) | -4.5 to -0.4 | -9.0 to -2.0 | -11.5 to -6.2 | -33.6 to -9.4 |
| Distance from 200-day average (%) | 4.1 to 12.1 | 2.9 to 25.5 | -3.7 to 2.0 | -17.3 to -2.2 |
| 50-day vs 200-day average (%) | 2.3 to 9.2 | -0.1 to 18.6 | -2.8 to 2.6 | -14.3 to -0.5 |
| 5-day return (%) | -0.9 to 1.5 | -0.8 to 2.4 | -1.7 to 1.6 | -2.0 to 2.8 |

![Each feature split by regime](/projects/nse-regime/charts/eda-features-by-regime.png)

*Each feature split by regime (boxes show the middle half, whiskers the 5th to 95th percentile). Drawdown and distance from the 200-day average separate the regimes clearly; the 5-day return and VIX 5-day change barely differ between them.*

![VIX against drawdown and against the 200-day average, coloured by regime](/projects/nse-regime/charts/eda-feature-pairs.png)

*Two features at a time. The regimes occupy different parts of the plane: the long-term trend (vertical position in the right panel) splits the rallies from the declines, and VIX splits the calm from the turbulent. The overlap at the edges is where the model makes its mistakes.*

### India VIX (`india_vix`)

**What it is:** the day's closing level of India VIX.

**Why it matters:** it is the market's own forecast of how bumpy the next month will be, and it reacts the moment fear rises, usually before realised volatility catches up. It is the best single separator of calm from turbulent regimes: the typical VIX is 13 to 18 in a Calm rally and 20 to 36 in Stress. It cannot tell a rally from a decline by itself, since a Volatile rally and Stress both have a high VIX.

**Reading it:** below about 15 is complacent, 15 to 20 is normal, above 25 is stressed. Its range in the data is 9.2 (very calm) to 85.1 (March 2020).

### VIX against its 60-day average (`vix_vs_avg_pct`)

**What it is:** today's VIX divided by its average over the previous 60 days with a VIX reading, minus one, as a percentage.

**Why it matters:** the level of VIX means different things in different years, and what matters is often whether fear is building or fading relative to its recent norm. +20% says the market is much more worried than usual; -10% says it is calming down. Stress days have the highest values (the upper quarter of Stress days sits at +19% or more), while rallies sit slightly below zero.

**Reading it:** positive means fear is rising against recent history. It ranges from -38% to +255%.

### VIX 5-day change (`vix_change_5d_pct`)

**What it is:** the percentage change in VIX over the last 5 trading days.

**Why it matters:** it is the fast signal. A VIX that jumps 30% in a week points to a sudden shock, while a falling VIX after a panic points to relief. It also tells a fresh sell-off from a long-running one when the other features look similar.

**An honest note:** on its own this is the weakest feature: the typical ranges for the four regimes overlap almost completely. It is included because the model can combine it with the others, and part of the project is to measure how much it actually adds. It ranges from -51% to +130%.

### Drawdown from the 1-year high (`drawdown_pct`)

**What it is:** how far the Nifty 200's close is below its highest close of the last 252 trading days (about a year). Zero means the index is at a new high; -10 means it is 10% below.

**Why it matters:** it separates a dip from a real decline, and it is the clearest signal of the trend regimes. In a Calm rally the index is typically within 0.4% to 4.5% of its high, in a Grind down 6% to 12% below it, and in Stress 9% to 34% below it. The deepest value is -64% (the 2008 crash).

**Reading it:** the more negative the number, the longer and deeper the damage.

### Distance from the 200-day average (`dist_200dma_pct`)

**What it is:** how far the close is above or below its average of the last 200 trading days, as a percentage.

**Why it matters:** the 200-day average is the most widely watched line for the market's long-term trend. Being above it means the long-term trend is up, and the further above, the stronger the run. It separates the rallies (typically +3% to +25%) from the declines (below zero in Grind down and Stress).

**Reading it:** the sign tells the long-term trend and the size tells how stretched the market is. It ranges from -47% to +44%.

### 50-day average against the 200-day average (`ma50_vs_ma200_pct`)

**What it is:** the average of the last 50 closes divided by the average of the last 200, minus one, as a percentage.

**Why it matters:** it describes the structure of the trend rather than the position of the price. When the medium-term average sits above the long-term one, the market has been rising for a while; when it falls below, a decline is established. It changes more slowly than the distance from the 200-day average, so it steadies the model through short wobbles, and it flips sign at the turns, which traders call the golden cross and the death cross.

**Reading it:** above zero means an established uptrend, below zero an established downtrend. It ranges from -31% to +33%.

### 5-day return (`return_5d_pct`)

**What it is:** the index's percentage change over the last 5 trading days.

**Why it matters:** it is the market's latest move and the only purely short-term price signal in the set, so it lets the model react to what just happened. Used alone it is weak, because a calm rally and a slow decline both include up and down weeks (the typical ranges overlap heavily); its value comes in combination, for example a sharp rise with a still-high VIX.

**An honest note:** a 5-day window partly overlaps the 63-day return used for the labels, so it is the feature most closely related to the labelling rule. It ranges from -19% to +20%.

### How the features relate

Several features move together: drawdown, distance from the 200-day average and the 50/200-day gap are strongly correlated (0.76 to 0.89), because all three measure how far the index has fallen or risen over a long stretch. India VIX and drawdown move in opposite directions (-0.75): a falling market raises fear. That overlap is acceptable for the models planned here, but it means the features do not each add a separate piece of information, and the feature importance chart should be read with that in mind.

![Correlation matrix of the features](/projects/nse-regime/charts/eda-correlation.png)

*Correlation between the features. Drawdown, distance from the 200-day average and the 50/200-day gap are strongly related (0.76 to 0.89), so they overlap; the VIX features and the 5-day return are largely independent of them.*

![Mutual information between each feature and the regime](/projects/nse-regime/charts/eda-feature-relevance.png)

*Mutual information between each feature and the regime, on its own. The three long-term trend measures (distance from the 200-day average, drawdown, 50/200-day gap) say the most, then VIX; the 5-day return and the two VIX-change features say very little by themselves.*

### What the exploration shows

- **The classes are unbalanced.** Calm rally is 53% of days and Volatile rally 10%, so accuracy alone flatters a lazy model. Balanced accuracy and a most-common-regime baseline are used below.
- **The long-term trend measures carry most of the signal.** Distance from the 200-day average, drawdown and the 50/200-day gap separate the rallies from the declines best, and VIX then separates the calm regimes from the turbulent ones. No single feature separates all four.
- **Three features say very little on their own.** The 5-day return, VIX against its average and the VIX 5-day change look much the same in every regime; any value they add must come from combining them with the others.
- **The features overlap.** Drawdown, the 200-day distance and the 50/200-day gap are highly correlated, so the model has fewer than seven independent pieces of information.
- **The regime edges are blurry.** The scatter plots show regimes shading into each other along the thresholds of the labelling rule. A model working from different indicators cannot see exactly where a threshold falls, so mistakes cluster at the boundaries, and the results below confirm that.
- **The data is thin where it matters.** Stress and Volatile rally each appear in only about 20 separate spells in eighteen years, and several of the longest Stress spells (2008 to 2009, 2011, 2020) dominate that regime's days.

## Approach

The task is a four-way classification: given the seven features on a day, name that day's regime.

**Testing without peeking.** Market data is a time series, so a random train/test split would let the model learn from days next to its test days, and the regimes last about 45 days, so neighbouring days look almost identical. Instead the model is tested walk-forward: the history is cut into six consecutive test periods (February 2011 to November 2026, 3,864 days in all), and each period is scored by a model trained only on the days before it. A gap of 63 trading days is left between training and test, because the trend measurement behind the labels looks back 63 days. Every number below comes from days the model had not seen.

**Candidates.** Four models, each on the same standardised features: logistic regression, logistic regression with the rare regimes weighted up (so each regime counts equally, not each day), a random forest and gradient boosting. The baseline is always guessing the most common regime in the training data.

**The choice.** The class-weighted logistic regression, because it scored best on every measure (below), and because it is the easiest to explain: its coefficients show what pushes each regime up or down. The model shown on this site is the same model fitted on all 4,509 days and exported to ONNX; the scores below come from the walk-forward models, not from that final fit.

## Model training

### How the walk-forward test was run

Each row is one round. The model is trained on every day before the test period (an expanding window that always starts in June 2008), then scored on the next stretch of about 640 days it has never seen. A 63-day gap separates the two, so no test day shares a trend window with a training day.

| Fold | Trained on | Training days | Tested on | Test days | Most-common-regime guess | Model |
|---|---|---|---|---|---|---|
| 1 | Jun 2008 to Nov 2010 | 582 | Feb 2011 to Sep 2013 | 644 | 23.4% | 59.8% |
| 2 | Jun 2008 to Jun 2013 | 1,226 | Sep 2013 to May 2016 | 644 | 61.6% | 72.0% |
| 3 | Jun 2008 to Feb 2016 | 1,870 | May 2016 to Dec 2018 | 644 | 78.9% | 73.9% |
| 4 | Jun 2008 to Sep 2018 | 2,514 | Dec 2018 to Aug 2021 | 644 | 56.7% | 65.4% |
| 5 | Jun 2008 to May 2021 | 3,158 | Aug 2021 to Apr 2024 | 644 | 61.5% | 80.6% |
| 6 | Jun 2008 to Dec 2023 | 3,802 | Apr 2024 to Nov 2026 | 644 | 50.2% | 65.4% |

The first round has only 582 training days (about two and a half years), and the guess to beat there is poor, because the training window is mostly the 2008 crash. The model gains from more history in the later rounds, but not steadily: it falls short of the simple guess in round 3, a calm stretch where one regime dominated.

### Settings

| Setting | Value |
|---|---|
| Algorithm | multinomial logistic regression (`LogisticRegression`, L2 penalty, C = 1) |
| Solver | lbfgs, converged in 33 iterations (limit 5,000) |
| Preprocessing | each feature standardised (mean 0, standard deviation 1), fitted on the training days only |
| Class weights | Calm rally 0.47, Volatile rally 2.49, Grind down 1.24, Stress 1.52 (balanced: rarer regimes count for more) |
| Final training set | 4,509 days, 23 Jun 2008 to 13 Nov 2026; 168 feature values capped to the form range |
| Walk-forward folds | 6 expanding windows, 63-day gap before each test period |
| Out-of-sample days | 3,864 (2011-02-10 to 2026-11-13) |
| Random seed | 42 |

The model has no tuned settings: the penalty strength is the library default, and nothing was searched on the test days. The one deliberate choice is the class weighting, made after comparing the four candidates (see Results), so the comparison is not entirely free of hindsight.

### How it does on each regime

| Regime | Precision | Recall | F1 | Test days |
|---|---|---|---|---|
| Calm rally | 88.0% | 74.3% | 0.81 | 2,210 |
| Volatile rally | 29.3% | 14.9% | 0.20 | 276 |
| Grind down | 49.8% | 85.7% | 0.63 | 879 |
| Stress | 71.8% | 49.9% | 0.59 | 499 |

Precision is how often a predicted regime was right; recall is how many of that regime's actual days were found; F1 combines them. Calm rally is found and trusted. Grind down is found often (86%) but over-predicted, which lowers its precision: this is where Calm rally days and Stress days end up when they are misread. Volatile rally is the weak spot on both measures.

### What the model learned

Standardised coefficients: how strongly a one-standard-deviation rise in a feature pushes the model towards each regime (positive) or away from it (negative).

| Feature (standardised) | Calm rally | Volatile rally | Grind down | Stress |
|---|---|---|---|---|
| India VIX | -0.94 | +1.59 | -1.80 | +1.14 |
| VIX vs its 60-day average | +0.13 | -0.19 | +0.02 | +0.04 |
| VIX change over 5 days | +0.25 | -0.32 | +0.38 | -0.31 |
| Nifty 200 below its 1-year high | +1.38 | -0.41 | -0.79 | -0.18 |
| Nifty 200 vs its 200-day average | +3.34 | +2.59 | -2.41 | -3.52 |
| 50-day vs 200-day average | -1.30 | -0.48 | +0.67 | +1.11 |
| Nifty 200 5-day return | -0.90 | -0.26 | +0.47 | +0.69 |
| *Intercept* | +0.07 | +0.57 | +0.01 | -0.66 |

Read these together, not one at a time. Distance from the 200-day average and the 50/200-day gap are highly correlated, so their coefficients partly offset each other (the 50/200 coefficient for Calm rally is negative even though a rising market is good for it): what counts is their combined effect. The clear patterns are that being above the 200-day average pushes towards Calm rally and Volatile rally and away from Grind down and Stress, and that a high VIX pushes towards the turbulent regimes (Volatile rally, Stress) and a low VIX towards the quiet ones.

### How good are the probabilities?

Probability quality on the out-of-sample days: log loss 0.823 (a model that always gave the training-period regime shares scores about 1.121), Brier score 0.439. Accuracy on the training days themselves is 72.3%, against 69.5% out of sample.

Log loss and the Brier score judge the whole set of probabilities, not just the top pick; lower is better. The model clearly improves on the fixed-shares reference. The training accuracy (a single fit on all days) is only a little higher than the walk-forward accuracy, so the model is not memorising the history: it is limited by the information in the features, not by overfitting.

## Results

| Model | Accuracy | Balanced accuracy | ROC AUC |
|---|---|---|---|
| Always the most common regime | 55.4% | 30.1% | n/a |
| Logistic regression | 68.2% | 48.7% | 0.844 |
| **Logistic regression, balanced classes** | **69.5%** | **56.2%** | **0.861** |
| Random forest | 68.9% | 48.3% | 0.811 |
| Gradient boosting | 65.0% | 46.8% | 0.774 |

![Accuracy and balanced accuracy of the candidate models](/projects/nse-regime/charts/model-comparison.png)

*Accuracy and balanced accuracy (the average of each regime's recall) for each candidate. Weighting the rare regimes more heavily helps the logistic regression most.*

Balanced accuracy is the average of each regime's recall, so a model cannot score well by ignoring the rare regimes. The chosen model beats the simple guess by 14 points of accuracy and almost doubles its balanced accuracy. The more flexible models (forest, boosting) did not do better: with only a few dozen real regime changes in eighteen years there is little for them to learn that a straight-line model misses.

**By regime.** Of the days that really were Grind down, 86% were identified; Calm rally 74%; Stress 50%; Volatile rally only 15%. When the model says Calm rally it is right 88% of the time, for Stress 72%, Grind down 50%, Volatile rally 29%. The two biggest confusions are Stress called Grind down (43% of Stress days) and Calm rally called Grind down (21%): both are the same market seen at the edge of a threshold, where the rule that defines the labels flips on a small move that the features cannot see.

![Confusion matrix from the walk-forward test](/projects/nse-regime/charts/confusion-matrix.png)

*Each row shows how the days of one actual regime were classified. Volatile rally is recognised only 15% of the time, and Stress is often mistaken for Grind down.*

**What drives it.** The distance of the index from its 200-day average does most of the work (it pushes strongly towards the rallies and away from Stress), and India VIX then separates the calm regimes from the turbulent ones (a high VIX favours Volatile rally and Stress, a low one Grind down). The VIX-against-average and VIX 5-day change features carry almost no weight, as the feature section suggested.

![Logistic regression coefficients by regime and feature](/projects/nse-regime/charts/coefficients.png)

*Standardised logistic regression coefficients: red pushes the model towards that regime, blue away from it.*

**Over time.** The model beats the simple guess in five of the six test periods. The exception is 2016 to 2018, a calm stretch where always guessing Calm rally scored 79% against the model's 74%. It is weakest in the first period (2011 to 2013, 60%), when it had only three years of history to learn from, and best in 2021 to 2024 (81%).

![Accuracy in each test period against the simple guess](/projects/nse-regime/charts/accuracy-by-period.png)

*Each test period is scored by a model trained only on earlier data. It beats the simple guess in five of six periods; the exception is the calm 2016 to 2018 stretch, where always guessing Calm rally is hard to beat.*

![Actual and predicted regimes over time with India VIX](/projects/nse-regime/charts/regime-timeline.png)

*India VIX with the actual regime (top strip) and the regime predicted for each day by a model that had only seen earlier data (bottom strip).*

![Predicted probability of Stress through time](/projects/nse-regime/charts/stress-probability.png)

*The model's probability of Stress each day (line) against the days that really were Stress (shaded).*

For context, a model told yesterday's regime would be right 98% of the time, because regimes last for weeks. That is a different and much easier question: this model has to name the regime from the day's indicators alone.

## Limitations

Regimes are a modelling construct, not a forecast. The labels are a fixed rule (trend and volatility thresholds), so the model approximates a definition, not a discovered truth, and it is weakest exactly where that rule flips. The history holds few independent regime changes, and Volatile rally in particular is rare (10% of days), which is why it is hard to learn. Past behaviour does not predict future returns. Educational purposes only, not investment advice.

## Conclusion

Seven indicators from the Nifty 200 and India VIX are enough to name the market's current regime correctly about seven days in ten, against five and a half in ten for always guessing the commonest regime, and that holds on data the model never saw. The simple, explainable model was as good as the complex ones, which says more about how little independent history there is than about the problem being easy. It is reliable on the clear cases (a calm uptrend, a long grind lower) and unreliable around the boundaries, where Stress is often read as a slow decline and a nervous rally is rarely recognised.

**What would improve it:**

- **Softer labels.** The model is judged against a hard rule with a sharp edge. Predicting the probability of each regime, or smoothing the thresholds, would stop the model being penalised for near-misses at the boundary.
- **Features that see the boundary.** The regimes are defined by 63-day trend and 20-day volatility; stronger indicators of those, such as an intraday range or index volume, would help, but Yahoo's Nifty 200 history lacks both before 2012 and 2015. A cleaner data source would allow them.
- **More markets.** Training on several indices (Nifty 50, Bank Nifty, global indices) would multiply the number of regime changes the model sees, which is the main limit on every model here.
- **A forecast, not just a label.** The more useful question is which regime comes next. That is much harder and would need its own baseline (persistence) to beat.
- **Regime-switching models.** A hidden Markov model fits regimes directly from returns and VIX, and would be a fair comparison for this rule-based approach.
