"""NSE market regime classifier: step 3, audit the data and build the features.

    uv run python features.py   # prints the audit, the feature table summary and per-regime medians

Seven features, all from the Nifty 200 close and India VIX. None is the 63-day return or the 20-day
volatility that define the labels (see labels.py). Every feature uses only data up to and including the day.

Two features from the first proposal were dropped after the data audit: Yahoo has no usable high-low range
for the Nifty 200 before 2012 (high equals low on about 1,000 days) and no index volume before mid-2015
(zero on 1,868 days). Keeping them would remove the 2008 crisis from the training data. A VIX 5-day change
replaces them as the fast signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import data
import labels

OUTLIER_MOVE_PCT = 6.0   # a daily index move beyond this is flagged (and kept: these are real events)
VIX_AVG_DAYS = 60
HIGH_DAYS = 252
FAST_MA, SLOW_MA = 50, 200

FEATURES = [
    "india_vix", "vix_vs_avg_pct", "vix_change_5d_pct", "drawdown_pct",
    "dist_200dma_pct", "ma50_vs_ma200_pct", "return_5d_pct",
]


@dataclass
class Audit:
    counts: dict
    notes: list[str]
    flagged: pd.DataFrame   # extreme daily index moves, kept


def build_features(close: pd.Series, vix: pd.Series) -> pd.DataFrame:
    """Features at each close. Index features use the full index history; VIX features use only days that have
    a VIX close (nothing is filled), and the two are then joined on the days where both exist."""
    ma_fast, ma_slow = close.rolling(FAST_MA).mean(), close.rolling(SLOW_MA).mean()
    index_part = pd.DataFrame({
        "drawdown_pct": (close / close.rolling(HIGH_DAYS, min_periods=HIGH_DAYS // 2).max() - 1) * 100,
        "dist_200dma_pct": (close / ma_slow - 1) * 100,
        "ma50_vs_ma200_pct": (ma_fast / ma_slow - 1) * 100,
        "return_5d_pct": close.pct_change(5) * 100,
    })
    vix = vix.dropna()
    vix_part = pd.DataFrame({
        "india_vix": vix,
        "vix_vs_avg_pct": (vix / vix.rolling(VIX_AVG_DAYS).mean() - 1) * 100,
        "vix_change_5d_pct": (vix / vix.shift(5) - 1) * 100,
    })
    return vix_part.join(index_part, how="inner")[FEATURES]


def prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, Audit]:
    """Checks the download, then returns one row per usable day: features plus the regime label."""
    idx, vx = data.TICKERS["index"], data.TICKERS["vix"]
    close, high, low, volume = (raw[(f, idx)] for f in ("Close", "High", "Low", "Volume"))
    vix = raw[("Close", vx)]
    notes: list[str] = []
    counts = {"rows downloaded": len(raw)}

    counts["duplicate dates"] = int(raw.index.duplicated().sum())
    counts["non-positive closes"] = int((close <= 0).sum() + (vix <= 0).sum())
    counts["index days with no close"] = int(close.isna().sum())
    notes.append(f"{counts['duplicate dates']} duplicate dates, {counts['non-positive closes']} non-positive closes, "
                 f"{counts['index days with no close']} index days without a close (the first row has no return).")

    before_vix = vix.index < pd.Timestamp("2008-03-03")
    counts["vix missing before launch"] = int(vix[before_vix].isna().sum())
    counts["vix missing after launch"] = int(vix[~before_vix].isna().sum())
    notes.append(f"India VIX has no close for {counts['vix missing before launch']} days before it starts on 2008-03-03 "
                 f"and for {counts['vix missing after launch']} days after (mostly 1 January and other holidays that "
                 "Yahoo lists for the index). Those days are dropped, never filled.")

    counts["index high equals low"] = int((high == low).sum())
    counts["index high equals low through 2011"] = int((high[high.index < "2012-01-01"] == low[low.index < "2012-01-01"]).sum())
    counts["index zero-volume days"] = int((volume == 0).sum())
    counts["index zero-volume days through mid-2015"] = int((volume[volume.index < "2015-07-01"] == 0).sum())
    notes.append(f"The index high equals the low on {counts['index high equals low']} days ({counts['index high equals low through 2011']} of them through 2011) "
                 f"and its volume is zero on {counts['index zero-volume days']} days ({counts['index zero-volume days through mid-2015']} of them before mid-2015): "
                 "Yahoo has only the close for those. A high-low range feature and a volume feature were therefore not used.")

    stale = (close.pct_change() == 0)
    counts["unchanged closes"] = int(stale.sum())
    notes.append(f"{counts['unchanged closes']} index close is identical to the previous day; it is kept (it does not distort any feature).")

    move = close.pct_change() * 100
    flagged = pd.DataFrame({"close": close, "move_pct": move})[move.abs() > OUTLIER_MOVE_PCT]
    counts["extreme daily moves"] = len(flagged)
    notes.append(f"{len(flagged)} days with an index move beyond {OUTLIER_MOVE_PCT:.0f}% were flagged; they are the 2008 crash, "
                 "the 2009 election rally and the March 2020 crash, so they were kept.")

    table = build_features(close, vix)
    table = table.join(labels.build_labels(raw)[["regime"]], how="inner")
    warm = len(table)
    table = table.dropna()
    counts["dropped for warm-up"] = warm - len(table)
    counts["rows used"] = len(table)
    notes.append(f"Rolling windows (up to 252 days) need history, so {counts['dropped for warm-up']} early days were dropped. "
                 f"{len(table)} days remain, from {table.index.min().date()} to {table.index.max().date()}.")
    return table, Audit(counts=counts, notes=notes, flagged=flagged)


def main() -> None:
    table, audit = prepare(data.load_raw())
    for note in audit.notes:
        print("-", note)
    print("\nflagged moves:"); print(audit.flagged.round(2).to_string())
    print("\nfeature summary:"); print(table[FEATURES].describe().T.round(2).to_string())
    print("\nmedian by regime:")
    print(table.groupby("regime")[FEATURES].median().reindex(labels.REGIMES).round(1).T.to_string())
    print("\n25th-75th percentile by regime:")
    q = table.groupby("regime")[FEATURES].quantile([0.25, 0.75]).round(1)
    for r in labels.REGIMES:
        print(r, {f: f"{q.loc[(r, 0.25), f]} to {q.loc[(r, 0.75), f]}" for f in FEATURES})
    print("\ncorrelation between features:"); print(table[FEATURES].corr().round(2).to_string())
    print("\nregime counts:"); print(table["regime"].value_counts().reindex(labels.REGIMES).to_string())


if __name__ == "__main__":
    main()
