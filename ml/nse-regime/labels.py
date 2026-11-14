"""NSE market regime classifier: step 2, the regime labels.

Four regimes from two measurements on the Nifty 200, trend x volatility:

    trend       trailing 63-day (about 3 months) return above zero -> up, otherwise down
    volatility  20-day realised volatility (annualised) above 18%  -> high, otherwise low

    Calm rally      up,   low volatility
    Volatile rally  up,   high volatility
    Grind down      down, low volatility
    Stress          down, high volatility

The thresholds are fixed numbers, not percentiles of the whole history, so a label never depends on the
future. A switch is confirmed only after the new regime has held for CONFIRM_DAYS trading days (also
causal), which stops the label flickering around a threshold.

The label is the *current* regime. The model's inputs must not be the two measurements above (VIX, drawdown,
distance from the 200-day average, short-term returns and so on instead), or it would only re-derive its label.

    uv run python labels.py    # prints the label summary and writes data/regime_timeline.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data

TREND_DAYS = 63
VOL_DAYS = 20
VOL_THRESHOLD = 18.0   # annualised %, about the 70th percentile of 2008-2026
CONFIRM_DAYS = 5
TRADING_DAYS = 252

REGIMES = ["Calm rally", "Volatile rally", "Grind down", "Stress"]
COLORS = {"Calm rally": "#2f7d4f", "Volatile rally": "#c98a0a", "Grind down": "#3b6fb6", "Stress": "#c0392b"}  # passes the colour-blind check


def raw_regime(trend_up: pd.Series, vol_high: pd.Series) -> pd.Series:
    names = np.select(
        [trend_up & ~vol_high, trend_up & vol_high, ~trend_up & ~vol_high, ~trend_up & vol_high],
        REGIMES, default="")
    return pd.Series(names, index=trend_up.index)


def confirm(raw: pd.Series, days: int = CONFIRM_DAYS) -> pd.Series:
    """Causal debounce: keep the current regime until a different one has appeared `days` days in a row."""
    out, current, candidate, run = [], None, None, 0
    for value in raw:
        if current is None:
            current = value
        elif value == current:
            candidate, run = None, 0
        else:
            run = run + 1 if value == candidate else 1
            candidate = value
            if run >= days:
                current, candidate, run = value, None, 0
        out.append(current)
    return pd.Series(out, index=raw.index)


def build_labels(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per day with both measurements, the unconfirmed regime and the final `regime` label."""
    px = raw[("Close", data.TICKERS["index"])].dropna()
    ret = np.log(px).diff()
    table = pd.DataFrame({
        "close": px,
        "return_63d_pct": px.pct_change(TREND_DAYS) * 100,
        "vol_20d_pct": ret.rolling(VOL_DAYS).std() * np.sqrt(TRADING_DAYS) * 100,
    }).dropna()
    table["raw_regime"] = raw_regime(table["return_63d_pct"] > 0, table["vol_20d_pct"] > VOL_THRESHOLD)
    table["regime"] = confirm(table["raw_regime"])
    return table


def summarise(table: pd.DataFrame) -> None:
    n = len(table)
    print(f"{n} labelled days, {table.index.min().date()} to {table.index.max().date()}\n")
    share = table["regime"].value_counts(normalize=True).reindex(REGIMES) * 100
    print("share of days (%):"); print(share.round(1).to_string())
    for name in ("raw_regime", "regime"):
        switches = int((table[name] != table[name].shift()).sum() - 1)
        print(f"\n{name}: {switches} switches, average run {n / (switches + 1):.1f} days")
    print("\ndays per year (%):")
    print((table.groupby(table.index.year)["regime"].value_counts(normalize=True).unstack().reindex(columns=REGIMES)
           .fillna(0) * 100).round(0).astype(int).to_string())
    print("\nmean 20-day forward return (%) after each regime (not used for the label):")
    forward = table["close"].shift(-20) / table["close"] - 1
    print((forward.groupby(table["regime"]).mean().reindex(REGIMES) * 100).round(2).to_string())


def plot(table: pd.DataFrame, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(table.index, table["close"], color="#222", lw=0.8)
    ax.set_yscale("log")
    ymin, ymax = ax.get_ylim()
    for name in REGIMES:
        ax.fill_between(table.index, ymin, ymax, where=(table["regime"] == name).to_numpy(),
                        color=COLORS[name], alpha=0.28, lw=0, label=name)
    ax.set_ylim(ymin, ymax)
    ax.set_title("Nifty 200 and the four regimes")
    ax.legend(loc="upper left", ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    table = build_labels(data.load_raw())
    summarise(table)
    plot(table, data.DATA_DIR / "regime_timeline.png")
    table.to_csv(data.DATA_DIR / "labels.csv")


if __name__ == "__main__":
    main()
