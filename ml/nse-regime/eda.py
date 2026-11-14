"""NSE market regime classifier: exploratory data analysis charts for the project page.

    uv run python eda.py   # writes eda-*.png into the web app's chart folder and prints the captions

Covers the raw data (history, gaps, return distribution), the regime labels (rule, frequency, durations,
transitions) and the seven features (history, distributions, relationship to the regimes, correlations).
Colours: one fixed colour per regime (checked for colour-blind separation), one accent for everything else.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

import data
import features
import labels
import train

ACCENT, MUTED, INK = "#4f46e5", "#78716c", "#222222"
REGIMES = labels.REGIMES
COLORS = labels.COLORS
NICE = {c: c.replace("_pct", "").replace("_", " ") for c in features.FEATURES}
SHORT = {"drawdown_pct": "Drawdown from 1-yr high", "dist_200dma_pct": "Distance from 200-day avg", "ma50_vs_ma200_pct": "50-day vs 200-day avg",
         "vix_vs_avg_pct": "VIX vs 60-day avg", "vix_change_5d_pct": "VIX 5-day change", "india_vix": "India VIX", "return_5d_pct": "5-day return"}
NICE.update({"india_vix": "India VIX", "vix_vs_avg_pct": "VIX vs 60-day avg (%)", "vix_change_5d_pct": "VIX 5-day change (%)",
             "drawdown_pct": "Drawdown from 1-year high (%)", "dist_200dma_pct": "Distance from 200-day avg (%)",
             "ma50_vs_ma200_pct": "50-day vs 200-day avg (%)", "return_5d_pct": "5-day return (%)"})


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "sans-serif", "font.size": 10.5, "axes.edgecolor": "#d6d3d1", "axes.labelcolor": "#44403c",
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 12, "axes.grid": True,
        "grid.color": "#f0efed", "grid.linewidth": 0.8,
    })


def _legend_patches():
    return [plt.Rectangle((0, 0), 1, 1, color=COLORS[r]) for r in REGIMES]


def _shade(ax, lab: pd.DataFrame, alpha=0.16, only=None):
    """Background bands for each run of a regime."""
    regime = lab["regime"]
    for name in (only or REGIMES):
        mask = (regime == name).to_numpy()
        ax.fill_between(lab.index, 0, 1, where=mask, transform=ax.get_xaxis_transform(), color=COLORS[name],
                        alpha=alpha, lw=0, step="mid")


def build_all(out_dir: Path, refresh: bool = False) -> list[dict]:
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = data.load_raw(refresh=refresh)
    table, audit = features.prepare(raw)
    full = labels.build_labels(raw)
    idx, vx = data.TICKERS["index"], data.TICKERS["vix"]
    close = raw[("Close", idx)]
    vix = raw[("Close", vx)]
    charts: list[dict] = []

    def save(fig, name, caption):
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=170)
        plt.close(fig)
        charts.append({"file": name, "caption": caption})

    # 1. History: index and VIX with regimes -------------------------------------------------------
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.2, 5.4), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    a.plot(close.index, close, color=INK, lw=1)
    a.set_yscale("log")
    a.set_ylabel("Nifty 200 (log scale)")
    _shade(a, full)
    a.set_title("Nifty 200 and India VIX, with the regimes shaded")
    b.plot(vix.index, vix, color=INK, lw=0.9)
    b.set_ylabel("India VIX")
    _shade(b, full)
    a.legend(_legend_patches(), REGIMES, loc="upper left", ncol=2, frameon=False, fontsize=9)
    save(fig, "eda-history.png", "The Nifty 200 (log scale) and India VIX from 2007 to 2026, shaded by regime. Stress sits on the 2008 crash, "
                                 "March 2020 and 2022; VIX spikes line up with the red bands.")

    # 2. Data quality by year ----------------------------------------------------------------------
    h, l, v = (raw[(f, idx)] for f in ("High", "Low", "Volume"))
    per_year = pd.DataFrame({
        "India VIX missing": vix.isna().groupby(vix.index.year).sum(),
        "Index high equals low": (h == l).groupby(h.index.year).sum(),
        "Index volume is zero": (v == 0).groupby(v.index.year).sum(),
    })
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 5), sharex=True)
    for ax, col in zip(axes, per_year.columns):
        ax.bar(per_year.index, per_year[col], color=ACCENT, width=0.7)
        ax.set_ylabel("days")
        ax.set_title(col, loc="left", fontsize=10.5)
    axes[0].set_title("India VIX missing (days per year)", loc="left", fontsize=10.5)
    axes[1].set_title("Index high equals low (days per year)", loc="left", fontsize=10.5)
    axes[2].set_title("Index volume is zero (days per year)", loc="left", fontsize=10.5)
    save(fig, "eda-data-quality.png", "Where the download has gaps. India VIX is missing before March 2008; the Nifty 200's high-low range is missing "
                                       "until 2011 and its volume until mid-2015, which is why those two features were not used.")

    # 3. Daily return distribution -----------------------------------------------------------------
    ret = close.pct_change().dropna() * 100
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    a.hist(ret.clip(-8, 8), bins=90, color=ACCENT, alpha=0.9)
    a.set_yscale("log")
    for s in (-features.OUTLIER_MOVE_PCT, features.OUTLIER_MOVE_PCT):
        a.axvline(s, color=COLORS["Stress"], lw=1.2, ls="--")
    a.set_xlabel("Daily move of the Nifty 200 (%, clipped at ±8)")
    a.set_ylabel("days (log scale)")
    a.set_title("Daily moves")
    top = ret.abs().sort_values(ascending=False).head(8).index
    vals = ret.loc[top]
    b.barh([d.strftime("%d %b %Y") for d in top], vals, color=[COLORS["Stress"] if x < 0 else COLORS["Calm rally"] for x in vals])
    b.invert_yaxis()
    b.set_xlabel("move (%)")
    b.set_title("Eight largest moves")
    b.grid(axis="y", visible=False)
    save(fig, "eda-returns.png", f"Most days move less than 2%, but the tails are fat. Dashed lines mark the 6% flag; {len(audit.flagged)} days beyond it "
                                 "(the 2008 crash, the 2009 election rally, March 2020) are real events and were kept.")

    # 4. The label rule ----------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in REGIMES:
        s = full[full["regime"] == name]
        ax.scatter(s["vol_20d_pct"], s["return_63d_pct"], s=7, alpha=0.45, color=COLORS[name], label=name, lw=0)
    ax.axhline(0, color=INK, lw=1)
    ax.axvline(labels.VOL_THRESHOLD, color=INK, lw=1)
    ax.set_xlim(0, 80)
    ax.set_ylim(-45, 70)
    ax.set_xlabel("20-day realised volatility (annualised %)")
    ax.set_ylabel("63-day return (%)")
    ax.set_title("How the regimes are defined")
    ax.legend(loc="upper right", frameon=False, markerscale=2.5)
    save(fig, "eda-label-rule.png", "Every day plotted by the two measurements that define the regimes. The lines are the thresholds: zero return and 18% "
                                    "volatility. The colour is the confirmed regime, so points near a line can carry the neighbouring colour.")

    # 5. Regimes by year and how long they last ----------------------------------------------------
    by_year = full.groupby(full.index.year)["regime"].value_counts(normalize=True).unstack().reindex(columns=REGIMES).fillna(0) * 100
    runs = (full["regime"] != full["regime"].shift()).cumsum()
    run_len = full.groupby(runs).agg(regime=("regime", "first"), days=("regime", "size"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4.3), gridspec_kw={"width_ratios": [1.6, 1]})
    bottom = np.zeros(len(by_year))
    for name in REGIMES:
        a.bar(by_year.index, by_year[name], bottom=bottom, color=COLORS[name], width=0.82, edgecolor="white", linewidth=1)
        bottom += by_year[name].to_numpy()
    a.set_ylabel("share of the year's days (%)")
    a.set_title("Regimes by year")
    a.set_xticks(range(2008, 2027, 2))
    a.tick_params(axis="x", labelsize=8.5)
    a.set_ylim(0, 100)
    a.grid(axis="x", visible=False)
    data_by = [run_len.loc[run_len["regime"] == r, "days"] for r in REGIMES]
    bp = b.boxplot(data_by, orientation="horizontal", tick_labels=REGIMES, patch_artist=True, showfliers=False, whis=(10, 90),
                   medianprops={"color": INK, "linewidth": 1.5})
    for patch, name in zip(bp["boxes"], REGIMES):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.85)
    b.invert_yaxis()
    b.set_xlabel("length of one spell (trading days)")
    b.set_title("How long a regime lasts")
    b.grid(axis="y", visible=False)
    a.legend(_legend_patches(), REGIMES, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False, fontsize=8.5)
    save(fig, "eda-regime-durations.png", "Left: the mix of regimes in each year (2008 is almost all Stress, 2014 and 2017 almost all Calm rally). Right: Calm rally spells "
                                           "last longest (median about 60 days); the other three typically last under a month, which is part of why they are harder to learn.")

    # 6. Regime transitions ------------------------------------------------------------------------
    pairs = pd.DataFrame({"from": full["regime"].shift()[full["regime"] != full["regime"].shift()],
                          "to": full["regime"][full["regime"] != full["regime"].shift()]}).dropna()
    mat = pd.crosstab(pairs["from"], pairs["to"]).reindex(index=REGIMES, columns=REGIMES).fillna(0)
    share = mat.div(mat.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.imshow(share.to_numpy(), cmap="Purples", vmin=0, vmax=0.8)
    ax.grid(False)
    ax.set_xticks(range(4), REGIMES, rotation=20)
    ax.set_yticks(range(4), REGIMES)
    for i in range(4):
        for j in range(4):
            if i != j:
                ax.text(j, i, f"{share.iloc[i, j]:.0%}\n({int(mat.iloc[i, j])})", ha="center", va="center",
                        color="white" if share.iloc[i, j] > 0.45 else INK)
            else:
                ax.text(j, i, "-", ha="center", va="center", color=MUTED)
    ax.set_xlabel("Regime it switches to")
    ax.set_ylabel("Regime it leaves")
    ax.set_title("Where each regime goes next")
    save(fig, "eda-transitions.png", f"Across the {int(mat.to_numpy().sum())} regime changes, what follows what. Stress usually gives way to Grind down (58%) or a Volatile rally (32%); "
                                      "a Calm rally mostly slips into Grind down (56%) or turns into a Volatile rally (32%) and rarely jumps straight to Stress (12%).")

    # 7. Feature history ---------------------------------------------------------------------------
    lab = table
    fig, axes = plt.subplots(len(features.FEATURES), 1, figsize=(8.2, 9.2), sharex=True)
    for ax, col in zip(axes, features.FEATURES):
        ax.plot(table.index, table[col], color=INK, lw=0.7)
        _shade(ax, lab, alpha=0.14, only=["Stress"])
        ax.set_ylabel(NICE[col].split(" (")[0], rotation=0, ha="right", va="center", fontsize=9)
        ax.tick_params(labelsize=8.5)
    axes[0].set_title("The seven features over time (Stress days shaded)")
    save(fig, "eda-feature-history.png", "Each feature from 2008 to 2026. VIX, drawdown and the distance from the 200-day average all swing hard in the shaded Stress periods; "
                                          "the 5-day return and the VIX changes are noisy and look much the same in every regime.")

    # 8. Feature distributions with the form ranges ------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(9.2, 4.9))
    for ax, inp in zip(axes.ravel(), train.INPUTS):
        col = inp["name"]
        ax.hist(table[col], bins=60, color=ACCENT, alpha=0.9)
        for edge in (inp["min"], inp["max"]):
            if table[col].min() < edge or table[col].max() > edge:
                ax.axvline(edge, color=COLORS["Stress"], lw=1, ls="--")
        ax.set_title(SHORT[col], fontsize=9.5, loc="left")
        ax.tick_params(labelsize=8)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(0, 0.6, "Dashed red lines:\nthe range the site form\naccepts. Values beyond\nthem were capped.", fontsize=9, color=INK)
    save(fig, "eda-feature-distributions.png", "How each feature is distributed. VIX and the VIX measures are skewed to the right (calm most of the time, occasional spikes); "
                                               "drawdown is bunched near zero. Dashed lines show the form ranges; few days fall outside them.")

    # 9. Features by regime -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(9.4, 5.2))
    for ax, col in zip(axes.ravel(), features.FEATURES):
        groups = [table.loc[table["regime"] == r, col] for r in REGIMES]
        bp = ax.boxplot(groups, patch_artist=True, showfliers=False, whis=(5, 95), medianprops={"color": INK, "linewidth": 1.4},
                        widths=0.65)
        for patch, name in zip(bp["boxes"], REGIMES):
            patch.set_facecolor(COLORS[name])
            patch.set_alpha(0.85)
        ax.set_xticks([])
        ax.set_title(SHORT[col], fontsize=9.5, loc="left")
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", visible=False)
    last = axes.ravel()[-1]
    last.axis("off")
    last.legend(_legend_patches(), REGIMES, loc="center left", frameon=False, fontsize=9)
    save(fig, "eda-features-by-regime.png", "Each feature split by regime (boxes show the middle half, whiskers the 5th to 95th percentile). Drawdown and distance from the "
                                             "200-day average separate the regimes clearly; the 5-day return and VIX 5-day change barely differ between them.")

    # 10. Scatter pairs ---------------------------------------------------------------------------
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 4.4))
    for name in REGIMES:
        s = table[table["regime"] == name]
        a.scatter(s["india_vix"], s["drawdown_pct"], s=6, alpha=0.4, color=COLORS[name], lw=0, label=name)
        b.scatter(s["india_vix"], s["dist_200dma_pct"], s=6, alpha=0.4, color=COLORS[name], lw=0)
    a.set_xlabel("India VIX")
    a.set_ylabel("Drawdown from 1-year high (%)")
    b.set_xlabel("India VIX")
    b.set_ylabel("Distance from 200-day average (%)")
    a.set_xlim(8, 60)
    b.set_xlim(8, 60)
    b.axhline(0, color=INK, lw=0.8)
    a.legend(frameon=False, markerscale=2.5, fontsize=9, loc="lower left")
    a.set_title("VIX and drawdown")
    b.set_title("VIX and the 200-day average")
    save(fig, "eda-feature-pairs.png", "Two features at a time. The regimes occupy different parts of the plane: the long-term trend (vertical position in the right panel) splits the rallies "
                                       "from the declines, and VIX splits the calm from the turbulent. The overlap at the edges is where the model makes its mistakes.")

    # 11. Correlation ------------------------------------------------------------------------------
    corr = table[features.FEATURES].corr()
    fig, ax = plt.subplots(figsize=(7, 5.6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(7), [NICE[c].replace(" (%)", "") for c in features.FEATURES], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(7), [NICE[c].replace(" (%)", "") for c in features.FEATURES], fontsize=9)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else INK)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("How the features move together")
    save(fig, "eda-correlation.png", "Correlation between the features. Drawdown, distance from the 200-day average and the 50/200-day gap are strongly related (0.76 to 0.89), "
                                     "so they overlap; the VIX features and the 5-day return are largely independent of them.")

    # 12. How informative is each feature ---------------------------------------------------------
    y = table["regime"].map({r: i for i, r in enumerate(REGIMES)}).to_numpy()
    mi = pd.Series(mutual_info_classif(table[features.FEATURES], y, random_state=train.SEED), index=features.FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.barh([NICE[c].replace(" (%)", "") for c in mi.index], mi.to_numpy(), color=ACCENT, height=0.6)
    ax.set_xlabel("mutual information with the regime (higher means more informative)")
    ax.set_title("How much each feature says about the regime")
    ax.grid(axis="y", visible=False)
    save(fig, "eda-feature-relevance.png", "Mutual information between each feature and the regime, on its own. The three long-term trend measures (distance from the "
                                           "200-day average, drawdown, 50/200-day gap) say the most, then VIX; the 5-day return and the two VIX-change features say very little by themselves.")

    return charts, mi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    out = args.repo_root / "src" / "MLPortfolio.Web" / "wwwroot" / "projects" / "nse-regime" / "charts"
    charts, mi = build_all(out, refresh=args.refresh)
    print(mi.round(3).to_string())
    print(json.dumps(charts, indent=2))


if __name__ == "__main__":
    main()
