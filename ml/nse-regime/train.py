"""NSE market regime classifier: step 4, model building.

Classifies the current regime (Calm rally, Volatile rally, Grind down, Stress) from the seven features in
features.py. Pipeline: audit and features -> walk-forward evaluation of several candidates -> fit the chosen
model on all days -> export ONNX -> verify with onnxruntime -> write the artefacts the ASP.NET site consumes:

  <web>/App_Data/models/nse-regime/{model.onnx, schema.json}
  <web>/wwwroot/projects/nse-regime/charts/*.png
  <web>/App_Data/models/nse-regime/snapshot.json   (latest day's features: the page shows the model's reading of them)
  <tests>/fixtures/nse-regime.json, nse-regime-snapshot.json   (reference outputs for the .NET parity tests)

Run:  uv run python train.py [--refresh]
There is no prediction form: the page shows one sentence, the model's reading of the latest day (`inferenceMode: snapshot`).
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import data
import features
import labels

SLUG = "nse-regime"
MODEL_VERSION = "1.0.0"
SEED = 42
LABELS = labels.REGIMES  # index = class id
ONNX_TOLERANCE = 1e-4
N_FOLDS = 6
GAP = 63   # trading days skipped between train and test: the label's trend window is 63 days long

# Input definition. Single source of truth for the generated schema.json and the ONNX input names.
INPUTS = [
    {"name": "india_vix", "label": "India VIX", "min": 8, "max": 60, "step": 0.5, "default": 16,
     "help": "The market's expected volatility over the next 30 days (index points)"},
    {"name": "vix_vs_avg_pct", "label": "VIX vs its 60-day average (%)", "min": -40, "max": 120, "step": 1, "default": 0,
     "help": "Positive means fear is higher than it has been recently"},
    {"name": "vix_change_5d_pct", "label": "VIX change over 5 days (%)", "min": -40, "max": 100, "step": 1, "default": 0,
     "help": "How fast fear moved in the last week"},
    {"name": "drawdown_pct", "label": "Nifty 200 below its 1-year high (%)", "min": -60, "max": 0, "step": 0.5, "default": -5,
     "help": "0 means at a 1-year high; -10 means 10% below it"},
    {"name": "dist_200dma_pct", "label": "Nifty 200 vs its 200-day average (%)", "min": -40, "max": 40, "step": 0.5, "default": 4,
     "help": "Positive means above the long-term average"},
    {"name": "ma50_vs_ma200_pct", "label": "50-day vs 200-day average (%)", "min": -30, "max": 30, "step": 0.5, "default": 3,
     "help": "Positive means the medium-term average is above the long-term one"},
    {"name": "return_5d_pct", "label": "Nifty 200 5-day return (%)", "min": -15, "max": 15, "step": 0.5, "default": 0.5,
     "help": "Index change over the last 5 trading days"},
]
COLUMNS = [i["name"] for i in INPUTS]
NICE_NAMES = {i["name"]: i["label"].replace(" (%)", "") for i in INPUTS}
assert COLUMNS == features.FEATURES, "INPUTS must match features.FEATURES"

# Filled in from the most confident correct walk-forward days (printed by `train.py`), then rounded.
PRESETS = [
    {"name": "Calm market", "values": {"india_vix": 13, "vix_vs_avg_pct": -8, "vix_change_5d_pct": -2, "drawdown_pct": -1,
                                       "dist_200dma_pct": 8, "ma50_vs_ma200_pct": 6, "return_5d_pct": 1}},
    {"name": "Panic sell-off", "values": {"india_vix": 45, "vix_vs_avg_pct": 60, "vix_change_5d_pct": 25, "drawdown_pct": -30,
                                          "dist_200dma_pct": -20, "ma50_vs_ma200_pct": -12, "return_5d_pct": -6}},
    {"name": "Slow decline", "values": {"india_vix": 17, "vix_vs_avg_pct": 2, "vix_change_5d_pct": 1, "drawdown_pct": -10,
                                        "dist_200dma_pct": -2, "ma50_vs_ma200_pct": -1, "return_5d_pct": -0.5}},
    {"name": "Nervous rebound", "values": {"india_vix": 28, "vix_vs_avg_pct": -10, "vix_change_5d_pct": -8, "drawdown_pct": -8,
                                           "dist_200dma_pct": 14, "ma50_vs_ma200_pct": 8, "return_5d_pct": 3}},
]

ACCENT = "#4f46e5"
MUTED = "#78716c"
REGIME_COLORS = [labels.COLORS[r] for r in LABELS]


# ---------------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------------

def _scaler() -> ColumnTransformer:
    return ColumnTransformer([("num", StandardScaler(), COLUMNS)])


def build_logistic(class_weight=None) -> Pipeline:
    return Pipeline([("pre", _scaler()),
                     ("clf", LogisticRegression(max_iter=5000, class_weight=class_weight, random_state=SEED))])


def build_forest() -> Pipeline:
    return Pipeline([("pre", _scaler()), ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=SEED, n_jobs=-1))])


def build_boosting() -> Pipeline:
    return Pipeline([("pre", _scaler()), ("clf", HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED))])


CANDIDATES = {   # name -> builder; CHOSEN (below) is the one exported to the site
    "Logistic regression": build_logistic,
    "Logistic regression, balanced classes": lambda: build_logistic("balanced"),
    "Random forest": build_forest,
    "Gradient boosting": build_boosting,
}
CHOSEN = "Logistic regression, balanced classes"


def to_onnx(pipeline: Pipeline) -> bytes:
    """One ONNX input per feature (float [N,1])."""
    initial_types = [(c, FloatTensorType([None, 1])) for c in COLUMNS]
    clf = pipeline.steps[-1][1]
    model = convert_sklearn(pipeline, initial_types=initial_types, options={id(clf): {"zipmap": False}},
                            target_opset={"": 17, "ai.onnx.ml": 3})
    return model.SerializeToString()


def onnx_predict(model_bytes: bytes, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    feed = {c: df[c].to_numpy(dtype=np.float32).reshape(-1, 1) for c in COLUMNS}
    out_labels, probs = session.run(None, feed)
    return out_labels, probs


# ---------------------------------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------------------------------

@dataclass
class Result:
    pipeline: Pipeline
    model_bytes: bytes
    table: pd.DataFrame            # all usable days: features (capped) and regime
    y: np.ndarray
    folds: list                    # [(train_idx, test_idx)]
    oos: dict                      # candidate name -> pooled out-of-sample probabilities (n_days x 4, NaN where unused)
    mask: np.ndarray               # days that were in some test fold
    audit: features.Audit
    metrics: dict


def _walk_forward(builder, x: pd.DataFrame, y: np.ndarray, folds) -> np.ndarray:
    proba = np.full((len(y), len(LABELS)), np.nan)
    for train_idx, test_idx in folds:
        model = builder().fit(x.iloc[train_idx], y[train_idx])
        p = np.zeros((len(test_idx), len(LABELS)))
        p[:, model.classes_] = model.predict_proba(x.iloc[test_idx])
        proba[test_idx] = p
    return proba


def _scores(y_true: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(axis=1)
    idx = list(range(len(LABELS)))
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", labels=idx, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba + 1e-9, multi_class="ovr", labels=idx)),
        "recall": {LABELS[i]: float(v) for i, v in enumerate(recall_score(y_true, pred, labels=idx, average=None, zero_division=0))},
        "precision": {LABELS[i]: float(v) for i, v in enumerate(precision_score(y_true, pred, labels=idx, average=None, zero_division=0))},
    }


def train_and_evaluate(refresh: bool = False) -> Result:
    table, audit = features.prepare(data.load_raw(refresh=refresh))
    capped = 0
    for inp in INPUTS:   # keep training values inside what the site form accepts
        col = table[inp["name"]]
        capped += int(((col < inp["min"]) | (col > inp["max"])).sum())
        table[inp["name"]] = col.clip(inp["min"], inp["max"])
    x = table[COLUMNS]
    y = table["regime"].map({r: i for i, r in enumerate(LABELS)}).to_numpy()

    folds = list(TimeSeriesSplit(n_splits=N_FOLDS, gap=GAP).split(x))
    oos = {name: _walk_forward(builder, x, y, folds) for name, builder in CANDIDATES.items()}
    mask = ~np.isnan(oos[CHOSEN][:, 0])
    y_test = y[mask]

    comparison = {name: _scores(y_test, p[mask]) for name, p in oos.items()}
    majority_acc, fold_majority = [], []
    for train_idx, test_idx in folds:
        majority = np.bincount(y[train_idx], minlength=len(LABELS)).argmax()
        fold_majority.append(float((y[test_idx] == majority).mean()))
    pooled_majority = np.concatenate([np.full(len(te), np.bincount(y[tr], minlength=4).argmax()) for tr, te in folds])
    majority_acc = float((y_test == pooled_majority).mean())
    comparison["Always the most common regime"] = {
        "accuracy": majority_acc,
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pooled_majority)),
        "macro_f1": float(f1_score(y_test, pooled_majority, average="macro", labels=list(range(4)), zero_division=0)),
    }

    chosen = oos[CHOSEN][mask]
    fold_accuracy = {name: [float(accuracy_score(y[te], p[te].argmax(axis=1))) for _, te in folds]
                     for name, p in oos.items()}
    prev_day = table["regime"].shift(1).map({r: i for i, r in enumerate(LABELS)}).to_numpy()
    prev_ref = float((prev_day[mask][1:] == y_test[1:]).mean())

    pipeline = CANDIDATES[CHOSEN]().fit(x, y)   # final model: all usable days
    model_bytes = to_onnx(pipeline)
    _, onnx_proba = onnx_predict(model_bytes, x)
    sk_proba = pipeline.predict_proba(x)
    max_diff = float(np.abs(onnx_proba - sk_proba).max())
    agree = float((onnx_proba.argmax(axis=1) == sk_proba.argmax(axis=1)).mean())
    assert max_diff < ONNX_TOLERANCE, f"ONNX and sklearn disagree (max diff {max_diff})"
    assert agree > 0.999, f"ONNX and sklearn predict different classes for {1 - agree:.2%} of rows"

    fold_ranges = [f"{table.index[te[0]].year}-{table.index[te[-1]].year}" for _, te in folds]
    metrics = {
        "chosen": CHOSEN,
        "n_days": int(len(table)), "n_oos": int(mask.sum()),
        "period": f"{table.index.min().date()} to {table.index.max().date()}",
        "oos_period": f"{table.index[np.where(mask)[0][0]].date()} to {table.index[-1].date()}",
        "folds": N_FOLDS, "gap_days": GAP, "fold_ranges": fold_ranges,
        "values_capped_to_form_range": capped,
        "class_share": {LABELS[i]: float((y == i).mean()) for i in range(len(LABELS))},
        "comparison": comparison,
        "chosen_scores": comparison[CHOSEN],
        "fold_accuracy": fold_accuracy, "fold_majority_accuracy": fold_majority,
        "previous_day_label_accuracy": prev_ref,
        "confusion_matrix": confusion_matrix(y_test, chosen.argmax(axis=1), labels=list(range(4))).tolist(),
        "onnx_max_abs_diff": max_diff,
    }
    return Result(pipeline, model_bytes, table, y, folds, oos, mask, audit, metrics)


def build_schema() -> dict:
    return {
        "modelFile": "model.onnx",
        "modelVersion": MODEL_VERSION,
        "inputs": [{"type": "number", **i} for i in INPUTS],
        "outputs": {"type": "classification", "labels": LABELS},
        "presets": PRESETS,
    }


def build_fixtures(result: Result, n_random: int = 6) -> list[dict]:
    rng = np.random.default_rng(SEED + 1)
    cases = [dict(p["values"]) for p in PRESETS]
    for _ in range(n_random):
        cases.append({i["name"]: round(float(rng.uniform(i["min"], i["max"])) / i["step"]) * i["step"] for i in INPUTS})
    df = pd.DataFrame(cases)[COLUMNS]
    proba = result.pipeline.predict_proba(df)
    return [{
        "input": {k: float(v) for k, v in row.items()},
        "label": LABELS[int(np.argmax(p))],
        "probabilities": {LABELS[i]: float(p[i]) for i in range(len(LABELS))},
    } for row, p in zip(df.to_dict("records"), proba)]


def build_snapshot(result: Result) -> tuple[dict, dict]:
    """The latest day's feature values (rounded to two decimals, inside the form ranges). The site runs the exported
    ONNX model on them to show 'the model predicts the current regime as ...'. Returns (snapshot.json, test fixture)."""
    last = result.table.iloc[-1]
    values = {c: round(float(last[c]), 2) for c in COLUMNS}
    row = pd.DataFrame([values])[COLUMNS]
    proba = result.pipeline.predict_proba(row)[0]
    _, onnx_proba = onnx_predict(result.model_bytes, row)
    assert int(np.argmax(onnx_proba[0])) == int(np.argmax(proba)), "ONNX and sklearn disagree on the latest day"
    snapshot = {"asOf": str(result.table.index[-1].date()), "features": values}
    fixture = {"asOf": snapshot["asOf"], "label": LABELS[int(np.argmax(proba))],
               "probabilities": {LABELS[i]: float(proba[i]) for i in range(len(LABELS))}}
    return snapshot, fixture


def confident_examples(result: Result) -> dict:
    """For choosing presets: per regime, the walk-forward day the model got right with the highest probability."""
    p = result.oos[CHOSEN]
    out = {}
    for i, name in enumerate(LABELS):
        ok = result.mask & (result.y == i) & (np.nan_to_num(p).argmax(axis=1) == i)
        if ok.any():
            j = np.where(ok)[0][np.nanargmax(p[ok, i])]
            out[name] = {"date": str(result.table.index[j].date()), "p": round(float(p[j, i]), 2),
                         **{c: round(float(result.table[c].iloc[j]), 1) for c in COLUMNS}}
    return out


# ---------------------------------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------------------------------

def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "sans-serif", "font.size": 11, "axes.edgecolor": "#d6d3d1", "axes.labelcolor": "#44403c",
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 13, "axes.grid": True, "grid.color": "#f0efed",
    })


def make_thumbnail(path: Path) -> None:
    """The homepage card image: 16:9 (the card's shape, so nothing is cropped), no axes or small print.
    The Nifty 200 over its whole history on a log scale, with each regime as a coloured band."""
    full = labels.build_labels(data.load_raw())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.axis("off")
    ax.margins(x=0)
    ax.plot(full.index, full["close"], color="#1c1917", lw=2.2, zorder=3)
    ax.set_yscale("log")
    lo, hi = full["close"].min(), full["close"].max()
    ax.set_ylim(lo / 1.35, hi * 1.25)
    for name in LABELS:
        ax.fill_between(full.index, 0, 1, where=(full["regime"] == name).to_numpy(), transform=ax.get_xaxis_transform(),
                        color=labels.COLORS[name], alpha=0.42, lw=0, step="mid", zorder=1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=labels.COLORS[n], alpha=0.8) for n in LABELS]
    leg = ax.legend(handles, LABELS, loc="upper left", ncol=2, frameon=True, fontsize=13, borderpad=0.7,
                    labelspacing=0.5, handlelength=1.2, columnspacing=1.2, facecolor="white", edgecolor="none",
                    framealpha=0.85, bbox_to_anchor=(0.03, 0.97))
    leg.set_zorder(5)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def make_charts(result: Result, out_dir: Path) -> list[dict]:
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    charts: list[dict] = []
    m, table = result.metrics, result.table
    chosen_p = result.oos[CHOSEN]
    pred_all = np.where(result.mask, np.nan_to_num(chosen_p).argmax(axis=1), -1)

    def save(fig, name: str, caption: str) -> None:
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=200)
        plt.close(fig)
        charts.append({"file": name, "caption": caption})

    # 1. Regime timeline: the index, the true regimes and the walk-forward predictions
    sel = result.mask
    dates = table.index[sel]
    fig, (ax, strip) = plt.subplots(2, 1, figsize=(10.5, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    ax.plot(dates, table["india_vix"][sel], color="#222", lw=0.9)
    ax.set_ylabel("India VIX")
    ax.set_title("Regimes over time: actual and predicted")
    for row, (series, name) in enumerate([(result.y[sel], "Actual"), (pred_all[sel], "Predicted")]):
        for i, color in enumerate(REGIME_COLORS):
            strip.fill_between(dates, 1 - row, 2 - row, where=(series == i), color=color, lw=0, step="mid")
    strip.set_yticks([0.5, 1.5], ["Predicted", "Actual"])
    strip.set_ylim(0, 2)
    strip.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in REGIME_COLORS]
    ax.legend(handles, LABELS, loc="upper left", ncol=4, frameon=False, fontsize=9)
    save(fig, "regime-timeline.png",
         "India VIX with the actual regime (top strip) and the regime predicted for each day by a model that had only "
         "seen earlier data (bottom strip).")

    # 2. Confusion matrix (pooled walk-forward)
    cm = np.array(m["confusion_matrix"])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    ax.imshow(cmn, cmap="Purples", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(4), LABELS, rotation=20)
    ax.set_yticks(range(4), LABELS)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cmn[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.55 else "#222", fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (walk-forward)")
    save(fig, "confusion-matrix.png",
         "Each row shows how the days of one actual regime were classified. Volatile rally is recognised only 15% of the time, and Stress is often mistaken for Grind down.")

    # 3. Model comparison
    names = ["Always the most common regime", "Logistic regression", "Logistic regression, balanced classes",
             "Random forest", "Gradient boosting"]
    comp = m["comparison"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y_pos = np.arange(len(names))
    ax.barh(y_pos - 0.2, [comp[n]["accuracy"] for n in names], 0.38, color=ACCENT, label="Accuracy")
    ax.barh(y_pos + 0.2, [comp[n]["balanced_accuracy"] for n in names], 0.38, color="#a5b4fc", label="Balanced accuracy")
    ax.set_yticks(y_pos, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right")
    ax.set_title("Models compared (walk-forward)")
    save(fig, "model-comparison.png",
         "Accuracy and balanced accuracy (the average of each regime's recall) for each candidate. Weighting the rare regimes "
         "more heavily helps the logistic regression most.")

    # 4. Coefficients
    coef = result.pipeline.named_steps["clf"].coef_
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    lim = np.abs(coef).max()
    im = ax.imshow(coef, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(COLUMNS)), [c.replace("_pct", "").replace("_", " ") for c in COLUMNS], rotation=30, ha="right")
    ax.set_yticks(range(4), LABELS)
    for i in range(4):
        for j in range(len(COLUMNS)):
            ax.text(j, i, f"{coef[i, j]:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(coef[i, j]) > 0.6 * lim else "#222")
    ax.set_title("What pushes each regime up or down")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, "coefficients.png",
         "Standardised logistic regression coefficients: red pushes the model towards that regime, blue away from it.")

    # 5. Accuracy by fold
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x_pos = np.arange(N_FOLDS)
    ax.bar(x_pos - 0.2, m["fold_accuracy"][CHOSEN], 0.38, color=ACCENT, label="Balanced logistic regression")
    ax.bar(x_pos + 0.2, m["fold_majority_accuracy"], 0.38, color="#d6d3d1", label="Most common regime")
    ax.set_xticks(x_pos, m["fold_ranges"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right")
    ax.set_title("Accuracy in each test period")
    save(fig, "accuracy-by-period.png",
         "Each test period is scored by a model trained only on earlier data. It beats the simple guess in five of six "
         "periods; the exception is the calm 2016 to 2018 stretch, where always guessing Calm rally is hard to beat.")

    # 6. Probability of Stress through time
    j = LABELS.index("Stress")
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.fill_between(dates, 0, 1, where=(result.y[sel] == j), color=labels.COLORS["Stress"], alpha=0.25, lw=0,
                    step="mid", label="Actual Stress")
    ax.plot(dates, chosen_p[sel, j], color=ACCENT, lw=1.0, label="Predicted probability of Stress")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Probability of Stress")
    save(fig, "stress-probability.png",
         "The model's probability of Stress each day (line) against the days that really were Stress (shaded).")

    make_thumbnail(out_dir / "thumbnail.png")
    return charts


def training_report(result: Result) -> str:
    """Markdown tables describing how the model was trained and tested (pasted into the site write-up)."""
    from sklearn.metrics import log_loss

    m, table, y = result.metrics, result.table, result.y
    idx = table.index
    clf = result.pipeline.named_steps["clf"]
    chosen = result.oos[CHOSEN]
    fold_rows = []
    for k, (tr, te) in enumerate(result.folds):
        acc = m["fold_accuracy"][CHOSEN][k]
        fold_rows.append(
            f"| {k + 1} | {idx[tr[0]].strftime('%b %Y')} to {idx[tr[-1]].strftime('%b %Y')} | {len(tr):,} | "
            f"{idx[te[0]].strftime('%b %Y')} to {idx[te[-1]].strftime('%b %Y')} | {len(te):,} | "
            f"{m['fold_majority_accuracy'][k]:.1%} | {acc:.1%} |")
    cm = np.array(m["confusion_matrix"])
    support = cm.sum(axis=1)
    per_class = []
    sc = m["chosen_scores"]
    for i, name in enumerate(LABELS):
        r, pr = sc["recall"][name], sc["precision"][name]
        f1 = 2 * r * pr / (r + pr) if r + pr else 0.0
        per_class.append(f"| {name} | {pr:.1%} | {r:.1%} | {f1:.2f} | {support[i]:,} |")
    coef = clf.coef_
    coef_rows = [f"| {NICE_NAMES[c]} | " + " | ".join(f"{coef[i, j]:+.2f}" for i in range(len(LABELS))) + " |"
                 for j, c in enumerate(COLUMNS)]
    intercepts = " | ".join(f"{v:+.2f}" for v in clf.intercept_)
    counts = np.bincount(y, minlength=len(LABELS))
    weights = len(y) / (len(LABELS) * counts)
    mask = result.mask
    ll = log_loss(y[mask], np.clip(chosen[mask], 1e-9, 1), labels=list(range(len(LABELS))))
    brier = float(np.mean(np.sum((chosen[mask] - np.eye(len(LABELS))[y[mask]]) ** 2, axis=1)))
    in_sample = float(accuracy_score(y, result.pipeline.predict(table[COLUMNS])))
    lines = [
        "| Fold | Trained on | Training days | Tested on | Test days | Most-common-regime guess | Model |",
        "|---|---|---|---|---|---|---|", *fold_rows,
        "",
        "| Setting | Value |", "|---|---|",
        f"| Algorithm | multinomial logistic regression (`LogisticRegression`, L2 penalty, C = {clf.C:g}) |",
        f"| Solver | {clf.solver}, converged in {int(np.max(clf.n_iter_))} iterations (limit {clf.max_iter:,}) |",
        "| Preprocessing | each feature standardised (mean 0, standard deviation 1), fitted on the training days only |",
        "| Class weights | " + ", ".join(f"{n} {w:.2f}" for n, w in zip(LABELS, weights)) + " (balanced: rarer regimes count for more) |",
        f"| Final training set | {len(y):,} days, {idx[0].strftime('%d %b %Y')} to {idx[-1].strftime('%d %b %Y')}; {m['values_capped_to_form_range']} feature values capped to the form range |",
        f"| Walk-forward folds | {m['folds']} expanding windows, {m['gap_days']}-day gap before each test period |",
        f"| Out-of-sample days | {m['n_oos']:,} ({m['oos_period'].replace(' to ', ' to ')}) |",
        f"| Random seed | {SEED} |",
        "",
        "| Regime | Precision | Recall | F1 | Test days |", "|---|---|---|---|---|", *per_class,
        "",
        "| Feature (standardised) | " + " | ".join(LABELS) + " |", "|---|---|---|---|---|", *coef_rows,
        f"| *Intercept* | {intercepts} |",
        "",
        f"Probability quality on the out-of-sample days: log loss {ll:.3f} (a model that always gave the training-period regime shares scores about "
        f"{log_loss(y[mask], np.tile(counts / counts.sum(), (int(mask.sum()), 1))):.3f}), Brier score {brier:.3f}. "
        f"Accuracy on the training days themselves is {in_sample:.1%}, against {m['chosen_scores']['accuracy']:.1%} out of sample.",
    ]
    return chr(10).join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2],
                        help="repository root containing src/ and tests/ (default: two levels up)")
    parser.add_argument("--refresh", action="store_true", help="download fresh data first")
    parser.add_argument("--report", action="store_true", help="print the training-stats tables used in the write-up and exit")
    args = parser.parse_args()
    web = args.repo_root / "src" / "MLPortfolio.Web"

    result = train_and_evaluate(refresh=args.refresh)
    if args.report:
        print(training_report(result))
        return
    print(json.dumps(result.metrics, indent=2))
    print("confident examples (for presets):")
    print(json.dumps(confident_examples(result), indent=2))

    model_dir = web / "App_Data" / "models" / SLUG
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.onnx").write_bytes(result.model_bytes)
    (model_dir / "schema.json").write_text(json.dumps(build_schema(), indent=2) + "\n", encoding="utf-8")

    fixtures_dir = args.repo_root / "tests" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    (fixtures_dir / f"{SLUG}.json").write_text(json.dumps(build_fixtures(result), indent=2) + "\n", encoding="utf-8")

    snapshot, snapshot_fixture = build_snapshot(result)
    (model_dir / "snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    (fixtures_dir / f"{SLUG}-snapshot.json").write_text(json.dumps(snapshot_fixture, indent=2) + "\n", encoding="utf-8")
    print("current regime reading:", snapshot_fixture["asOf"], snapshot_fixture["label"],
          {k: round(v, 3) for k, v in snapshot_fixture["probabilities"].items()})

    charts = make_charts(result, web / "wwwroot" / "projects" / SLUG / "charts")
    print("charts (paste into projects.json):")
    print(json.dumps(charts, indent=2))
    print(f"wrote model, schema, fixtures and charts for '{SLUG}'")


if __name__ == "__main__":
    main()
