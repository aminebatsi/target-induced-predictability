"""Does the target-transform effect generalise across filters and horizons?

The headline result of this project is measured with one smoother (the causal
Kalman local linear trend) at one horizon (h = 7). Neither choice should matter
if the effect is really a property of smoothing rather than of that particular
smoother, so this module re-runs the three-truths comparison over the full
filter registry and over h in {3, 7, 14}.

Three forecasters are scored on every combination:

  SLOPE   d_hat = h * (level_t - level_{t-1}). Fits nothing. Included because
          it is the sharpest available statement of the effect: if a rule with
          no parameters tracks the fitted models, the accuracy is coming from
          the target.
  LR      the flattened-window linear model, which is within 0.001 of the best
          model in the main table and costs almost nothing to fit.
  GRU     optional (--gru), the best model in the main table. On a T4 the full
          sweep is a few hours, so it is off by default.

Every combination is scored against the same three truths as the main table:
the filtered trend, the raw h-day price change, and the next-day return.

The leaky filters are the point of the exercise, not an oversight. They show
what a study measures when the target transform is applied to the whole series
before splitting, which is what leakage_suite.py flags and what the wavelet
frameworks in the literature do.

The 90-column linear design is strongly collinear, so LR uses the same fixed
Ridge penalty as the main forecast table rather than an unregularised
least-squares solve. SLOPE remains the primary reference because it has no
fitted coefficients at all.

Output -> results/horizons/
"""
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ASSETS, FOLDS, HORIZONS, L, PLOT_RC, RESULTS
from evaluation import dir_acc, fold_indices, pt_test
from filters import FILTERS, apply_filter
from forecast import _norm_end, _series
from models import predict
from targets import build_windows

OUT = RESULTS / "horizons"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(PLOT_RC)


def _fold_arrays(asset, ts, tz, filt, h):
    """Windows, target and the three truths for one (asset, fold, filter, h)."""
    dates, lp = _series(asset, tz)
    if len(lp) < 700:
        return None
    norm_end = _norm_end(dates, ts, tz, h=h)
    if norm_end is None:
        return None
    trend = apply_filter(filt, lp, norm_end)
    Xtab, Xseq, anchors = build_windows(lp, trend, min_anchor=L - 1, h=h)
    idx = fold_indices(dates[anchors], ts, tz, h=h)
    if idx is None:
        return None
    a = anchors
    d = trend[a + h] - trend[a]                       # the learning target
    slope = h * (trend[a] - trend[a - 1])             # the free baseline
    truths = dict(trend=d,
                  price_h=lp[a + h] - lp[a],
                  price_1=lp[a + 1] - lp[a])
    return Xtab, Xseq, d, slope, truths, idx


def run(filt, h, models=("SLOPE", "LR")):
    """One (filter, horizon) cell: pooled accuracy for each forecaster."""
    acc = {m: {k: [] for k in ("trend", "price_h", "price_1")} for m in models}
    n_tot = 0
    for asset in ASSETS:
        for ts, tz in FOLDS:
            fa = _fold_arrays(asset, ts, tz, filt, h)
            if fa is None:
                continue
            Xtab, Xseq, d, slope, truths, (tr, va, te) = fa
            preds = {}
            if "SLOPE" in models:
                preds["SLOPE"] = slope[te]
            for m in models:
                if m == "SLOPE":
                    continue
                preds[m] = predict(m, Xtab, Xseq, d, tr, va, te)[1]
            n_tot += len(te)
            for m, yp in preds.items():
                for k, y in truths.items():
                    acc[m][k].append((np.sign(yp) == np.sign(y[te])).sum())
    rows = []
    for m in models:
        r = {"filter": filt, "label": FILTERS[filt]["label"],
             "causal": FILTERS[filt]["causal"], "h": h, "model": m, "n": n_tot}
        for k in ("trend", "price_h", "price_1"):
            r[f"DA_{k}"] = sum(acc[m][k]) / n_tot if n_tot else np.nan
        r["gap"] = r["DA_trend"] - r["DA_price_1"]
        rows.append(r)
    return rows


def main(use_gru=False):
    models = ("SLOPE", "LR") + (("GRU",) if use_gru else ())
    print(f"filters {len(FILTERS)} x horizons {HORIZONS} x models {models}")
    out = []
    for filt in FILTERS:
        for h in HORIZONS:
            rows = run(filt, h, models)
            out.extend(rows)
            tag = "causal" if FILTERS[filt]["causal"] else "LEAKY "
            for r in rows:
                print(f"  [{tag}] {filt:11} h={h:<3} {r['model']:6} "
                      f"trend={r['DA_trend']:.4f}  price_h={r['DA_price_h']:.4f}  "
                      f"price_1={r['DA_price_1']:.4f}", flush=True)
    T = pd.DataFrame(out)
    T.to_csv(OUT / "filter_horizon_sweep.csv", index=False)
    _summary(T)
    _figure(T)
    return T


def _summary(T):
    print("\n" + "=" * 72)
    print("DOES THE EFFECT GENERALISE?")
    print("=" * 72)
    for h in HORIZONS:
        s = T[(T.h == h) & (T.model == "LR")]
        c, k = s[s.causal], s[~s.causal]
        print(f"  h={h:<3} causal filters: trend {c.DA_trend.mean():.3f} "
              f"vs price_1 {c.DA_price_1.mean():.3f} "
              f"(gap {c.gap.mean():.3f})")
        print(f"        leaky  filters: trend {k.DA_trend.mean():.3f} "
              f"vs price_1 {k.DA_price_1.mean():.3f} "
              f"(gap {k.gap.mean():.3f})")

    print("\n  price accuracy above 0.5 (the thing that must NOT happen):")
    for causal in (True, False):
        s = T[T.causal == causal]
        n1 = int((s.DA_price_1 > 0.5).sum())
        nh = int((s.DA_price_h > 0.5).sum())
        print(f"    {'causal' if causal else 'leaky '} filters: "
              f"next-day {n1}/{len(s)} cells, h-day {nh}/{len(s)} cells")

    print("\n  does SLOPE match the fitted model on the trend target?")
    p = T.pivot_table(index=["filter", "h"], columns="model", values="DA_trend")
    if "LR" in p:
        d = p["SLOPE"] - p["LR"]
        print(f"    SLOPE minus LR: mean {d.mean():+.4f}, "
              f"SLOPE wins {int((d > 0).sum())}/{len(d)} cells")


def _figure(T):
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(5.0 * len(HORIZONS), 4.6),
                             sharey=True)
    order = [f for f in FILTERS]
    for ax, h in zip(np.atleast_1d(axes), HORIZONS):
        s = T[(T.h == h) & (T.model == "LR")].set_index("filter").loc[order]
        x = np.arange(len(order))
        col = ["seagreen" if c else "firebrick" for c in s.causal]
        ax.bar(x - 0.21, s.DA_trend, 0.42, color=col, label="vs filtered trend")
        ax.bar(x + 0.21, s.DA_price_1, 0.42, color=col, alpha=0.38,
               label="vs next-day return")
        ax.axhline(0.5, color="k", ls="--", lw=1.1, label="coin flip (50%)")
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=55, ha="right", fontsize=8)
        ax.set_title(f"$h={h}$")
        ax.set_ylim(0.35, 1.0)
    np.atleast_1d(axes)[0].set_ylabel("direction accuracy")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("The gap between accuracy on a filtered target and accuracy on "
                 "price,\nacross nine filters and three horizons. Green filters "
                 "are causal, red read the future.",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / "filter_horizon_sweep.png")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(0 if main("--gru" in sys.argv) is not None else 1)
