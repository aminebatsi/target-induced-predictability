"""Figure of the experimental design: observation-index-purged walk-forward splits.

A reviewer's first question about any financial ML result is how the data were
split. This renders it rather than describing it: for each fold, the train,
validation, embargo and test spans exactly as `evaluation.fold_indices` computes
them, so the figure cannot drift from the code.

Panels:
  (a) the five anchored walk-forward folds on a common timeline, with an inset
      that zooms one validation-to-test boundary.  The coloured interval marks
      the calendar span induced by the exact h-observation purge;
  (b) usable history per asset, split by which universe the asset belongs to.

Outputs -> results/design/
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (ASSETS, FOLDS, H, L, MIN_TEST, MIN_TRAIN,
                    ORIGINAL_ASSETS, PLOT_RC, RESULTS, SMA_REGIME)
from data import load_asset
from evaluation import fold_indices
from forecast import _norm_end
from targets import build_windows, kalman_causal

plt.rcParams.update(PLOT_RC)
OUT = RESULTS / "design"

# colourblind-safe, and ordered the way the pipeline consumes the data
C_TRAIN, C_VAL, C_PURGE, C_TEST = "#3B6EA5", "#E8963C", "#C0392B", "#4F9D69"
C_EVAL, C_PORT = "#3B6EA5", "#9AAFC0"


def _fold_spans(name="LTC"):
    """Actual index spans per fold for one asset, taken from the split code."""
    df = load_asset(name)
    dates, lp = df["date"].values, df["logprice"].values.astype(float)
    rows = []
    for k, (ts, tz) in enumerate(FOLDS):
        cut = np.searchsorted(dates, np.datetime64(tz) + np.timedelta64(2, "D"))
        d_, p_ = dates[:cut], lp[:cut]
        if len(p_) < 700:
            continue
        norm_end = _norm_end(d_, ts, tz)
        if norm_end is None:
            continue
        trend, _ = kalman_causal(p_, norm_end)
        _, _, anchors = build_windows(p_, trend, min_anchor=L - 1)
        idx = fold_indices(d_[anchors], ts, tz)
        if idx is None:
            continue
        tr, va, te = idx
        te = te[anchors[te] >= SMA_REGIME]
        ad = pd.DatetimeIndex(d_[anchors])
        rows.append(dict(fold=k, year=pd.Timestamp(ts).year,
                         train_start=ad[tr[0]], train_end=ad[tr[-1]],
                         val_start=ad[va[0]], val_end=ad[va[-1]],
                         test_start=ad[te[0]], test_end=ad[te[-1]],
                         n_train=len(tr), n_val=len(va), n_test=len(te)))
    return pd.DataFrame(rows)


def _draw_folds(ax, S, bar_h=0.58, labels=False):
    """The four spans of every fold as one stacked horizontal bar per fold."""
    for _, r in S.iterrows():
        y = r["fold"]
        for start, end, colour in (
                (r.train_start, r.train_end, C_TRAIN),
                (r.val_start, r.val_end, C_VAL),
                (r.val_end, r.test_start, C_PURGE),    # h-observation purge span
                (r.test_start, r.test_end, C_TEST)):
            ax.barh(y, (end - start).days, left=start, height=bar_h,
                    color=colour, linewidth=0)
        if labels:
            ax.text(r.test_end + pd.Timedelta(days=55), y,
                    f"{r.n_train} / {r.n_val} / {r.n_test}",
                    va="center", ha="left", fontsize=8.5, color="0.25")


def run(asset="LTC"):
    OUT.mkdir(parents=True, exist_ok=True)
    S = _fold_spans(asset)
    S.to_csv(OUT / "fold_spans.csv", index=False)

    fig = plt.figure(figsize=(11.5, 8.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.26, 1.05],
                          width_ratios=[1.0, 1.0], hspace=0.60, wspace=0.18)
    a1 = fig.add_subplot(gs[0, :])
    axz = fig.add_subplot(gs[1, 0])
    a2 = fig.add_subplot(gs[2, :])

    # ---------------- (a) walk-forward folds ----------------
    _draw_folds(a1, S, labels=True)
    a1.set_yticks(S["fold"],
                  [f"fold {int(k)}\n{int(y)} to {int(y) + 1}"
                   for k, y in zip(S["fold"], S["year"])])
    a1.invert_yaxis()
    a1.set_xlabel("date")
    a1.xaxis.set_major_locator(mdates.YearLocator(2))
    a1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    a1.set_title("(a) Anchored walk-forward design on "
                 f"{asset}: five non-overlapping annual test folds", loc="left")
    a1.grid(axis="y", visible=False)
    # room on the right for the size labels, and below for the legend
    a1.set_xlim(S.train_start.min() - pd.Timedelta(days=120),
                S.test_end.max() + pd.Timedelta(days=900))
    a1.margins(y=0.16)

    handles = [mpatches.Patch(color=C_TRAIN, label="train"),
               mpatches.Patch(color=C_VAL, label="validation"),
               mpatches.Patch(color=C_PURGE,
                              label=f"purge interval ($h={H}$ observations)"),
               mpatches.Patch(color=C_TEST, label="test")]
    a1.legend(handles=handles, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.20), frameon=False)
    a1.text(0.995, 1.02, "labels give $n_{train}$ / $n_{val}$ / $n_{test}$",
            transform=a1.transAxes, ha="right", va="bottom",
            fontsize=8.5, color="0.35")

    # ---- zoom strip: show the calendar span induced by the index purge ----
    r = S.iloc[-1]
    _draw_folds(axz, S.iloc[[-1]], bar_h=0.55)
    axz.set_xlim(r.val_end - pd.Timedelta(days=34),
                 r.test_start + pd.Timedelta(days=34))
    axz.set_ylim(r["fold"] + 0.55, r["fold"] - 0.75)
    axz.set_yticks([])
    axz.grid(False)
    axz.tick_params(labelsize=8)
    axz.xaxis.set_major_locator(mdates.DayLocator(interval=10))
    axz.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    y0 = r["fold"] - 0.44
    axz.annotate("", xy=(r.val_end, y0), xytext=(r.test_start, y0),
                 arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.0))
    axz.text(r.val_end + (r.test_start - r.val_end) / 2, y0 - 0.06,
             f"$h={H}$-observation purge", ha="center", va="bottom",
             fontsize=8.5, color="0.2")
    axz.set_title(f"Zoom on the fold {int(r['fold'])} boundary: no validation "
                  f"label overlaps the test window", fontsize=9, loc="left")

    # ---------------- (b) usable history per asset ----------------
    names = list(dict.fromkeys(list(ASSETS) + list(ORIGINAL_ASSETS)))
    spans = []
    for n in names:
        d = load_asset(n)
        spans.append((n, d["date"].min(), d["date"].max(), n in ASSETS))
    spans.sort(key=lambda t: (not t[3], t[1]))        # eval universe first

    for i, (n, s, e, is_eval) in enumerate(spans):
        a2.barh(i, (e - s).days, left=s, height=0.62,
                color=C_EVAL if is_eval else C_PORT, linewidth=0)
    a2.set_yticks(range(len(spans)), [n for n, _, _, _ in spans], fontsize=8.5)
    a2.invert_yaxis()
    a2.grid(axis="y", visible=False)

    lo, hi = pd.Timestamp(FOLDS[0][0]), pd.Timestamp(FOLDS[-1][1])
    a2.axvspan(lo, hi, color=C_TEST, alpha=0.13, zorder=0)
    for ts, _ in FOLDS:
        a2.axvline(pd.Timestamp(ts), color="0.45", ls=":", lw=0.8, zorder=1)
    a2.set_xlabel("date")
    a2.xaxis.set_major_locator(mdates.YearLocator(2))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    a2.set_title("(b) Available history by asset; the shaded band is the "
                 "walk-forward evaluation span", loc="left")
    a2.legend(handles=[
        mpatches.Patch(color=C_EVAL, label="forecast-evaluation universe"),
        mpatches.Patch(color=C_PORT, label="portfolio universe"),
        mpatches.Patch(color=C_TEST, alpha=0.13, label="evaluation span")],
        ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)

    fig.savefig(OUT / "experimental_design.png")
    plt.close(fig)

    print("== DESIGN FIGURE ==")
    print(S[["year", "n_train", "n_val", "n_test"]].to_string(index=False))
    print(f"  window L={L}, horizon h={H}, observation-index purge=h, "
          f"min train/test={MIN_TRAIN}/{MIN_TEST}")
    print(f"-> {OUT}")
    return S


if __name__ == "__main__":
    run()
