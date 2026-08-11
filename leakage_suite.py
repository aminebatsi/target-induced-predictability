"""Causality audit applied to every filter in filters.py, not just the Kalman.

leakage.py answers the question for the one target the paper uses. This module
asks it of every smoother in the registry, because the paper's recommendation
is that any target transform should be audited, and a recommendation is worth
more when it is shown catching something.

The same two tests are used for all of them:

  A  influence kernel   d level[t0] / d p[t0+k] for k > 0, which a causal
                        filter must leave at exactly zero.
  B  future corruption  randomise every price after t0, recompute, and require
                        the series up to t0 to be unchanged bit for bit.

A filter passes only if both measure exactly 0.0. No tolerance is used, on
purpose: a tolerance is what lets a small leak through.

Output -> results/leakage/filter_audit.csv, filter_audit.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOT_RC, RESULTS
from data import load_asset
from filters import FILTERS, apply_filter

OUT = RESULTS / "leakage"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(PLOT_RC)
WINDOW = 30


def influence_kernel(name, lp, t0, ks, eps=1e-2):
    """Numerical d level[t0] / d p[t0+k], one bump at a time."""
    base = apply_filter(name, lp, norm_end=t0 + 1)[t0]
    out = np.empty(len(ks))
    for i, k in enumerate(ks):
        if not 0 <= t0 + k < len(lp):
            out[i] = np.nan
            continue
        p2 = lp.copy()
        p2[t0 + k] += eps
        out[i] = (apply_filter(name, p2, norm_end=t0 + 1)[t0] - base) / eps
    return out


def corrupt_future(name, lp, t0, seed=0):
    """Series with the post-t0 prices randomised, filtered the same way."""
    rng = np.random.default_rng(seed)
    p2 = lp.copy()
    p2[t0 + 1:] += rng.normal(0, 0.05, len(lp) - t0 - 1)
    return (apply_filter(name, lp, norm_end=t0 + 1),
            apply_filter(name, p2, norm_end=t0 + 1))


def audit_all(asset="LTC", verbose=True):
    lp = load_asset(asset)["logprice"].values.astype(float)
    t0 = len(lp) - 300
    ks = np.arange(-WINDOW, WINDOW + 1)
    fut = ks > 0

    rows, kernels = [], {}
    for name, meta in FILTERS.items():
        kern = influence_kernel(name, lp, t0, ks)
        kernels[name] = kern
        future_gain = float(np.nanmax(np.abs(kern[fut])))
        a, b = corrupt_future(name, lp, t0)
        pre_diff = float(np.max(np.abs(a[:t0 + 1] - b[:t0 + 1])))
        passed = (future_gain == 0.0) and (pre_diff == 0.0)
        rows.append(dict(filter=name, label=meta["label"],
                         expected_causal=meta["causal"],
                         max_future_influence=future_gain,
                         max_pre_cut_diff=pre_diff,
                         PASSED=passed,
                         matches_expectation=(passed == meta["causal"])))
        if verbose:
            print(f"  {name:12} future={future_gain:9.2e} "
                  f"pre_diff={pre_diff:9.2e}  "
                  f"{'PASS' if passed else 'FAIL'}"
                  f"{'' if passed == meta['causal'] else '   <-- UNEXPECTED'}")

    T = pd.DataFrame(rows)
    T.to_csv(OUT / "filter_audit.csv", index=False)
    json.dump({"asset": asset, "t0": int(t0),
               "n_pass": int(T.PASSED.sum()), "n_total": len(T),
               "all_as_expected": bool(T.matches_expectation.all())},
              open(OUT / "filter_audit.json", "w"), indent=2)
    _figure(T, kernels, ks, fut, lp, t0)
    return T


def _figure(T, kernels, ks, fut, lp, t0):
    """One panel per filter: the influence kernel with the future shaded."""
    n = len(FILTERS)
    ncol, nrow = 3, int(np.ceil(n / 3))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.0 * nrow))
    for ax, (name, meta) in zip(axes.ravel(), FILTERS.items()):
        kern = kernels[name]
        row = T[T["filter"] == name].iloc[0]
        ok = bool(row.PASSED)
        ax.axvspan(0.5, ks[-1] + 0.5, color="orange", alpha=0.13)
        ax.plot(ks[~fut], kern[~fut], color="seagreen", lw=1.3)
        ax.plot(ks[fut], kern[fut], color="crimson", lw=1.6)
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(0, color="navy", ls=":", lw=1.0)
        ax.set_title(f"{meta['label']}\n"
                     rf"$\max_{{k>0}}|\partial \ell_{{t_0}}/\partial p| = $"
                     f"{row.max_future_influence:.1e}  "
                     f"{'PASS' if ok else 'LEAKS'}",
                     fontsize=8.5, color="seagreen" if ok else "firebrick")
        ax.tick_params(labelsize=7)
        ax.set_xlabel(r"lag $k$", fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Causality audit of nine trend-extraction filters. The shaded "
                 "band is the future.\nA causal transform is exactly flat "
                 "there; four of the nine are not.",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "filter_audit.png")
    plt.close(fig)


def main():
    print("Causality audit over all filters (exact zeros required):")
    T = audit_all()
    n_ok = int(T.PASSED.sum())
    print(f"\n  {n_ok}/{len(T)} filters are causal; "
          f"{len(T) - n_ok} read the future.")
    print(f"  every filter matched its expected label: "
          f"{bool(T.matches_expectation.all())}")
    print(f"  -> {OUT / 'filter_audit.csv'}")
    return bool(T.matches_expectation.all())


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
