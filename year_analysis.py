"""Per-fold return attribution: what made a year good or bad?

`fold_analysis.py` reports per-fold Sharpe by signal. This module answers the
adjacent and more interpretable question a reader actually asks of a backtest:
what was the percentage return each year, and why was it that number?

Sharpe hides the two things that produce it. A fold can be weak because the
strategy earned nothing (a NUMERATOR problem) or because it earned a normal
amount while carrying far more risk (a DENOMINATOR problem), and the remedies
are opposite. So every fold is decomposed into:

  return          net compound return over the fold, in per cent
  volatility      annualised realised volatility of the book
  Sharpe          the ratio of the two
  long / short    the return contributed by each leg
  cost drag       fees and funding, as a return
  market context  what the assets themselves did (drift, dispersion, average
                  pairwise correlation, trend efficiency)
  signal quality  direction accuracy against real next-day returns, and the
                  share of assets that were profitable (breadth)

The market-context columns are properties of the DATA, not of the strategy, so
a weak fold can be attributed to a hostile market rather than to model decay --
or not, which is the more serious finding.

Outputs -> results/year_analysis/
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import COST, FOLDS, ORIGINAL_ASSETS, PLOT_RC, RESULTS
from evaluation import max_drawdown, net_daily, sharpe
from strategy import (_aggregate, _best_model, _blended, _components,
                      _nz_sign, _portfolio, _regime_sign, flagship_book_scale)

plt.rcParams.update(PLOT_RC)
OUT = RESULTS / "year_analysis"
YEARS = [pd.Timestamp(ts).year for ts, _ in FOLDS]


def _pct(x):
    """Compound return of a log-return series, in per cent."""
    return float(np.exp(np.sum(x)) - 1.0) * 100.0


def run(model=None):
    OUT.mkdir(parents=True, exist_ok=True)
    model = model or _best_model()
    print(f"== YEAR ANALYSIS (per-fold return attribution; signal = {model}) ==")

    comps = {}
    for name in ORIGINAL_ASSETS:
        for c in _components(name, model):
            comps[(c["fold"], name)] = c
        print(f"  {name} done", flush=True)

    k_ser = flagship_book_scale(comps)
    book = _portfolio(comps, lambda c, r: _nz_sign(c["w"]), blend=True)

    rows = []
    for k, (ts, tz) in enumerate(FOLDS):
        cs = [(a, c) for (kk, a), c in comps.items() if kk == k]
        if not cs:
            continue
        m = (book.index >= ts) & (book.index < tz)
        x = book.values[m]
        if len(x) < 10:
            continue

        # ---- strategy side: split the book into its long and short legs ----
        lo, sh, drag, turn = [], [], [], []
        for a, c in cs:
            sc = k_ser.reindex(c["dates"]).ffill().fillna(1.0).values
            pos = _blended(c, _nz_sign(c["w"]), sc, blend=True)
            lo.append(np.sum(np.maximum(pos, 0.0) * c["r"]))
            sh.append(np.sum(np.minimum(pos, 0.0) * c["r"]))
            gross = pos * c["r"]
            drag.append(np.sum(gross - net_daily(pos, c["r"], COST, c["fund"])))
            turn.append(np.mean(np.abs(np.diff(pos, prepend=0.0))))

        # ---- market side: properties of the assets, not of the strategy ----
        R = pd.concat({a: pd.Series(c["r"], index=c["dates"]) for a, c in cs},
                      axis=1, sort=True).dropna()
        C = R.corr().values
        avg_corr = float((C.sum() - len(C)) / (len(C) * (len(C) - 1)))
        drift = float(np.mean([np.sum(c["r"]) for _, c in cs]))
        disp = float(np.std([np.sum(c["r"]) for _, c in cs]))
        teff = float(np.mean([abs(np.sum(c["r"])) / (np.sum(np.abs(c["r"])) + 1e-12)
                              for _, c in cs]))
        avol = float(np.mean([np.std(c["r"]) * np.sqrt(365) for _, c in cs]))

        # ---- signal side ----
        hit, per_asset = [], []
        for a, c in cs:
            sc = k_ser.reindex(c["dates"]).ffill().fillna(1.0).values
            pos = _blended(c, _nz_sign(c["w"]), sc, blend=True)
            live = np.abs(pos) > 1e-9
            if live.sum() > 10:
                hit.append(float(np.mean(np.sign(pos[live]) == np.sign(c["r"][live]))))
            per_asset.append(sharpe(net_daily(pos, c["r"], COST, c["fund"])))

        rows.append(dict(
            year=YEARS[k],
            return_pct=_pct(x), ann_vol=float(np.std(x) * np.sqrt(365)),
            sharpe=sharpe(x), max_dd_pct=(np.exp(max_drawdown(x)) - 1) * 100,
            long_leg_pct=(np.exp(np.mean(lo)) - 1) * 100,
            short_leg_pct=(np.exp(np.mean(sh)) - 1) * 100,
            cost_drag_pct=(1 - np.exp(-np.mean(drag))) * 100,
            turnover=float(np.mean(turn)),
            asset_drift_pct=(np.exp(drift) - 1) * 100,
            asset_dispersion=disp, asset_ann_vol=avol,
            avg_pair_corr=avg_corr, trend_efficiency=teff,
            hit_rate=float(np.mean(hit)) if hit else np.nan,
            breadth=float(np.mean(np.array(per_asset) > 0)),
            n_assets=len(cs)))

    Y = pd.DataFrame(rows).set_index("year")
    Y.to_csv(OUT / "year_attribution.csv")

    print("\nPER-YEAR RETURN AND RISK")
    print(Y[["return_pct", "ann_vol", "sharpe", "max_dd_pct"]].round(2).to_string())
    print("\nWHERE THE RETURN CAME FROM (per cent)")
    print(Y[["long_leg_pct", "short_leg_pct", "cost_drag_pct", "turnover"]]
          .round(2).to_string())
    print("\nWHAT THE MARKET WAS DOING (properties of the data, not the strategy)")
    print(Y[["asset_drift_pct", "asset_dispersion", "asset_ann_vol",
             "avg_pair_corr", "trend_efficiency"]].round(3).to_string())
    print("\nSIGNAL QUALITY")
    print(Y[["hit_rate", "breadth"]].round(3).to_string())

    # ---- narrative read, generated from the numbers rather than written by hand ----
    good = Y[Y.return_pct >= Y.return_pct.median()]
    bad = Y[Y.return_pct < Y.return_pct.median()]
    lines = ["# Per-year read", ""]
    for y, r in Y.iterrows():
        drivers = []
        if r.avg_pair_corr > Y.avg_pair_corr.median():
            drivers.append(f"high cross-asset correlation ({r.avg_pair_corr:.2f}), "
                           f"so the {int(r.n_assets)}-asset book had less effective breadth")
        if r.trend_efficiency < Y.trend_efficiency.median():
            drivers.append(f"low trend efficiency ({r.trend_efficiency:.3f}), i.e. choppy "
                           f"paths that a trend-following stack cannot monetise")
        if r.breadth < 0.6:
            drivers.append(f"narrow breadth ({r.breadth:.0%} of assets profitable)")
        if r.cost_drag_pct > Y.cost_drag_pct.median():
            drivers.append(f"above-median cost drag ({r.cost_drag_pct:.1f}%)")
        if r.short_leg_pct < 0:
            drivers.append(f"a loss-making short leg ({r.short_leg_pct:+.1f}%)")
        verdict = "strong" if r.return_pct >= Y.return_pct.median() else "weak"
        lines.append(f"**{y} ({verdict}, {r.return_pct:+.1f}%, Sharpe {r.sharpe:+.2f}).** "
                     f"Long leg {r.long_leg_pct:+.1f}%, short leg {r.short_leg_pct:+.1f}%, "
                     f"costs {r.cost_drag_pct:.1f}%. Assets themselves returned "
                     f"{r.asset_drift_pct:+.1f}% on average. "
                     + ("Adverse conditions: " + "; ".join(drivers) + "."
                        if drivers else "No adverse condition flagged.") + "")
        lines.append("")
    lines += ["", f"Correlation of each diagnostic with the fold return "
                  f"(n={len(Y)} folds -- descriptive only):", ""]
    corr = Y.corr(numeric_only=True)["return_pct"].drop("return_pct").sort_values()
    lines += [f"- `{k}`: {v:+.2f}" for k, v in corr.items()]
    (OUT / "year_read.md").write_text("\n".join(lines), encoding="utf-8")
    corr.to_frame("corr_with_return").to_csv(OUT / "return_drivers.csv")
    print("\nCORRELATION WITH FOLD RETURN (n=5, descriptive only)")
    print(corr.round(2).to_string())

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    a0, a1, a2 = axes
    xs = np.arange(len(Y))
    cols = ["#55A868" if v >= 0 else "#C44E52" for v in Y.return_pct]
    a0.bar(xs, Y.return_pct, color=cols, alpha=0.9)
    for i, v in enumerate(Y.return_pct):
        a0.text(i, v, f"{v:+.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)
    a0.axhline(0, color="k", lw=0.9)
    a0.set_xticks(xs, Y.index)
    a0.set_xlabel("test fold")
    a0.set_ylabel("net return (%)")
    a0.set_title("(a) Net return by test fold")

    w = 0.26
    a1.bar(xs - w, Y.long_leg_pct, w, label="long leg", color="#4C72B0")
    a1.bar(xs, Y.short_leg_pct, w, label="short leg", color="#DD8452")
    a1.bar(xs + w, -Y.cost_drag_pct, w, label="costs and funding", color="#C44E52")
    a1.axhline(0, color="k", lw=0.9)
    a1.set_xticks(xs, Y.index)
    a1.set_xlabel("test fold")
    a1.set_ylabel("contribution to return (%)")
    a1.set_title("(b) Decomposition of the return")
    a1.legend()

    a2.plot(xs, Y.avg_pair_corr, "o-", label="mean pairwise correlation")
    a2.plot(xs, Y.trend_efficiency, "s-", label="trend efficiency")
    a2.plot(xs, Y.breadth, "^-", label="breadth")
    a2.set_xticks(xs, Y.index)
    a2.set_xlabel("test fold")
    a2.set_ylabel("value")
    a2.set_title("(c) Market conditions and breadth")
    a2.legend()
    fig.suptitle(f"Per-fold return attribution, signal model {model}",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "year_attribution.png")
    plt.close(fig)

    print(f"\n-> {OUT}")
    return Y


if __name__ == "__main__":
    run()
