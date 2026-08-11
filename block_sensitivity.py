"""Stationary-bootstrap block-length sensitivity.

Every inferential statement in the paper uses a mean block of 20 days. This
module repeats the key ones at 10, 20 and 40 and records whether any
qualitative conclusion moves. Nothing else changes: same data, same arms, same
estimators, same B.

Covered:
  1. the per-fold accuracy table aggregates;
  2. the per-model directional intervals of the significance table;
  3. the superior predictive ability tests, for the two contrasts the paper's
     conclusions rest on;
  4. MATCH-B dSharpe intervals and p-values;
  5. the LONG-book residual Sharpe intervals;
  6. the sub-chance slope-sign diagnostic.

Outputs -> results/analysis/block_sensitivity.csv
"""
import numpy as np
import pandas as pd

from portfolio_replay import (ROOT, SIGNS, book_scale, load_components,
                             paired_tests, portfolio, price_positions,
                             raw_positions, sharpe, stationary_bootstrap_idx)
from excess_accuracy import (attach_predictions, build_cells, subchance,
                                table2_ci)
from return_mechanism import long_residual
from predictive_ability import (B as SPA_B, MODELS, _boot_means, load_predictions,
                          panel, spa)

OUT = ROOT / "results" / "analysis"
BLOCKS = (10, 20, 40)
B = 10000
TOP4 = ["GRU", "LR", "TimeFilter", "TimeMixer"]
ROWS = []


def add(analysis, quantity, block, value, lo=None, hi=None, p=None):
    ROWS.append(dict(analysis=analysis, quantity=quantity, block=block,
                     value=value, lo=lo, hi=hi, p=p))


# ---------------------------------------------------------------- 1 + 6
def forecast_side(cells):
    for blk in BLOCKS:
        T2, _ = table2_ci(B=B, block=blk)
        for _, r in T2.iterrows():
            add("per-fold accuracy table", r.truth, blk, r.point, r.lo, r.hi,
                r.p_vs_half)
        S = subchance(cells, B=B, block=blk)
        add("sub-chance slope sign", "next-day DA", blk, S["da"],
            S["da_lo"], S["da_hi"], S["p_da_below_half"])
        add("sub-chance slope sign", "signed return (bps/day)", blk,
            S["signed_bps"], S["signed_lo"], S["signed_hi"])
        add("sub-chance slope sign", "beta on next-day return", blk,
            S["beta"], S["beta_lo"], S["beta_hi"])


# ---------------------------------------------------------------- 2
def directional_intervals(P):
    """Pooled direction accuracy against the trend, per carried model."""
    for m in TOP4:
        d = P[P.model == m]
        g = pd.DataFrame({"h": (np.sign(d.y_pred) == np.sign(d.y_true)).values,
                          "date": d.date.values}).groupby("date").h.mean()
        x = g.values
        for blk in BLOCKS:
            I = stationary_bootstrap_idx(len(x), B, blk, 17)
            dr = x[I].mean(axis=1)
            add("directional interval", m, blk, float(x.mean()),
                float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5)),
                float((1 + np.sum(dr <= 0.5)) / (B + 1)))


# ---------------------------------------------------------------- 3
def spa_blocks(P):
    models = sorted(P.model.unique())
    P = P.copy()
    P["hit_trend"] = (np.sign(P.y_pred) == np.sign(P.y_true)).astype(float)
    P["hit_p1"] = (np.sign(P.y_pred) == np.sign(P.y_true_price_1)).astype(float)
    sl = pd.read_csv(ROOT / "results" / "analysis" / "slope_daily_accuracy.csv",
                     parse_dates=["date"]).set_index("date")
    Ht = panel(P, "hit_trend")[models]
    common = Ht.index.intersection(sl.index)
    Xs = Ht.loc[common].values - sl.loc[common, ["hit_trend"]].values
    Xp = panel(P, "hit_p1")[models].values - 0.5

    for blk in BLOCKS:
        for name, X in (("direction on trend vs SLOPE", Xs),
                        ("direction on next-day return vs chance", Xp)):
            bm = _boot_means(X, stationary_bootstrap_idx(X.shape[0], B, blk, 13))
            res, _, _, _, _ = spa(X, bm=bm)
            add("SPA", name, blk, res["t_spa"], p=res["spa_c"])


# ---------------------------------------------------------------- 4
def matchb_blocks(store):
    for model, comps in store.items():
        pm = raw_positions(comps, SIGNS["PRED"], gated=False)
        ks = book_scale(price_positions(comps, pm))
        p = portfolio(comps, SIGNS["PRED"], gated=False, shared_scale=ks)
        f = portfolio(comps, SIGNS["FLIP"], gated=False, shared_scale=ks)
        al = pd.concat({"f": f, "p": p}, axis=1).dropna()
        for blk in BLOCKS:
            t = paired_tests(al["f"].values, al["p"].values, B=B, mean_block=blk)
            add("MATCH-B dSharpe", model, blk, t["d_sharpe"], t["ci_lo"],
                t["ci_hi"], t["p_sharpe"])


# ---------------------------------------------------------------- 5
def residual_blocks(store):
    for blk in BLOCKS:
        R = long_residual(store, block=blk, B=B)
        for _, r in R.iterrows():
            add("LONG residual Sharpe", f"{r.model} {r.arm}", blk,
                r.sharpe_resid, r.resid_lo, r.resid_hi)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = build_cells()
    P = load_predictions()
    store = load_components()

    print("1/6 forecast-side (per-fold table, sub-chance)...", flush=True)
    forecast_side(cells)
    print("2/6 directional intervals...", flush=True)
    directional_intervals(P)
    print("3/6 SPA...", flush=True)
    spa_blocks(P)
    print("4/6 MATCH-B...", flush=True)
    matchb_blocks(store)
    print("5/6 LONG residual...", flush=True)
    residual_blocks(store)

    S = pd.DataFrame(ROWS)
    S.to_csv(OUT / "block_sensitivity.csv", index=False)

    print("\n=== block-length sensitivity ===")
    for a in S.analysis.unique():
        piv = S[S.analysis == a].pivot_table(index="quantity", columns="block",
                                             values=["value", "lo", "hi", "p"])
        print(f"\n--- {a}")
        print(piv.round(4).to_string())

    # does any qualitative conclusion move?
    print("\n=== sign / exclusion-of-zero stability ===")
    flips = []
    for q, g in S[S.lo.notna()].groupby(["analysis", "quantity"]):
        excl = set(np.sign(g.lo) == np.sign(g.hi))
        if len(excl) > 1:
            flips.append((q, g[["block", "value", "lo", "hi"]].round(4)
                          .to_string(index=False)))
    if flips:
        for q, txt in flips:
            print(f"  CHANGES: {q}\n{txt}")
    else:
        print("  no interval changes whether it excludes zero across 10/20/40")
    print(f"\n-> {OUT / 'block_sensitivity.csv'}")


if __name__ == "__main__":
    main()
