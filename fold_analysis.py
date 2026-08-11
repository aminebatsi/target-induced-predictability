"""Per-fold diagnostic: WHY do some test folds underperform?

The asset x test-fold-year heatmap (results/strategy/heatmap_asset_year.png)
shows the strategy's Sharpe is far from uniform across folds -- 2021-2023 are
strong, 2024-2025 are flat to negative. This module asks why, and separates
the two explanations a reviewer will immediately demand:

  A. MARKET-WIDE      the fold was hard for EVERY directional signal (the
                      no-ML baselines LONG and REGIME fail too) -> a regime
                      effect, not model decay.
  B. MODEL-SPECIFIC   only PRED degrades while LONG/REGIME hold up -> the
                      prediction itself stopped working (alpha decay), the
                      far more serious finding.

Everything is computed from the SAME execution stack strategy.py uses
(`strategy._components`), so nothing here can drift from the headline result.

Per-fold diagnostics:

  sharpe_{PRED,LONG,REGIME,FLIP,RANDOM}  per-fold Sharpe of each signal
  sharpe_PRED_gross  PRED Sharpe before fees/funding -- isolates cost drag
  sharpe_PRED_noscale  PRED without the book-level vol target -- isolates how
                     much of each fold is risk control vs directional edge
  DA                 direction accuracy of the prediction vs the realized
                     h-step Kalman-trend change (is the FORECAST still right?)
  trend_efficiency   |sum r| / sum|r| per asset, averaged. 1 = clean trend,
                     ~0 = chop. A trend-following stack needs this high.
  sma_flips          SMA-200 bull<->bear transitions per asset (regime churn)
  ann_vol            realized annualized vol of the underlying assets
  turnover           mean |position change| per day (what drives cost)
  cost_drag          annualized gross-minus-net return (fees + funding)
  breadth            fraction of assets with positive PRED Sharpe in the fold

CAVEAT stated up front: there are only 5 folds. Cross-fold correlations below
are DESCRIPTIVE, not inferential -- with n=5 nothing here is significant, and
it is reported as a diagnostic narrative, never as a tested claim.

Outputs -> results/fold_analysis/
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (COST, FOLDS, ORIGINAL_ASSETS, PLOT_RC, RESULTS,
                    SIGNAL_BLEND, VOL_TARGET)
from evaluation import max_drawdown, net_daily, sharpe
from strategy import (N_PERM, _aggregate, _best_model, _components, _gate,
                      _nz_sign, _pos_net, _regime_sign, book_scale)

plt.rcParams.update(PLOT_RC)
OUT = RESULTS / "fold_analysis"
ORDER = ["BLEND", "PRED", "LONG", "REGIME", "FLIP"]
SIGN_FNS = {
    "PRED": lambda c, rng: _nz_sign(c["w"]),
    "LONG": lambda c, rng: np.ones(len(c["w"])),
    "REGIME": lambda c, rng: _regime_sign(c),
    "FLIP": lambda c, rng: -_nz_sign(c["w"]),
}
# BLEND runs the PRED signs through the stack and mixes the resulting exposure
# with the regime trade -- so it is a position rule, not another sign function.
BLEND_OF = "PRED"


def _positions(c, s, scale=None, blend=False):
    """The execution stack's final position -- identical to strategy._position
    (+ strategy._blended), but exposed here so gross/net and turnover split."""
    def stack(sig):
        g = _gate(sig, c["bull"])
        g_ex = np.where((g != 0) & (c["c_te"] >= c["tau"]), 0.0, g)
        p = g_ex * c["vsc"]
        return p if scale is None else p * scale

    pos = stack(s)
    if blend:
        pos = SIGNAL_BLEND * pos + (1.0 - SIGNAL_BLEND) * stack(_regime_sign(c))
    return pos


def _scale_for(comps, signs, blend=False):
    """The book-level vol-target scale for one directional signal.

    Computed from the FULL cross-fold book (the test folds are contiguous
    calendar time, so a live book would carry this history across the July
    boundaries) and then sliced per fold -- so every per-fold number below is
    the same size the headline strategy actually traded.
    """
    raw = {}
    for (k, a), c in comps.items():
        raw.setdefault(k, {})[a] = _pos_net(c, signs[(k, a)], blend=blend)
    return book_scale(_aggregate(raw))


def _fold_portfolio(comps, k, sign_fn, rng=None, cost=COST, with_fund=True,
                    scale=None, blend=False):
    """Equal-weight portfolio of one fold's assets under one directional signal."""
    dd = {}
    for (kk, a), c in comps.items():
        if kk != k:
            continue
        sc = None if scale is None else scale.reindex(c["dates"]).ffill().fillna(1.0).values
        pos = _positions(c, sign_fn(c, rng), sc, blend=blend)
        fund = c["fund"] if with_fund else 0.0
        dd[a] = pd.Series(net_daily(pos, c["r"], cost, fund), index=c["dates"])
    return pd.concat(dd, axis=1, sort=True).mean(axis=1, skipna=True).dropna()


def _pinned(comps, fn, rng=None):
    """Freeze one draw of a (possibly stochastic) sign function, and return a
    sign_fn replaying it. The book scale and the priced book must see the SAME
    draw, and _fold_portfolio is called once per fold."""
    by_id = {id(c): fn(c, rng) for c in comps.values()}
    return lambda c, r: by_id[id(c)], {key: by_id[id(c)] for key, c in comps.items()}


def _signal_sharpes(comps, folds):
    """Per-fold Sharpe for each named signal, plus the RANDOM permutation null.
    Every leg is priced at the book-level vol-targeted size, so these match the
    numbers strategy.py reports."""
    rows = {k: {} for k in folds}
    for m, fn in SIGN_FNS.items():
        pin, signs = _pinned(comps, fn)
        sc = _scale_for(comps, signs)
        for k in folds:
            rows[k][m] = sharpe(_fold_portfolio(comps, k, pin, scale=sc).values)

    # the shipped blend: PRED signs, exposure mixed with the regime trade
    pin, signs = _pinned(comps, SIGN_FNS[BLEND_OF])
    sc_b = _scale_for(comps, signs, blend=True)
    for k in folds:
        rows[k]["BLEND"] = sharpe(
            _fold_portfolio(comps, k, pin, scale=sc_b, blend=True).values)

    pin, signs = _pinned(comps, SIGN_FNS["PRED"])
    sc_pred = _scale_for(comps, signs)
    for k in folds:
        # gross: no fees, no funding -- isolates cost drag from lost edge
        rows[k]["PRED_gross"] = sharpe(_fold_portfolio(
            comps, k, pin, cost=0.0, with_fund=False, scale=sc_pred).values)
        # the same book WITHOUT the vol target, to size the overlay's effect
        rows[k]["PRED_noscale"] = sharpe(_fold_portfolio(comps, k, pin).values)

    draws = {k: [] for k in folds}
    for seed in range(N_PERM):
        rng = np.random.default_rng(seed)
        pin, signs = _pinned(
            comps, lambda c, r: r.choice([-1.0, 1.0], size=len(c["w"])), rng)
        sc = _scale_for(comps, signs)
        for k in folds:
            draws[k].append(sharpe(_fold_portfolio(comps, k, pin, scale=sc).values))
    for k in folds:
        rows[k]["RANDOM"] = float(np.mean(draws[k]))
    return pd.DataFrame(rows).T


def _market_diagnostics(comps, folds):
    """Per-fold market character + execution diagnostics, averaged over assets."""
    _, signs = _pinned(comps, SIGN_FNS["PRED"])
    sc_pred = _scale_for(comps, signs)
    rows = {}
    for k in folds:
        cs = [(a, c) for (kk, a), c in comps.items() if kk == k]
        das, da_h, da_1 = [], [], []
        teff, flips, vols, turns, drags, per_asset_sh = [], [], [], [], [], []
        for a, c in cs:
            sc = sc_pred.reindex(c["dates"]).ffill().fillna(1.0).values
            pos = _positions(c, SIGN_FNS["PRED"](c, None), sc)
            r = c["r"]
            # three direction accuracies, against three different truths:
            #   DA_trend  the SMOOTHED Kalman h-step trend change (what forecast.py reports)
            #   DA_price_h  the RAW h-day price change (is the trend call tradeable?)
            #   DA_price_1  the RAW next-day return (the direction P&L is actually earned on)
            das.append(float(np.mean(np.sign(c["w"]) == np.sign(c["d_te"]))))
            da_h.append(float(np.mean(np.sign(c["w"]) == np.sign(c["fwd_h"]))))
            da_1.append(float(np.mean(np.sign(c["w"]) == np.sign(r))))
            teff.append(float(abs(np.sum(r)) / (np.sum(np.abs(r)) + 1e-18)))
            flips.append(int(np.sum(np.diff(c["bull"].astype(int)) != 0)))
            vols.append(float(np.std(r) * np.sqrt(365)))
            turns.append(float(np.mean(np.abs(np.diff(pos, prepend=0.0)))))
            gross = pos * r
            net = net_daily(pos, r, COST, c["fund"])
            drags.append(float(np.mean(gross - net) * 365))
            per_asset_sh.append(sharpe(net))
        rows[k] = dict(DA_trend=np.mean(das), DA_price_h=np.mean(da_h),
                       DA_price_1=np.mean(da_1), trend_efficiency=np.mean(teff),
                       sma_flips=np.mean(flips), ann_vol=np.mean(vols),
                       turnover=np.mean(turns), cost_drag=np.mean(drags),
                       breadth=float(np.mean(np.array(per_asset_sh) > 0)),
                       n_assets=len(cs))
    return pd.DataFrame(rows).T


def _verdict(S, D):
    """Descriptive read of each weak fold: market-wide vs model-specific."""
    lines = []
    baseline = S[["LONG", "REGIME"]].mean(axis=1)
    for k in S.index:
        if S.loc[k, "PRED"] >= 0.5:
            continue
        pred, base = S.loc[k, "PRED"], baseline[k]
        if base < 0.3 and pred >= base - 0.25:
            tag = "MARKET-WIDE -- the no-ML baselines fail here too"
        elif base >= 0.3 and pred < base - 0.25:
            tag = "MODEL-SPECIFIC -- baselines held up, the prediction did not"
        else:
            tag = "MIXED -- weak baselines and no PRED edge over them"
        lines.append(f"  fold {int(S.loc[k, 'year'])}: PRED {pred:+.2f} vs "
                     f"LONG/REGIME mean {base:+.2f}  ->  {tag}\n"
                     f"      DA trend {D.loc[k, 'DA_trend']:.3f} / price-h "
                     f"{D.loc[k, 'DA_price_h']:.3f} / price-1d {D.loc[k, 'DA_price_1']:.3f} "
                     f"| trend-eff {D.loc[k, 'trend_efficiency']:.3f} "
                     f"| SMA flips {D.loc[k, 'sma_flips']:.1f} | cost drag "
                     f"{D.loc[k, 'cost_drag']:.3f}/yr | breadth {D.loc[k, 'breadth']:.0%}")
    return lines


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    model = _best_model()
    print(f"== FOLD ANALYSIS (why do some folds underperform? model = {model}) ==")
    comps = {}
    for name in ORIGINAL_ASSETS:
        for c in _components(name, model):
            comps[(c["fold"], name)] = c
        print(f"  {name} done", flush=True)

    folds = sorted({k for (k, _) in comps})
    S = _signal_sharpes(comps, folds)
    D = _market_diagnostics(comps, folds)
    years = pd.Series({k: pd.Timestamp(FOLDS[k][0]).year for k in folds})
    S.insert(0, "year", years)
    D.insert(0, "year", years)
    S.to_csv(OUT / "fold_signal_sharpe.csv")
    D.to_csv(OUT / "fold_diagnostics.csv")

    print("\nPER-FOLD SHARPE BY SIGNAL (execution stack held fixed):")
    print(S.round(3).to_string(index=False))
    print("\nPER-FOLD MARKET / EXECUTION DIAGNOSTICS:")
    print(D.round(3).to_string(index=False))

    corr = pd.Series({c: float(np.corrcoef(D[c].values.astype(float),
                                           S["PRED"].values.astype(float))[0, 1])
                      for c in ["DA_trend", "DA_price_h", "DA_price_1",
                                "trend_efficiency", "sma_flips", "ann_vol",
                                "turnover", "cost_drag", "breadth"]}).sort_values()
    corr.to_csv(OUT / "diagnostic_correlation.csv", header=["corr_with_PRED_sharpe"])
    print(f"\nCORRELATION WITH PRED FOLD SHARPE  [n={len(folds)} folds -- DESCRIPTIVE ONLY,"
          f"\nnothing here is statistically significant at this sample size]:")
    print(corr.round(3).to_string())

    print("\nWEAK-FOLD READ:")
    lines = _verdict(S, D)
    print("\n".join(lines) if lines else "  (no fold with PRED Sharpe < 0.5)")

    # ---------------- figure 1: per-fold Sharpe by signal ----------------
    colors = {"BLEND": "navy", "PRED": "steelblue", "LONG": "darkorange",
              "REGIME": "seagreen", "FLIP": "crimson", "RANDOM": "gray"}
    fig, ax = plt.subplots(figsize=(12, 5.5))
    xs = np.arange(len(folds))
    bars = ORDER + ["RANDOM"]
    w = 0.9 / len(bars)
    for i, m in enumerate(bars):
        ax.bar(xs + (i - (len(bars) - 1) / 2) * w, S[m].values, w, label=m,
               color=colors[m], alpha=0.9)
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(xs, [f"{int(y)}--{int(y) + 1}" for y in S["year"]])
    ax.set_xlabel("walk-forward test fold")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Sharpe ratio by directional signal and test fold, "
                 "execution stack held fixed")
    ax.legend(ncol=len(bars))
    fig.tight_layout()
    fig.savefig(OUT / "fold_signal_sharpe.png")
    plt.close(fig)

    # ---------------- figure 2: diagnostics vs fold ----------------
    panels = [("(a) Sharpe ratio, PRED", S["PRED"].values, "navy", None),
              ("(b) Direction accuracy vs smoothed trend", D["DA_trend"].values, "purple", 0.5),
              ("(c) Direction accuracy vs raw $h$-day price", D["DA_price_h"].values, "indigo", 0.5),
              ("(d) SMA-200 regime transitions per asset", D["sma_flips"].values, "darkorange", None),
              ("(e) Cost drag (annualised return)", D["cost_drag"].values, "crimson", None),
              ("(f) Breadth (fraction of assets with Sharpe > 0)", D["breadth"].values, "teal", None)]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0))
    labels = [f"{int(y)}" for y in S["year"]]
    for ax, (title, vals, col, ref) in zip(axes.ravel(), panels):
        ax.bar(range(len(vals)), vals, color=col, alpha=0.85)
        ax.set_xticks(range(len(vals)), labels, fontsize=8)
        ax.set_xlabel("test fold")
        ax.set_title(title, fontsize=9.5)
        ax.axhline(0, color="k", lw=0.8)
        if ref is not None:
            ax.axhline(ref, color="k", ls="--", lw=1)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7.5)
    fig.suptitle("Per-fold market and execution diagnostics "
                 f"($n={len(folds)}$ folds; descriptive, not inferential)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "fold_diagnostics.png")
    plt.close(fig)

    # ---------------- figure 3: gross vs net vs un-throttled (decomposition) ----------------
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.bar(xs - 0.27, S["PRED_gross"].values, 0.27,
           label="gross of fees and funding", color="#9bb8d6")
    ax.bar(xs, S["PRED"].values, 0.27,
           label=f"net, {COST * 1e4:.0f} bps per side", color="navy")
    ax.bar(xs + 0.27, S["PRED_noscale"].values, 0.27,
           label=f"net, without the {VOL_TARGET:.0%} book volatility target",
           color="#c96f6f")
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(xs, [f"{int(y)}--{int(y) + 1}" for y in S["year"]])
    ax.set_xlabel("walk-forward test fold")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Decomposition of fold performance into transaction cost, "
                 "directional edge and risk control")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fold_gross_vs_net.png")
    plt.close(fig)

    return S, D


if __name__ == "__main__":
    run()
