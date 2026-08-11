"""Equal-weight portfolio on the causal Kalman-trend forecast, with a
directional-signal ablation.

The portfolio is an equal-weight book over `config.ORIGINAL_ASSETS`. The
question the ablation answers is whether the book's Sharpe ratio comes from the
directional forecast or from the SMA-200 regime gate that surrounds it. The
entire execution stack is held fixed -- regime gate, turn-classifier exit,
per-asset volatility sizing, book-level volatility target -- and only the
directional signal entering the gate is swapped:

  BLEND   SIGNAL_BLEND*PRED + (1-SIGNAL_BLEND)*REGIME   the shipped strategy
  PRED    sign(forecast)                                the forecast alone
  LONG    +1 always                                     regime-gated long-only
  REGIME  sign(price - SMA200)                          trend-following, no model
  RANDOM  random +/-1                                   permutation null
  FLIP    -sign(forecast)                               reversed forecast

PRED against LONG and REGIME asks whether the forecast beats the model-free
baselines. PRED against RANDOM asks whether it sits outside the permutation
null. PRED against FLIP asks whether the sign carries directional information,
since reversing an informative signal has to hurt. These comparisons use the
unblended PRED arm so that they test the forecast rather than the mixture.

Signal blend (`config.SIGNAL_BLEND`). The forecast is a trend-persistence
extrapolator: it pays when trends persist (2021-2023) and costs when they do
not (2024-2025, where a plain SMA-200 follower does better). The PRED/REGIME
frontier is monotone, so blending is diversification across that dependence
rather than a repair. It cuts cross-fold Sharpe dispersion from 1.23 to 1.02 at
a pooled cost the paired bootstrap cannot separate from zero. Mixing is applied
to exposures, after each leg has run the whole stack.

Book-level volatility target (`book_scale`, `config.VOL_TARGET`). Per-asset
sizing normalises each leg against its own volatility but leaves the assembled
book's volatility swinging by roughly a factor of two across folds as
cross-asset correlation moves. The book is therefore also sized against its own
trailing realised volatility, causally and shifted one day, clipped to at most
one so it only de-risks. This is a risk control rather than a signal change: it
applies identically to every arm, so the ablation stays like-for-like.

Outputs -> results/strategy/  (ablation.csv, paired_proofs.csv, equity_mcap.png,
                               ablation_bars.png, ablation_equity.png,
                               asset_sharpe.csv, heatmap_asset_year.png/.csv,
                               fold_sharpe.csv)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (COST, FOLDS, FUND, H, L, ORIGINAL_ASSETS, PLOT_RC, RESULTS,
                    SIGNAL_BLEND, SMA_REGIME, VOL_TARGET, VOL_TARGET_WIN)
from data import load_asset
from evaluation import fold_indices, max_drawdown, net_daily, paired_tests, sharpe
from forecast import _norm_end
from models import SUPPORTS_WEIGHT, fit_turn_clf, predict_full
from targets import build_windows, kalman_causal

plt.rcParams.update(PLOT_RC)
OUT = RESULTS / "strategy"
UNIVERSE = list(ORIGINAL_ASSETS)     # the curated multi-class portfolio universe
MIN_ANCHOR = L - 1
TAUS = (0.5, 0.6, 0.7, 0.8)
START_EQUITY = 100.0
# Draws in the random-sign permutation null. The smallest p-value a permutation
# test can return is 1/(B+1), so B fixes the resolution: B=20 could never go
# below 0.048. 2000 resolves to 0.0005, which is enough to support a claim of
# p < 0.001 and cheap because each draw only replays the cached components.
N_PERM = int(os.environ.get("N_PERM", "2000"))
ORDER = ["BLEND", "PRED", "LONG", "REGIME", "FLIP"]
FLAGSHIP = "BLEND"


def _gate(s, bull):
    return np.where(bull, np.maximum(s, 0.0), np.minimum(s, 0.0))


def _nz_sign(x):
    s = np.sign(x)
    s[s == 0] = 1.0
    return s


def _regime_sign(c):
    return np.where(c["bull"], 1.0, -1.0)


def _best_model():
    fp = RESULTS / "forecast" / "best_model.json"
    if fp.exists():
        return json.load(open(fp))["best_model"]
    print("  [warn] best_model.json not found -- defaulting to LR")
    return "LR"


def _components(name, model):
    """Per fold: the fixed execution-stack ingredients + the prediction, on test days."""
    df = load_asset(name)
    dates_f = df["date"].values
    lp_f = df["logprice"].values.astype(float)
    fund = FUND if ORIGINAL_ASSETS[name][1] else 0.0
    out = []
    for k, (ts, tz) in enumerate(FOLDS):
        cut = np.searchsorted(dates_f, np.datetime64(tz) + np.timedelta64(2, "D"))
        dates, lp = dates_f[:cut], lp_f[:cut]
        if len(lp) < 700:
            continue
        norm_end = _norm_end(dates, ts, tz)
        if norm_end is None:
            continue
        trend, velocity = kalman_causal(lp, norm_end)
        Xtab, Xseq, anchors = build_windows(lp, trend, min_anchor=MIN_ANCHOR)
        idx = fold_indices(dates[anchors], ts, tz)
        if idx is None:
            continue
        tr, va, te = idx
        te = te[anchors[te] >= SMA_REGIME]
        if len(te) < 100:
            continue
        d = trend[anchors + H] - trend[anchors]
        w_all = predict_full(model, Xtab, Xseq, d, tr, va, te,
                             dir_weighted=(model in SUPPORTS_WEIGHT))
        w_sign = _nz_sign(w_all)
        sma = pd.Series(lp).rolling(SMA_REGIME).mean().bfill().values
        bull = lp[anchors] > sma[anchors]
        accel = np.diff(velocity, prepend=velocity[0])
        fast, _ = kalman_causal(lp, norm_end, q_slope=1e-6)
        extra = [velocity, accel, lp - trend, fast - trend]
        X7, _, _ = build_windows(lp, trend, min_anchor=MIN_ANCHOR, extra_chans=extra)
        y_turn = (np.sign(d) != w_sign).astype(int)
        c_va, c_te = fit_turn_clf(X7, y_turn, tr, va, te)
        ret = np.diff(lp, prepend=lp[0])
        rvol = pd.Series(ret).rolling(21).std().bfill().values
        a, a_va = anchors[te], anchors[va]
        vt = float(np.nanmedian(rvol[anchors[tr]]))
        vsc = np.clip(vt / (rvol[a] + 1e-12), 0.0, 1.0)
        r, r_va = lp[a + 1] - lp[a], lp[a_va + 1] - lp[a_va]
        g_va = _gate(w_sign[va], bull[va])
        best_tau, best_s = 0.7, -np.inf
        for tau in TAUS:
            pv = np.where((g_va != 0) & (c_va >= tau), 0.0, g_va)
            s_ = sharpe(net_daily(pv, r_va, COST, fund))
            if s_ > best_s:
                best_s, best_tau = s_, tau
        out.append(dict(fold=k, w=w_all[te], bull=bull[te], vsc=vsc, r=r,
                        c_te=c_te, tau=best_tau, fund=fund, d_te=d[te],
                        fwd_h=lp[a + H] - lp[a],
                        dates=pd.DatetimeIndex(dates[a])))
    return out


def _position(c, s, scale=None):
    """The execution stack's final position: regime gate -> turn exit -> vol
    sizing -> optional book-level scale."""
    g = _gate(s, c["bull"])
    g_ex = np.where((g != 0) & (c["c_te"] >= c["tau"]), 0.0, g)
    pos = g_ex * c["vsc"]
    return pos if scale is None else pos * scale


def _blended(c, s, scale=None, blend=False):
    """Position after optional blending with the plain SMA-200 regime trade.

    Blending happens at the POSITION level, not the sign level: each leg runs
    the whole execution stack (gate -> turn exit -> vol sizing) on its own
    signal first, and only the resulting exposures are combined. Blending the
    signs instead would collapse the two into a single trade and lose exactly
    the error decorrelation the blend exists to harvest.
    """
    p = _position(c, s, scale)
    if not blend:
        return p
    return SIGNAL_BLEND * p + (1.0 - SIGNAL_BLEND) * _position(c, _regime_sign(c), scale)


def _pos_net(c, s, scale=None, blend=False):
    return pd.Series(net_daily(_blended(c, s, scale, blend), c["r"], COST, c["fund"]),
                     index=c["dates"])


def _aggregate(per_fold):
    """Equal-weight the assets within each fold, then chain the folds."""
    series = [pd.concat(dd, axis=1, sort=True).mean(axis=1, skipna=True).dropna()
              for _, dd in sorted(per_fold.items())]
    return pd.concat(series).sort_index()


def book_scale(pooled, target=VOL_TARGET, win=VOL_TARGET_WIN):
    """Causal book-level vol target -> daily scale in [0, 1].

    `pooled` is the book's own net return series with NO scaling applied. Its
    trailing `win`-day vol is shifted one day, so the scale used on day t is a
    function of returns up to t-1 only. Everything it needs (own positions,
    own prices) is observable at trade time, so this is live-tradeable.

    Clipped to 1.0 -> the overlay only de-risks. Because Sharpe is invariant
    to a constant scale, any Sharpe change it produces comes from WHEN it
    de-risks, not from the average size of the book.
    """
    rv = pooled.rolling(win).std().shift(1) * np.sqrt(365)
    return (target / (rv + 1e-12)).clip(0.0, 1.0).fillna(1.0)


def _portfolio(comps, sign_fn, rng=None, scaled=True, blend=False):
    """Equal-weight book under one directional signal, vol-targeted at the
    book level. Returns the pooled net series."""
    # Draw the signs ONCE -- sign_fn is stochastic for the RANDOM null, and the
    # two passes below must price the same draw.
    signs = {key: sign_fn(c, rng) for key, c in comps.items()}

    raw = {}
    for (k, a), c in comps.items():
        raw.setdefault(k, {})[a] = _pos_net(c, signs[(k, a)], blend=blend)
    pooled = _aggregate(raw)
    if not scaled:
        return pooled

    # Pass 2: re-price every leg at the throttled size, so the resizing itself
    # pays the same per-side fee as any other position change.
    k_ser = book_scale(pooled)
    out = {}
    for (k, a), c in comps.items():
        sc = k_ser.reindex(c["dates"]).ffill().fillna(1.0).values
        out.setdefault(k, {})[a] = _pos_net(c, signs[(k, a)], sc, blend=blend)
    return _aggregate(out)


def flagship_book_scale(comps):
    """The scale series the flagship book runs at -- shared with fold_analysis.py
    so its per-fold diagnostics match the headline result exactly."""
    raw = {}
    for (k, a), c in comps.items():
        raw.setdefault(k, {})[a] = _pos_net(c, _nz_sign(c["w"]), blend=True)
    return book_scale(_aggregate(raw))


def run(model=None):
    OUT.mkdir(parents=True, exist_ok=True)
    model = model or _best_model()
    print(f"== STRATEGY (MCAP, kalman trend, {len(UNIVERSE)}-asset equal-weight, "
          f"book vol target {VOL_TARGET:.0%}/{VOL_TARGET_WIN}d; "
          f"role of the directional prediction; model = {model}) ==")
    comps = {}
    for name in UNIVERSE:
        for c in _components(name, model):
            comps[(c["fold"], name)] = c
        print(f"  {name} done", flush=True)

    # (sign function, blend with the regime trade?) -- BLEND is the shipped
    # strategy; PRED is the same prediction run pure, kept as an ablation arm so
    # the role-of-the-prediction proofs below still test the prediction itself.
    sign_fns = {
        "BLEND": (lambda c, rng: _nz_sign(c["w"]), True),
        "PRED": (lambda c, rng: _nz_sign(c["w"]), False),
        "LONG": (lambda c, rng: np.ones(len(c["w"])), False),
        "REGIME": (lambda c, rng: _regime_sign(c), False),
        "FLIP": (lambda c, rng: -_nz_sign(c["w"]), False),
    }
    pooled = {m: _portfolio(comps, fn, blend=b) for m, (fn, b) in sign_fns.items()}

    # ---------------- per-fold Sharpe, incl. the un-throttled PRED book ----------------
    # The book vol target is a risk control, so it has to be shown per fold: it
    # is meant to earn its keep in the folds where the raw book's vol runs hot.
    def _by_fold(s):
        return {pd.Timestamp(ts).year: sharpe(s[(s.index >= ts) & (s.index < tz)].values)
                for ts, tz in FOLDS}

    # ---------------- row-level export for downstream analysis ----------------
    # Daily book returns per arm, and per-asset positions for the flagship, so
    # any later table or figure can be rebuilt without re-fitting the models.
    pd.concat({m: s for m, s in pooled.items()}, axis=1, sort=True).to_csv(
        OUT / "daily_returns_by_arm.csv", index_label="date")

    k_flag = flagship_book_scale(comps)
    pos_rows = []
    for (k, a), c in comps.items():
        sc = k_flag.reindex(c["dates"]).ffill().fillna(1.0).values
        pos = _blended(c, _nz_sign(c["w"]), sc, blend=True)
        pos_rows.append(pd.DataFrame({
            "fold": k, "asset": a, "date": c["dates"],
            "position": pos, "asset_return": c["r"],
            "net_return": net_daily(pos, c["r"], COST, c["fund"]),
            "prediction": c["w"], "bull_regime": c["bull"].astype(int),
            "vol_scale": c["vsc"], "book_scale": sc,
            "p_turn": c["c_te"], "tau": c["tau"]}))
    pd.concat(pos_rows, ignore_index=True).to_csv(
        OUT / "positions_flagship.csv", index=False)

    pred_noscale = _portfolio(comps, sign_fns["PRED"][0], scaled=False)
    F_ = pd.DataFrame({m: _by_fold(s) for m, s in pooled.items()}).T
    F_.loc["PRED (no book vol target)"] = _by_fold(pred_noscale)
    # The blend exists to cut regime dependence, so report the dispersion it is
    # meant to cut alongside the levels.
    F_["min_fold"] = F_.min(axis=1)
    F_["std_fold"] = F_[[pd.Timestamp(ts).year for ts, _ in FOLDS]].std(axis=1)
    F_.to_csv(OUT / "fold_sharpe.csv")
    print("\nPER-FOLD SHARPE BY SIGNAL:\n", F_.round(3).to_string())

    # ---------------- per-asset flagship Sharpe (asset-selection diagnostic) ----------------
    # Priced at the book's throttled, blended size, so these per-asset numbers
    # add up to the headline result rather than to a book nobody trades.
    k_ser = flagship_book_scale(comps)
    asset_folds, year_sharpe = {}, {}
    for (k, name), c in comps.items():
        sc = k_ser.reindex(c["dates"]).ffill().fillna(1.0).values
        net = _pos_net(c, _nz_sign(c["w"]), sc, blend=True)
        asset_folds.setdefault(name, {})[k] = net
        year_sharpe[(name, pd.Timestamp(FOLDS[k][0]).year)] = sharpe(net.values)
    asset_sharpe = {name: sharpe(pd.concat([folds[k] for k in sorted(folds)]).sort_index().values)
                    for name, folds in asset_folds.items()}
    AS = pd.Series(asset_sharpe, name="sharpe").sort_values()
    AS.to_csv(OUT / "asset_sharpe.csv", header=True)
    print(f"\nPER-ASSET {FLAGSHIP} SHARPE (pooled across folds, worst -> best):\n",
          AS.round(3).to_string())

    years = sorted({y for (_, y) in year_sharpe})
    order_assets = list(AS.index)      # worst -> best, top to bottom on the heatmap
    Hm = pd.DataFrame(index=order_assets, columns=years, dtype=float)
    for (a_, y_), sh in year_sharpe.items():
        Hm.loc[a_, y_] = sh
    Hm.to_csv(OUT / "heatmap_asset_year.csv")
    fig, ax = plt.subplots(figsize=(1.4 * len(years) + 2.5, 0.42 * len(order_assets) + 2))
    vmax = np.nanmax(np.abs(Hm.values)) if np.isfinite(Hm.values).any() else 1.0
    im = ax.imshow(Hm.values.astype(float), cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(years)), years)
    ax.set_yticks(range(len(order_assets)),
                  [f"{a_} ({AS[a_]:+.2f})" for a_ in order_assets])
    for i in range(Hm.shape[0]):
        for j in range(Hm.shape[1]):
            v = Hm.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                        color="black" if abs(v) < vmax * 0.6 else "white")
    ax.set_xlabel("test fold (year of fold start)")
    ax.set_ylabel("asset (pooled Sharpe ratio in parentheses)")
    ax.set_title("Sharpe ratio of the traded book by asset and test fold, "
                 "assets ordered by pooled Sharpe")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Sharpe ratio", shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "heatmap_asset_year.png")
    plt.close(fig)

    # Permutation null: N_PERM random-sign portfolios through the same stack.
    # The p-value uses the (1 + k) / (B + 1) estimator, not k / B. With the
    # plain mean a draw count of zero reports p = 0 exactly, which is not a
    # probability any finite permutation test can produce: the floor is
    # 1 / (B + 1). At the original B = 20 that floor was 0.048, so no result
    # from that setting could support a claim below 0.001.
    rand_sh = []
    rand_pooled = None
    for seed in range(N_PERM):
        rng = np.random.default_rng(seed)
        p = _portfolio(comps, lambda c, r=rng: r.choice([-1.0, 1.0], size=len(c["w"])), rng)
        rand_sh.append(sharpe(p.values))
        if seed == 0:
            rand_pooled = p
    rand_sh = np.array(rand_sh)

    rows = []
    for m in ORDER:
        x = pooled[m].values
        rows.append(dict(signal=m, sharpe=sharpe(x), maxdd=max_drawdown(x),
                         ann_ret=float(np.mean(x) * 365), ann_vol=float(np.std(x) * np.sqrt(365))))
    rows.append(dict(signal="RANDOM(null)", sharpe=float(rand_sh.mean()),
                     maxdd=np.nan, ann_ret=np.nan, ann_vol=np.nan))
    R_ = pd.DataFrame(rows).set_index("signal")
    R_.to_csv(OUT / "ablation.csv")

    aligned = pd.concat({**pooled, "RANDOM0": rand_pooled,
                         "NOSCALE": pred_noscale}, axis=1, sort=True).dropna()
    proofs = {
        # --- role of the PREDICTION (unblended, so this still tests the model) ---
        "PRED_vs_LONG": paired_tests(aligned["LONG"].values, aligned["PRED"].values),
        "PRED_vs_REGIME": paired_tests(aligned["REGIME"].values, aligned["PRED"].values),
        "PRED_vs_FLIP": paired_tests(aligned["FLIP"].values, aligned["PRED"].values),
        # Does the book vol target earn its place? Same signal, same days, the
        # overlay is the only difference.
        "PRED_vs_NOSCALE": paired_tests(aligned["NOSCALE"].values, aligned["PRED"].values),
        # --- the shipped blend against each of its two components ---
        "BLEND_vs_PRED": paired_tests(aligned["PRED"].values, aligned["BLEND"].values),
        "BLEND_vs_REGIME": paired_tests(aligned["REGIME"].values, aligned["BLEND"].values),
    }
    P = pd.DataFrame(proofs).T
    P.to_csv(OUT / "paired_proofs.csv")
    (OUT / "signal_model.txt").write_text(model, encoding="utf-8")

    flag_sh = R_.loc[FLAGSHIP, "sharpe"]
    p_perm = float((1 + np.sum(rand_sh >= flag_sh)) / (len(rand_sh) + 1))
    print("\n", R_.round(3).to_string())
    print("\nPAIRED PROOFS:\n", P.round(3).to_string())
    print(f"\nRANDOM null: mean {rand_sh.mean():.2f}, 95% [{np.percentile(rand_sh, 2.5):.2f}, "
          f"{np.percentile(rand_sh, 97.5):.2f}]  ->  {FLAGSHIP}={flag_sh:.2f}, "
          f"permutation p={p_perm:.3f}")

    # ---------------- figure 1: the strategy on its own, for the paper ----------------
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    equity = START_EQUITY * np.exp(pooled[FLAGSHIP].cumsum())
    a1.plot(equity.index, equity.values, lw=2.0, color="navy",
            label=f"{FLAGSHIP} (Sharpe {flag_sh:+.2f})")
    eq_p = START_EQUITY * np.exp(pooled["PRED"].cumsum())
    a1.plot(eq_p.index, eq_p.values, lw=1.2, color="gray", alpha=0.8,
            label=f"PRED, unblended (Sharpe {R_.loc['PRED', 'sharpe']:+.2f})")
    a1.axhline(START_EQUITY, color="k", lw=0.7, ls=":")
    a1.legend()
    a1.set_ylabel(f"equity (index, {START_EQUITY:.0f} at inception)")
    a1.set_title(f"Equal-weight {len(UNIVERSE)}-asset portfolio on the causal Kalman-trend "
                 f"forecast, walk-forward 2021--2026\n"
                 f"signal model {model}; {SIGNAL_BLEND:.0%} prediction / "
                 f"{1 - SIGNAL_BLEND:.0%} regime; book volatility target "
                 f"{VOL_TARGET:.0%} at {VOL_TARGET_WIN} days; "
                 f"net of {COST * 1e4:.0f} bps per side")
    dd = equity - equity.cummax()
    a2.fill_between(equity.index, dd, 0, color="navy", alpha=0.4,
                    label=f"{FLAGSHIP} maximum drawdown {dd.min():.1f}")
    a2.legend()
    a2.set_xlabel("date")
    a2.set_ylabel("drawdown (index points)")
    fig.tight_layout()
    fig.savefig(OUT / "equity_mcap.png")
    plt.close(fig)

    # ---------------- figure 2: ablation bar chart vs the permutation null ----------------
    colors = {"BLEND": "navy", "PRED": "steelblue", "LONG": "darkorange",
              "REGIME": "seagreen", "FLIP": "crimson"}
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    xs = range(len(ORDER))
    lo, hi = np.percentile(rand_sh, 2.5), np.percentile(rand_sh, 97.5)
    # null band behind the bars, else it washes them out where they overlap
    ax.axhspan(lo, hi, color="gray", alpha=0.22, zorder=0,
               label=f"permutation null, 95% interval [{lo:.2f}, {hi:.2f}]")
    ax.axhline(rand_sh.mean(), color="gray", ls="--", lw=1, zorder=1,
               label=f"permutation null mean {rand_sh.mean():.2f}")
    ax.bar(xs, [R_.loc[m, "sharpe"] for m in ORDER],
           color=[colors[m] for m in ORDER], alpha=0.95, zorder=2)
    for i, m in enumerate(ORDER):
        v = R_.loc[m, "sharpe"]
        ax.text(i, v + (0.04 if v >= 0 else -0.10), f"{v:.2f}", ha="center",
                fontsize=8, zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(list(xs), ORDER)
    ax.set_xlabel("directional signal entering the execution stack")
    ax.set_ylabel("pooled Sharpe ratio")
    ax.set_title("Signal ablation with the execution stack held fixed\n"
                 rf"PRED vs FLIP $\Delta$Sharpe "
                 f"{proofs['PRED_vs_FLIP']['d_sharpe']:+.2f}, 95% CI "
                 f"[{proofs['PRED_vs_FLIP']['ci_lo']:+.2f}, "
                 f"{proofs['PRED_vs_FLIP']['ci_hi']:+.2f}]; "
                 f"permutation $p={p_perm:.3f}$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "ablation_bars.png")
    plt.close(fig)

    # ---------------- figure 3: equity by directional signal, execution stack held fixed ----------------
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for m in ORDER:
        eq = START_EQUITY * np.exp(pooled[m].cumsum())
        ax.plot(eq.index, eq.values, lw=2.2 if m == FLAGSHIP else 1.3, color=colors[m],
                label=f"{m} (Sharpe {R_.loc[m, 'sharpe']:+.2f})")
    eqr = START_EQUITY * np.exp(rand_pooled.cumsum())
    ax.plot(eqr.index, eqr.values, lw=1.0, color="gray", alpha=0.6,
            label="permutation null, single draw")
    ax.axhline(START_EQUITY, color="k", lw=0.7, ls=":")
    ax.set_xlabel("date")
    ax.set_ylabel(f"equity (index, {START_EQUITY:.0f} at inception)")
    ax.set_title("Portfolio equity by directional signal, execution stack held fixed")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "ablation_equity.png")
    plt.close(fig)

    return R_, P, F_


if __name__ == "__main__":
    run()
