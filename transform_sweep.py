"""Does "accuracy on a smoothed target measures the smoother" hold beyond one
smoother and one horizon? A model-free sweep.

Nothing is fitted anywhere in this module. Every cell is the parameter-free
SLOPE rule -- extrapolate the last one-day increment of the smoothed series --
scored on a target built by one causal smoother at one horizon for one asset in
one walk-forward fold. That is the sharpest available form of the paper's
claim: if a rule with no parameters tracks a single property of the target
across smoothers, strengths, horizons and assets, then the accuracy belongs to
the transform.

THE PROPERTY. For the SLOPE rule the relevant quantity is not the persistence
of the h-step change but its correlation with the single increment the rule
extrapolates,

    rho_1 = Corr(level_{t+h} - level_t,  level_t - level_{t-1}).

If that pair were bivariate normal with zero means, the probability that the
two share a sign has the closed form

    P(sign agree) = 1/2 + arcsin(rho_1) / pi,

so the accuracy would be a deterministic function of rho_1 with no room for a
model to contribute anything. The sweep tests that prediction directly: the
`da_gaussian` column is the curve, `da_slope` is what the rule actually scores,
and the gap between them is how much of the reported accuracy is NOT explained
by the transform.

DESIGN. Strictly causal smoothers only, at several smoothing strengths:
the local-linear-trend Kalman over five measurement-noise settings (the
filter's smoothing dial), the local-level Kalman, causal EMA over five spans,
trailing-edge Savitzky-Golay over three windows, and the unsmoothed price as
the zero-smoothing anchor. Horizons h in {1, 3, 7, 14, 30}. Fifteen assets,
five purged walk-forward folds, test anchors only.

Outputs -> results/analysis/
"""
import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.stats import skew as _skew  # noqa: F401  (kept explicit for clarity)
from scipy.signal import savgol_coeffs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (ASSET_CLASSES, ASSETS, FOLDS, L, ORIGINAL_ASSET_CLASSES,
                    ORIGINAL_ASSETS, PLOT_RC)
from data import load_asset
from portfolio_replay import OUT
from targets import kalman_causal

plt.rcParams.update(PLOT_RC)

HORIZONS = [1, 3, 7, 14, 30]
UNIVERSE = list(dict.fromkeys(list(ASSETS) + list(ORIGINAL_ASSETS)))
EVAL_ASSETS = list(ASSETS)
CLASSES = {**ORIGINAL_ASSET_CLASSES, **ASSET_CLASSES}
MIN_TRAIN, MIN_TEST = 400, 100
SHIPPED_H = 7                      # the manuscript's primary horizon
SHIPPED_TAG = "(shipped)"          # marks the manuscript's Kalman setting


# ------------------------------------------------------------------ smoothers
def _sg_trailing(p, win, poly=2):
    """Savitzky-Golay read at the RIGHT EDGE of a trailing window: the same
    polynomial smoothing as the textbook version with no future sample in it."""
    p = np.asarray(p, float)
    c = savgol_coeffs(win, poly, pos=win - 1, use="dot")
    out = np.empty(len(p))
    for t in range(len(p)):
        out[t] = p[:t + 1].mean() if t < win - 1 else float(c @ p[t - win + 1:t + 1])
    return out


def _kalman_level_only(p, norm_end, q=1e-5, r0=1e-3):
    """Local level, no velocity state. Same adaptive measurement noise."""
    p = np.asarray(p, float)
    n = len(p)
    ret = np.diff(p, prepend=p[0])
    rv = pd.Series(ret ** 2).ewm(span=20).mean().values
    rv = rv / (np.median(rv[20:max(norm_end, 40)]) + 1e-18)
    x, P = p[0], 1.0
    out = np.empty(n)
    for t in range(n):
        P += q
        R = r0 * rv[t]
        K = P / (P + R)
        x += K * (p[t] - x)
        P *= (1 - K)
        out[t] = x
    return out


# label -> (family, strength tag, callable(logprice, norm_end) -> level)
def smoothers():
    S = {}
    for r0 in (1e-4, 1e-3, 1e-2, 1e-1, 1e0):
        tag = f"kalman r={r0:g}" + (" (shipped)" if r0 == 1e-3 else "")
        S[tag] = ("Kalman local linear trend", r0,
                  lambda p, ne, _r=r0: kalman_causal(p, ne, r=_r)[0])
    S["kalman local level"] = ("Kalman local level", 1e-3,
                               lambda p, ne: _kalman_level_only(p, ne))
    for span in (5, 10, 20, 40, 80):
        S[f"EMA span={span}"] = ("causal EMA", span,
                                 lambda p, ne, _s=span: pd.Series(np.asarray(p, float))
                                 .ewm(span=_s, adjust=False).mean().values)
    for win in (11, 21, 41):
        S[f"SG trailing w={win}"] = ("trailing Savitzky-Golay", win,
                                     lambda p, ne, _w=win: _sg_trailing(p, _w))
    S["none (raw log price)"] = ("no smoothing", 0,
                                 lambda p, ne: np.asarray(p, float))
    return S


# ------------------------------------------------------------------ splits
def fold_split(anchor_dates, ts, tz, h):
    """Purged chronological split in the asset's own observation index.

    Same rule as `evaluation.fold_indices`, with the horizon under test rather
    than the package's h = 7 constant: a pre-test anchor is kept only if its
    label endpoint t+h falls strictly before the first test anchor. Nothing is
    fitted in this sweep, so the split affects only the smoother's normalising
    constant; the rule is matched to the fitted-model protocol so that the two
    parts of the study cannot diverge.
    """
    ts, tz = np.datetime64(ts), np.datetime64(tz)
    te = np.where((anchor_dates >= ts) & (anchor_dates < tz))[0]
    if len(te) < MIN_TEST:
        return None
    pre = np.arange(0, max(int(te[0]) - int(h), 0), dtype=np.intp)
    if len(pre) < MIN_TRAIN:
        return None
    assert pre[-1] + int(h) < te[0], "pre-test label crosses the test boundary"
    return pre, te


def norm_end_for(dates, ts, tz, h):
    """Index bounding what the smoother's normalising constant may see: the end
    of the training segment, exactly as forecast.py pins it."""
    anchors = np.arange(L - 1, len(dates) - h)
    sp = fold_split(dates[anchors], ts, tz, h)
    if sp is None:
        return None
    return int(anchors[sp[0][-1]]) + 1


# ------------------------------------------------------------------ sweep
def sweep():
    S = smoothers()
    rows, slope_daily = [], []
    for asset in UNIVERSE:
        df = load_asset(asset)
        dates_f = df["date"].values
        lp_f = df["logprice"].values.astype(float)
        for k, (ts, tz) in enumerate(FOLDS):
            cut = np.searchsorted(dates_f, np.datetime64(tz) + np.timedelta64(2, "D"))
            dates, lp = dates_f[:cut], lp_f[:cut]
            if len(lp) < 700:
                continue
            # One normalising constant per (asset, fold), pinned at the primary
            # horizon, so a smoother is the SAME transform across horizons and
            # only the target moves.
            ne = norm_end_for(dates, ts, tz, 7)
            if ne is None:
                continue
            levels = {name: fn(lp, ne) for name, (_, _, fn) in S.items()}
            dlp = np.diff(lp)
            for h in HORIZONS:
                anchors = np.arange(L - 1, len(lp) - h)
                sp = fold_split(dates[anchors], ts, tz, h)
                if sp is None:
                    continue
                a = anchors[sp[1]]
                for name, lev in levels.items():
                    fam, strength, _ = S[name]
                    tgt = lev[a + h] - lev[a]
                    inc = lev[a] - lev[a - 1]
                    if np.std(tgt) < 1e-15 or np.std(inc) < 1e-15:
                        continue
                    rho = float(np.corrcoef(tgt, inc)[0, 1])
                    hit = (np.sign(inc) == np.sign(tgt)).astype(float)
                    # The Gaussian benchmark assumes zero means. Centering
                    # leaves rho unchanged, so comparing sign agreement before
                    # and after centering isolates how much of the residual is
                    # attributable to nonzero location (drift) alone.
                    hit_c = (np.sign(inc - inc.mean())
                             == np.sign(tgt - tgt.mean())).astype(float)
                    rows.append(dict(
                        asset=asset, asset_class=CLASSES.get(asset, "?"),
                        fold=k, h=h, smoother=name, family=fam,
                        strength=strength, n=len(a), rho1=rho,
                        da_slope=float(hit.mean()),
                        da_slope_centered=float(hit_c.mean()),
                        da_gaussian=0.5 + np.arcsin(np.clip(rho, -1, 1)) / np.pi,
                        skew_tgt=float(st.skew(tgt)),
                        kurt_tgt=float(st.kurtosis(tgt, fisher=False)),
                        mean_tgt_over_sd=float(tgt.mean() / (tgt.std() + 1e-18)),
                        mean_inc_over_sd=float(inc.mean() / (inc.std() + 1e-18)),
                        da_price_1=float(np.mean(np.sign(inc) == np.sign(lp[a + 1] - lp[a]))),
                        da_price_h=float(np.mean(np.sign(inc) == np.sign(lp[a + h] - lp[a]))),
                        smooth_ratio=float(np.std(np.diff(lev)) / (np.std(dlp) + 1e-18)),
                        ac1_increment=float(np.corrcoef(np.diff(lev)[1:],
                                                        np.diff(lev)[:-1])[0, 1])))
                    if (h == SHIPPED_H and SHIPPED_TAG in name
                            and asset in EVAL_ASSETS):
                        slope_daily.append(pd.DataFrame(
                            {"date": dates[a], "hit_trend": hit}))
    T = pd.DataFrame(rows)
    D = (pd.concat(slope_daily).groupby("date").mean() if slope_daily
         else pd.DataFrame(columns=["hit_trend"]))
    return T, D


# ------------------------------------------------------------------ reporting
def report(T):
    ok = T.dropna(subset=["rho1", "da_slope"])
    r = st.pearsonr(ok.rho1, ok.da_slope)
    rs = st.spearmanr(ok.rho1, ok.da_slope)
    gap = ok.da_slope - ok.da_gaussian
    full = len(smoothers()) * len(HORIZONS) * len(UNIVERSE) * len(FOLDS)
    print(f"\ncells: {len(ok)} of a full Cartesian product of {full}  "
          f"({ok.smoother.nunique()} transforms x {ok.h.nunique()} horizons "
          f"x {ok.asset.nunique()} assets x {ok.fold.nunique()} folds)")
    if len(ok) < full:
        seen = set(zip(ok.asset, ok.fold, ok.h))
        gone = [(a, k, h) for a in UNIVERSE for k in range(len(FOLDS))
                for h in HORIZONS if (a, k, h) not in seen]
        # A cell is dropped only when the purged pre-test block is too short for
        # the protocol's 400-anchor minimum, never on anything result-dependent.
        for a, k, h in gone:
            print(f"  dropped (asset, fold, h) = ({a}, {k}, {h}): "
                  f"insufficient purged training history at that horizon "
                  f"-> {len(smoothers())} transform cells lost")
    # Reported descriptively. Horizons, transforms and folds share underlying
    # price histories, so these cells are not independent replications and no
    # p-value is attached to the correlation.
    print(f"rho_1 vs slope accuracy : Pearson r = {r[0]:+.4f}, "
          f"Spearman = {rs[0]:+.4f}  [descriptive; cells are not independent]")
    print(f"arcsin law               : mean |da_slope - da_gaussian| = "
          f"{gap.abs().mean():.4f}, bias {gap.mean():+.4f}, "
          f"max {gap.abs().max():.4f}")
    print(f"accuracy against the next-day return vs rho_1 : "
          f"r = {st.pearsonr(ok.rho1, ok.da_price_1)[0]:+.4f}")
    print(f"accuracy against the same-horizon price change vs rho_1 : "
          f"r = {st.pearsonr(ok.rho1, ok.da_price_h)[0]:+.4f}")
    print(f"pooled DA on next-day return = {ok.da_price_1.mean():.4f} "
          f"(above 0.5 in {int((ok.da_price_1 > 0.5).sum())}/{len(ok)} cells)")
    print(f"pooled DA on same-horizon price = {ok.da_price_h.mean():.4f} "
          f"(above 0.5 in {int((ok.da_price_h > 0.5).sum())}/{len(ok)} cells)")

    piv = ok.pivot_table(index="smoother", columns="h",
                         values=["rho1", "da_slope"]).round(3)
    print("\nrho_1 and slope accuracy by smoother and horizon "
          "(mean over assets and folds):")
    print(piv.to_string())
    return dict(n_cells=int(len(ok)), n_cells_full=int(full),
                pearson_r=float(r[0]), spearman=float(rs[0]),
                mae_arcsin=float(gap.abs().mean()),
                bias_arcsin=float(gap.mean()),
                r_rho_vs_price1=float(st.pearsonr(ok.rho1, ok.da_price_1)[0]),
                da_price1_mean=float(ok.da_price_1.mean()),
                r_rho_vs_priceh=float(st.pearsonr(ok.rho1, ok.da_price_h)[0]),
                da_priceh_mean=float(ok.da_price_h.mean()))


def figure(T, path):
    ok = T.dropna(subset=["rho1", "da_slope"])
    fams = list(dict.fromkeys(ok.family))
    cols = dict(zip(fams, ["navy", "seagreen", "darkorange", "crimson", "grey"]))
    marks = {1: "o", 3: "s", 7: "D", 14: "^", 30: "v"}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # (a) the law
    for fam in fams:
        s = ok[ok.family == fam]
        a1.scatter(s.rho1, s.da_slope, s=11, alpha=0.40, marker="o",
                   color=cols[fam], linewidths=0, label=fam)
    x = np.linspace(-0.25, 0.995, 500)
    a1.plot(x, 0.5 + np.arcsin(x) / np.pi, color="black", lw=2.2, zorder=5,
            label=r"$B(\rho_1)=\frac{1}{2}+\frac{\arcsin\rho_1}{\pi}$"
                  "\n(structural reference, fitted to nothing)")
    a1.axhline(0.5, color="k", ls=":", lw=1.0)
    r = st.pearsonr(ok.rho1, ok.da_slope)[0]
    mae = (ok.da_slope - ok.da_gaussian).abs().mean()
    a1.set_xlabel(r"$\rho_1 = \mathrm{Corr}(\ell_{t+h}-\ell_t,\;"
                  r"\ell_t-\ell_{t-1})$")
    a1.set_ylabel("direction accuracy of the parameter-free SLOPE rule")
    a1.set_title(f"(a) Accuracy tracks one property of the target\n"
                 f"{len(ok)} cells (of 5625), $r={r:+.3f}$ descriptive, "
                 f"mean $|$error$|$ from the curve {mae:.3f}")
    a1.legend(loc="upper left", fontsize=8, markerscale=1.6)

    # (b) the dial
    for h, mk in marks.items():
        s = ok[ok.h == h]
        a2.scatter(s.smooth_ratio, s.da_slope, s=13, alpha=0.40, marker=mk,
                   color="#1f3f8f", linewidths=0,
                   label=f"$h={h}$" if h else None)
    a2.scatter(ok.smooth_ratio, ok.da_price_h, s=9, alpha=0.24, marker="s",
               color="#c07a2c", linewidths=0,
               label="same-horizon raw price")
    a2.scatter(ok.smooth_ratio, ok.da_price_1, s=9, alpha=0.24, marker="o",
               color="#8f8f8f", linewidths=0,
               label="next-day raw price")
    a2.axhline(0.5, color="k", ls=":", lw=1.0)
    a2.set_xscale("log")
    a2.set_xlabel(r"smoothing strength  $\mathrm{sd}(\Delta\ell)/\mathrm{sd}(\Delta p)$"
                  "   (1 = no smoothing, left = smoother)")
    a2.set_ylabel("direction accuracy")
    a2.set_title("(b) The same rule, dialled by smoothing strength alone\n"
                 "blue: transformed target; orange: same-horizon price; "
                 "grey: next-day price")
    a2.legend(loc="lower left", fontsize=8, ncol=2, markerscale=1.4,
              title=None)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T, D = sweep()
    T.to_csv(OUT / "slope_generality_sweep.csv", index=False)
    D.to_csv(OUT / "slope_daily_accuracy.csv")
    summary = report(T)
    pd.Series(summary).to_csv(OUT / "slope_generality_summary.csv", header=False)
    figure(T, OUT / "fig__slope_generality.png")

    # per (smoother, horizon) table for the manuscript
    G = (T.groupby(["family", "smoother", "h"])
         .agg(rho1=("rho1", "mean"), da_slope=("da_slope", "mean"),
              da_gaussian=("da_gaussian", "mean"),
              da_price_h=("da_price_h", "mean"),
              da_price_1=("da_price_1", "mean"),
              smooth_ratio=("smooth_ratio", "mean"), cells=("rho1", "size"))
         .reset_index())
    G.to_csv(OUT / "slope_generality_by_cell.csv", index=False)
    print(f"\n-> {OUT}")
    return T, G, summary


if __name__ == "__main__":
    main()
