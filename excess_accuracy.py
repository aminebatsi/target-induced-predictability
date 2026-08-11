"""Forecast-side inference: accuracy intervals and excess accuracy.

  1.  dependence-aware bootstrap intervals for the per-fold accuracy table on
      the ten portfolio assets;
  2.  whether the pooled 0.481 next-day accuracy reflects a common short-horizon
      reversal, tested on a single model-free signal instead of treating
      thirteen dependent models as thirteen confirmations;
  3.  whether the evaluation and portfolio universes really differ on raw
      h-day accuracy;
  4.  excess directional accuracy, ExDA = DA - B, with the arcsine benchmark
      B computed CELL BY CELL on the same anchors and then aggregated. The
      nonlinear arcsine is never evaluated at a pooled correlation.

No model is refitted. Predictions are read from the exported panel; the filter
is recomputed from cached prices so that rho_1 is measured on exactly the
anchors the models were scored on.

Outputs -> results/analysis/
"""
import numpy as np
import pandas as pd

from config import ASSETS, FOLDS, L
from data import load_asset
from evaluation import fold_indices, norm_end as _norm_end
from portfolio_replay import ROOT, SIGNS, load_components, nz_sign, stationary_bootstrap_idx
from predictive_ability import MODELS, PRED_DIR
from targets import kalman_causal

OUT = ROOT / "results" / "analysis"
B_BIG = 10000
B_EXDA = 10000
BLOCK = 20
EVAL = list(ASSETS)
FOLD_YEAR = {k: 2021 + k for k in range(5)}


# ================================================================ panel
# The filter's normalising constant is a preprocessing parameter, and the trend
# recomputed here must be the trend the models were actually scored against or
# the accuracies in the two tables will not agree. `evaluation.norm_end` is the
# single definition the forecasting run itself uses; it is imported rather than
# restated so the two cannot drift apart.


def build_cells(h=7):
    """One record per (asset, fold): dates, trend target, the increment the
    slope rule extrapolates, and the raw price outcomes -- on exactly the test
    anchors used by forecast.py."""
    cells = {}
    for asset in EVAL:
        df = load_asset(asset)
        dts, lp_f = df["date"].values, df["logprice"].values.astype(float)
        for k, (ts, tz) in enumerate(FOLDS):
            cut = np.searchsorted(dts, np.datetime64(tz) + np.timedelta64(2, "D"))
            dates, lp = dts[:cut], lp_f[:cut]
            if len(lp) < 700:
                continue
            ne = _norm_end(dates, ts, tz, h)
            if ne is None:
                continue
            trend, _ = kalman_causal(lp, ne)
            anchors = np.arange(L - 1, len(lp) - h)
            idx = fold_indices(dates[anchors], ts, tz)
            if idx is None:
                continue
            a = anchors[idx[2]]
            cells[(asset, k)] = dict(
                dates=pd.DatetimeIndex(dates[a]),
                y_true=trend[a + h] - trend[a],
                inc=trend[a] - trend[a - 1],
                r1=lp[a + 1] - lp[a],
                rh=lp[a + h] - lp[a])
    return cells


def attach_predictions(cells):
    """Join each model's exported predictions onto the recomputed anchors."""
    preds = {}
    for m in MODELS:
        fp = PRED_DIR / f"forecast__{m}__predictions.csv"
        if not fp.exists():
            continue
        d = pd.read_csv(fp, parse_dates=["date"])
        for (asset, k), c in cells.items():
            s = d[(d.asset == asset) & (d.fold == k)].set_index("date")["y_pred"]
            v = s.reindex(c["dates"]).values
            assert np.isfinite(v).all(), f"missing predictions for {m} {asset} {k}"
            preds.setdefault(m, {})[(asset, k)] = v
    return preds


# ================================================================ 5A-5C ExDA
def benchmark(rho):
    return 0.5 + np.arcsin(np.clip(rho, -1.0, 1.0)) / np.pi


def exda_point(cells, preds):
    """Cell-by-cell ExDA, aggregated anchor-weighted to match the pooled DA the
    manuscript reports."""
    rows = []
    for name, series in list(preds.items()) + [("SLOPE", None)]:
        num_da = num_b = den = 0.0
        for key, c in cells.items():
            yp = c["inc"] if series is None else series[key]
            hit = (np.sign(yp) == np.sign(c["y_true"])).mean()
            rho = np.corrcoef(c["y_true"], c["inc"])[0, 1]
            n = len(c["y_true"])
            num_da += hit * n
            num_b += benchmark(rho) * n
            den += n
        rows.append(dict(model=name, DA=num_da / den, benchmark=num_b / den,
                         ExDA=(num_da - num_b) / den, n=int(den)))
    return pd.DataFrame(rows)


def exda_bootstrap(cells, preds, B=B_EXDA, block=BLOCK, seed=31):
    """Date-block bootstrap that recomputes DA, rho_1 and the benchmark inside
    every draw, so the interval carries the benchmark's own uncertainty."""
    all_dates = pd.DatetimeIndex(sorted(set().union(
        *[set(c["dates"]) for c in cells.values()])))
    pos = {d: i for i, d in enumerate(all_dates)}
    nD = len(all_dates)

    # for each cell, map global date slot -> local row (or -1)
    lut, data = {}, {}
    for key, c in cells.items():
        t = np.full(nD, -1, np.int64)
        for j, d in enumerate(c["dates"]):
            t[pos[d]] = j
        lut[key] = t
        data[key] = c

    names = list(preds) + ["SLOPE"]
    draws = {n: np.empty(B) for n in names}
    idx = stationary_bootstrap_idx(nD, B, block, seed)

    for b in range(B):
        seq = idx[b]
        num = {n: 0.0 for n in names}
        numb = 0.0
        den = 0.0
        for key, c in data.items():
            rows = lut[key][seq]
            rows = rows[rows >= 0]
            if len(rows) < 30:
                continue
            yt = c["y_true"][rows]
            inc = c["inc"][rows]
            sd = yt.std() * inc.std()
            rho = 0.0 if sd < 1e-18 else float(np.corrcoef(yt, inc)[0, 1])
            nb = benchmark(rho)
            n = len(rows)
            den += n
            numb += nb * n
            st_yt = np.sign(yt)
            for nm in names:
                yp = inc if nm == "SLOPE" else preds[nm][key][rows]
                num[nm] += float((np.sign(yp) == st_yt).mean()) * n
        for nm in names:
            draws[nm][b] = (num[nm] - numb) / den

    out = []
    for nm in names:
        d = draws[nm]
        out.append(dict(model=nm, ExDA_boot_mean=float(d.mean()),
                        lo=float(np.percentile(d, 2.5)),
                        hi=float(np.percentile(d, 97.5)),
                        p_gt0=float((1 + np.sum(d <= 0)) / (B + 1)), B=B))
    fitted = np.mean([draws[n] for n in preds], axis=0)
    out.append(dict(model="MEAN of 13 fitted", ExDA_boot_mean=float(fitted.mean()),
                    lo=float(np.percentile(fitted, 2.5)),
                    hi=float(np.percentile(fitted, 97.5)),
                    p_gt0=float((1 + np.sum(fitted <= 0)) / (B + 1)), B=B))
    return pd.DataFrame(out)


# ================================================================ 4A Table 2
def table2_ci(B=B_BIG, block=BLOCK, seed=41):
    """Bootstrap the per-fold accuracy table on the ten portfolio assets.

    Within a fold the daily asset-averaged correctness series is resampled by
    date blocks; the five fold estimates are then averaged with the equal
    weighting the table itself uses.
    """
    store = load_components()
    comps = store["GRU"]
    truths = {"trend": "d_te", "price_h": "fwd_h", "price_1": "r"}
    fold_series = {t: {} for t in truths}
    for (k, a), c in comps.items():
        s = nz_sign(c["w"])
        for t, col in truths.items():
            fold_series[t].setdefault(k, []).append(
                pd.Series((s == np.sign(c[col])).astype(float), index=c["dates"]))

    rows, draws_by_t = [], {}
    for t in truths:
        per_fold_daily = {k: pd.concat(v, axis=1).mean(axis=1).sort_index()
                          for k, v in fold_series[t].items()}
        point = np.mean([v.mean() for v in per_fold_daily.values()])
        ests = np.zeros(B)
        for k, v in per_fold_daily.items():
            x = v.values
            I = stationary_bootstrap_idx(len(x), B, block, seed + k)
            ests += x[I].mean(axis=1)
        ests /= len(per_fold_daily)
        draws_by_t[t] = ests
        rows.append(dict(truth=t, point=float(point),
                         boot_mean=float(ests.mean()),
                         lo=float(np.percentile(ests, 2.5)),
                         hi=float(np.percentile(ests, 97.5)),
                         p_vs_half=float((1 + np.sum(ests <= 0.5)) / (B + 1)),
                         block=block, B=B))
    return pd.DataFrame(rows), draws_by_t


# ================================================================ 4B sub-chance
def subchance(cells, B=B_BIG, block=BLOCK, seed=53):
    """Is the below-chance next-day accuracy a common short-horizon reversal?

    Tested on ONE model-free signal, s_t = sign(l_t - l_{t-1}), rather than on
    thirteen dependent models. Dates are blocked so cross-asset observations on
    the same day move together.
    """
    frames = [pd.DataFrame({"date": c["dates"],
                            "s": np.sign(c["inc"]),
                            "r": c["r1"]}) for c in cells.values()]
    D = pd.concat(frames, ignore_index=True)
    D.loc[D.s == 0, "s"] = 1.0
    g = D.groupby("date").agg(hit=("s", lambda z: np.nan),
                              n=("s", "size"))
    # per-date means of the two quantities of interest
    D["hit"] = (D.s == np.sign(D.r)).astype(float)
    D["sr"] = D.s * D.r
    G = D.groupby("date")[["hit", "sr"]].mean()
    hit, sr = G["hit"].values, G["sr"].values
    I = stationary_bootstrap_idx(len(G), B, block, seed)

    da_d = hit[I].mean(axis=1)
    sr_d = sr[I].mean(axis=1)

    # regression r_{t+1} = alpha + beta s_t, blocked by date
    Gr = D.groupby("date").agg(s=("s", "mean"), r=("r", "mean"))
    x, y = Gr["s"].values, Gr["r"].values
    v = x.var()
    beta = float(((x - x.mean()) * (y - y.mean())).mean() / v)
    bb = np.empty(B)
    for b in range(B):
        xx, yy = x[I[b]], y[I[b]]
        vv = xx.var()
        bb[b] = ((xx - xx.mean()) * (yy - yy.mean())).mean() / vv if vv > 0 else 0.0

    return dict(n_dates=int(len(G)), n_obs=int(len(D)),
                da=float(hit.mean()),
                da_lo=float(np.percentile(da_d, 2.5)),
                da_hi=float(np.percentile(da_d, 97.5)),
                p_da_below_half=float((1 + np.sum(da_d >= 0.5)) / (B + 1)),
                signed_bps=float(sr.mean() * 1e4),
                signed_lo=float(np.percentile(sr_d, 2.5) * 1e4),
                signed_hi=float(np.percentile(sr_d, 97.5) * 1e4),
                beta=beta,
                beta_lo=float(np.percentile(bb, 2.5)),
                beta_hi=float(np.percentile(bb, 97.5)),
                block=block, B=B)


# ================================================================ 4D universes
def universe_gap(cells, preds, B=B_BIG, block=BLOCK, seed=67):
    """DA on the raw h-day change, portfolio universe minus evaluation universe,
    for the GRU signal. The two universes hold different assets, so the draw is
    unpaired: each universe is resampled on its own dates with the same block
    length and the difference is taken within a draw."""
    ev = [pd.DataFrame({"date": c["dates"],
                        "hit": (np.sign(preds["GRU"][k]) == np.sign(c["rh"]))
                        .astype(float)}) for k, c in cells.items()]
    E = pd.concat(ev).groupby("date").hit.mean()

    store = load_components()
    pf = []
    for (k, a), c in store["GRU"].items():
        s = nz_sign(c["w"])
        pf.append(pd.DataFrame({"date": c["dates"],
                                "hit": (s == np.sign(c["fwd_h"])).astype(float)}))
    P = pd.concat(pf).groupby("date").hit.mean()

    Ie = stationary_bootstrap_idx(len(E), B, block, seed)
    Ip = stationary_bootstrap_idx(len(P), B, block, seed + 1)
    de = E.values[Ie].mean(axis=1)
    dp = P.values[Ip].mean(axis=1)
    diff = dp - de
    return dict(da_eval=float(E.mean()), da_port=float(P.mean()),
                delta=float(P.mean() - E.mean()),
                lo=float(np.percentile(diff, 2.5)),
                hi=float(np.percentile(diff, 97.5)),
                p_two_sided=float(2 * min((1 + np.sum(diff <= 0)) / (B + 1),
                                          (1 + np.sum(diff >= 0)) / (B + 1))),
                block=block, B=B)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = build_cells()
    preds = attach_predictions(cells)
    print(f"cells {len(cells)}, models {len(preds)}, "
          f"anchors {sum(len(c['y_true']) for c in cells.values())}")

    print("\n== 5A/5B excess directional accuracy (cell-by-cell benchmark) ==")
    E = exda_point(cells, preds)
    E = E.sort_values("ExDA", ascending=False)
    E.to_csv(OUT / "exda_point.csv", index=False)
    print(E.round(4).to_string(index=False))
    fitted = E[E.model != "SLOPE"]
    print(f"  mean of 13 fitted: DA {fitted.DA.mean():.4f}  "
          f"benchmark {fitted.benchmark.mean():.4f}  ExDA {fitted.ExDA.mean():+.4f}")

    print("\n== 5C bootstrap ExDA ==")
    EB = exda_bootstrap(cells, preds)
    EB.to_csv(OUT / "exda_bootstrap.csv", index=False)
    print(EB.round(4).to_string(index=False))

    print("\n== 4A per-fold accuracy table, bootstrap intervals ==")
    T2, _ = table2_ci()
    T2.to_csv(OUT / "table2_ci.csv", index=False)
    print(T2.round(4).to_string(index=False))

    print("\n== 4B sub-chance diagnostic on the model-free slope sign ==")
    S = subchance(cells)
    pd.Series(S).to_frame("value").to_csv(OUT / "subchance.csv")
    for k, v in S.items():
        print(f"  {k:20} {v}")

    print("\n== 4D evaluation vs portfolio universe, raw h-day accuracy ==")
    U = universe_gap(cells, preds)
    pd.Series(U).to_frame("value").to_csv(OUT / "universe_gap.csv")
    for k, v in U.items():
        print(f"  {k:20} {v}")

    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
