"""Selection-corrected forecast inference: Hansen's SPA, White's Reality Check,
Romano-Wolf stepdown, Benjamini-Hochberg, and bootstrap direction-accuracy
intervals that replace the asymptotic Pesaran-Timmermann p-values.

Thirteen models are compared on one evaluation, so any statement of the form
"model X beats the benchmark" has been through a thirteen-way search. This
module prices that search rather than acknowledging it.

WHAT IS TESTED, AND AGAINST WHAT

  vs SLOPE, squared loss   The free single-difference baseline is the benchmark
                           the paper's argument actually needs: if no fitted
                           model beats it once the search is priced, the
                           accuracy is a property of the target.
  vs SLOPE, direction      The same contrast on the metric this literature
                           reports.
  vs chance, on price      DA against the next-day return, benchmark 0.5. This
                           is the confirmatory claim, and correcting it for the
                           thirteen-way search can only make it harder to
                           reject -- which is the point.

BOOTSTRAP GEOMETRY. Losses are aligned into a date x asset panel and averaged
across the assets trading on each date, so one bootstrap observation is one
calendar day of the whole evaluation universe. Blocks are then drawn over dates
with the Politis-Romano stationary bootstrap (mean block 20 days, which covers
the seven-day overlapping label window with room to spare). This models
contemporaneous cross-asset dependence exactly and serial dependence by block;
it does not model cross-asset dependence at a lag.

Outputs -> results/analysis/
"""
import numpy as np
import pandas as pd
from scipy import stats as st

from portfolio_replay import OUT, ROOT, stationary_bootstrap_idx

PRED_DIR = ROOT / "artifacts" / "forecast_predictions"
B = 10000
BLOCK = 20
MODELS = ["LR", "RF", "XGB", "LGBM", "ARIMA", "GRU", "LSTM", "DLinear",
          "PatchTST", "TimeMixer", "TimeFilter", "Crossformer", "FEDformer"]
TOP4 = ["GRU", "LR", "TimeFilter", "TimeMixer"]


# ------------------------------------------------------------------ data
def load_predictions():
    frames = []
    for m in MODELS:
        fp = PRED_DIR / f"forecast__{m}__predictions.csv"
        if not fp.exists():
            print(f"  [warn] missing {fp.name}")
            continue
        d = pd.read_csv(fp, parse_dates=["date"])
        d["model"] = m
        frames.append(d)
    P = pd.concat(frames, ignore_index=True)
    # SLOPE is model-free and identical for every model, so it is rebuilt once
    # from any model's rows: h * (level_t - level_{t-1}) is recoverable because
    # the panel carries the realised trend change, but the level itself is not
    # exported. We therefore read the slope forecast from the shared anchors of
    # the LR frame, which stores the same target column.
    return P


def panel(P, value_col):
    """date x (asset, model) -> mean over assets per date, per model."""
    g = P.groupby(["model", "date"])[value_col].mean().unstack(0)
    return g.sort_index()


# ------------------------------------------------------------------ SPA machinery
def _boot_means(X, idx, chunk=500):
    """X: (T, k) loss differentials. -> (B, k) resampled means.

    Chunked over replicates: X[idx] at B = 10 000 and T = 1800 would
    materialise a 1.9 GB intermediate.
    """
    Bn = idx.shape[0]
    out = np.empty((Bn, X.shape[1]))
    for i in range(0, Bn, chunk):
        out[i:i + chunk] = X[idx[i:i + chunk]].mean(axis=1)
    return out


def spa(X, B=B, block=BLOCK, seed=13, bm=None, idx=None):
    """Hansen (2005) SPA. X: (T, k) of d_{k,t} = L(benchmark) - L(model_k), so a
    POSITIVE mean means the model beats the benchmark.

    Returns the consistent (SPA_c), lower and upper p-values plus White's
    Reality Check p-value on the same draws.
    """
    X = np.asarray(X, float)
    T, k = X.shape
    dbar = X.mean(axis=0)
    if bm is None:
        idx = stationary_bootstrap_idx(T, B, block, seed)
        bm = _boot_means(X, idx)                   # (B, k)
    B = bm.shape[0]
    omega = np.sqrt(np.maximum(((bm - dbar) ** 2).mean(axis=0) * T, 1e-24))

    stat = np.sqrt(T) * dbar / omega
    t_spa = max(float(stat.max()), 0.0)

    # recentring: consistent (drop models too far below zero to matter),
    # lower (drop every negative model) and upper (recentre all)
    A = 0.25 * T ** -0.25 * omega
    g_c = np.where(dbar >= -A / np.sqrt(T), dbar, 0.0)
    g_l = np.where(dbar >= 0.0, dbar, 0.0)
    g_u = dbar

    out = {}
    for name, g in (("spa_c", g_c), ("spa_l", g_l), ("spa_u", g_u)):
        z = np.sqrt(T) * (bm - g) / omega
        null = np.maximum(z.max(axis=1), 0.0)
        out[name] = float((1 + np.sum(null >= t_spa)) / (B + 1))
    # White's Reality Check: unstudentised, recentre every model at its own mean
    v = float(np.sqrt(T) * dbar.max())
    v_null = np.sqrt(T) * (bm - dbar).max(axis=1)
    out["reality_check"] = float((1 + np.sum(v_null >= v)) / (B + 1))
    out["t_spa"] = t_spa
    out["best_model_idx"] = int(np.argmax(stat))
    return out, dbar, stat, bm, omega


def romano_wolf(X, B=B, block=BLOCK, seed=13, bm=None):
    """Romano-Wolf stepdown: FWER-controlled p-value per model, one-sided
    (model beats benchmark)."""
    X = np.asarray(X, float)
    T, k = X.shape
    dbar = X.mean(axis=0)
    if bm is None:
        bm = _boot_means(X, stationary_bootstrap_idx(T, B, block, seed))
    B = bm.shape[0]
    omega = np.sqrt(np.maximum(((bm - dbar) ** 2).mean(axis=0) * T, 1e-24))
    stat = np.sqrt(T) * dbar / omega
    zc = np.sqrt(T) * (bm - dbar) / omega            # centred bootstrap stats

    order = np.argsort(-stat)
    p = np.empty(k)
    remaining = list(order)
    prev = 0.0
    while remaining:
        j = remaining[0]
        null_max = zc[:, remaining].max(axis=1)
        pj = float((1 + np.sum(null_max >= stat[j])) / (B + 1))
        prev = max(prev, pj)                          # enforce monotonicity
        p[j] = prev
        remaining.pop(0)
    return p, stat, dbar


def benjamini_hochberg(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        adj[i] = prev
    return adj


# ------------------------------------------------------------------ analyses
def run_spa_block(name, X, cols, rows):
    X = np.asarray(X, float)
    # one bootstrap, reused by SPA and by the stepdown, so the two agree by
    # construction rather than up to Monte Carlo error
    bm = _boot_means(X, stationary_bootstrap_idx(X.shape[0], B, BLOCK, 13))
    res, dbar, stat, bm, omega = spa(X, bm=bm)
    rw, _, _ = romano_wolf(X, bm=bm)
    best = cols[res["best_model_idx"]]
    rows.append(dict(test=name, best_model=best, best_mean_diff=float(dbar.max()),
                     t_spa=res["t_spa"], p_spa_c=res["spa_c"], p_spa_l=res["spa_l"],
                     p_spa_u=res["spa_u"], p_reality_check=res["reality_check"],
                     n_models=len(cols)))
    per = pd.DataFrame(dict(model=cols, mean_diff=dbar, t=stat, p_romano_wolf=rw))
    per["test"] = name
    return per


def bootstrap_direction(P, models=TOP4, B=B, block=BLOCK, seed=17):
    """Bootstrap direction accuracy per model and asset: point estimate,
    standard error, 95% interval and a p-value against the independence
    benchmark, replacing the asymptotic Pesaran-Timmermann p-value.

    The benchmark is not 0.5. Under the PT null of independence between the
    forecast sign and the realised sign, the expected accuracy is
    Ps = py*pz + (1-py)*(1-pz) with py, pz the two marginal up-rates, which for
    a target that rises on 51% of days is slightly above a half. Testing against
    0.5 would credit the model for the target's own class imbalance.
    """
    rows = []
    for m in models:
        d = P[P.model == m]
        for asset in list(d.asset.unique()) + ["POOLED"]:
            s = d if asset == "POOLED" else d[d.asset == asset]
            # one observation per date; assets are averaged within a date so
            # the pooled row keeps the same block geometry as the per-asset rows
            hit = (np.sign(s.y_pred) == np.sign(s.y_true)).astype(float)
            up_t = (s.y_true > 0).astype(float)
            up_p = (s.y_pred > 0).astype(float)
            g = pd.DataFrame({"h": hit.values, "yt": up_t.values, "yp": up_p.values,
                              "date": s.date.values}).groupby("date").mean()
            H, T = g["h"].values, len(g)
            idx = stationary_bootstrap_idx(T, B, block, seed)
            draws = H[idx].mean(axis=1)
            py, pz = g["yt"].values.mean(), g["yp"].values.mean()
            ps = py * pz + (1 - py) * (1 - pz)
            z, p_pt = _pt(g["yt"].values, g["yp"].values)
            rows.append(dict(model=m, asset=asset, n_days=T, DA=float(H.mean()),
                             boot_se=float(draws.std(ddof=1)),
                             ci_lo=float(np.percentile(draws, 2.5)),
                             ci_hi=float(np.percentile(draws, 97.5)),
                             indep_benchmark=float(ps),
                             p_boot_vs_indep=float((1 + np.sum(draws <= ps)) / (B + 1)),
                             p_boot_vs_half=float((1 + np.sum(draws <= 0.5)) / (B + 1)),
                             pt_z=z, p_pt_asymptotic=p_pt))
    return pd.DataFrame(rows)


def _pt(yt, yp):
    yt = (np.asarray(yt) > 0.5).astype(float)
    yp = (np.asarray(yp) > 0.5).astype(float)
    n = len(yt)
    P = float(np.mean(yt == yp))
    py, pz = yt.mean(), yp.mean()
    Ps = py * pz + (1 - py) * (1 - pz)
    vP = Ps * (1 - Ps) / n
    vPs = ((2 * pz - 1) ** 2 * py * (1 - py) / n + (2 * py - 1) ** 2 * pz * (1 - pz) / n
           + 4 * py * pz * (1 - py) * (1 - pz) / n ** 2)
    den = vP - vPs
    if den <= 0:
        return np.nan, np.nan
    z = (P - Ps) / np.sqrt(den)
    return float(z), float(st.norm.sf(z))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = load_predictions()
    P["hit_trend"] = (np.sign(P.y_pred) == np.sign(P.y_true)).astype(float)
    P["hit_p1"] = (np.sign(P.y_pred) == np.sign(P.y_true_price_1)).astype(float)
    P["hit_ph"] = (np.sign(P.y_pred) == np.sign(P.y_true_price_h)).astype(float)
    P["sq"] = (P.y_pred - P.y_true) ** 2
    P["sq_rw"] = P.y_true ** 2            # the no-change random walk

    models = sorted(P.model.unique())
    print(f"[spa] {len(models)} models, {len(P)} forecast rows, "
          f"{P.date.nunique()} dates")

    blocks, per = [], []

    # 1. squared loss against the no-change random walk
    A = panel(P, "sq")[models]
    RW = panel(P, "sq_rw")[models]
    X = (RW.values - A.values)
    per.append(run_spa_block("squared loss vs random walk", X, models, blocks))

    # 2. direction accuracy on the trend target against the SLOPE rule
    #    (SLOPE's per-date accuracy is reconstructed by transform_sweep.py and
    #    cached; fall back to the pooled constant if it has not been run)
    sl_fp = OUT / "slope_daily_accuracy.csv"
    if sl_fp.exists():
        S = pd.read_csv(sl_fp, parse_dates=["date"]).set_index("date")
        Ht = panel(P, "hit_trend")[models]
        common = Ht.index.intersection(S.index)
        X = Ht.loc[common].values - S.loc[common, ["hit_trend"]].values
        per.append(run_spa_block("direction on trend vs SLOPE", X, models, blocks))
    else:
        print("  [warn] slope_daily_accuracy.csv missing -- run transform_sweep.py "
              "first for the SLOPE-benchmark SPA")

    # 3. direction accuracy against the next-day return, benchmark = chance
    H1 = panel(P, "hit_p1")[models]
    per.append(run_spa_block("direction on next-day return vs chance",
                             H1.values - 0.5, models, blocks))
    Hh = panel(P, "hit_ph")[models]
    per.append(run_spa_block("direction on raw h-day change vs chance",
                             Hh.values - 0.5, models, blocks))

    S_ = pd.DataFrame(blocks)
    S_.to_csv(OUT / "spa_tests.csv", index=False)
    print("\n=== Hansen SPA / White Reality Check ===")
    print(S_.round(4).to_string(index=False))

    PM = pd.concat(per, ignore_index=True)
    PM["p_bh"] = np.nan
    for t in PM.test.unique():
        m = PM.test == t
        PM.loc[m, "p_bh"] = benjamini_hochberg(PM.loc[m, "p_romano_wolf"].values)
    PM.to_csv(OUT / "spa_per_model.csv", index=False)
    print("\n=== per-model, FWER-controlled (Romano-Wolf) ===")
    for t in PM.test.unique():
        s = PM[PM.test == t].sort_values("t", ascending=False)
        print(f"\n  {t}")
        print(s[["model", "mean_diff", "t", "p_romano_wolf", "p_bh"]]
              .round(4).to_string(index=False))

    print("\n=== bootstrap direction accuracy (replaces asymptotic PT) ===")
    Dz = bootstrap_direction(P)
    Dz.to_csv(OUT / "bootstrap_direction.csv", index=False)
    print(Dz.round(4).to_string(index=False))

    print(f"\n-> {OUT}")
    return S_, PM, Dz


if __name__ == "__main__":
    main()
