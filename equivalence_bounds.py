"""Bounded null statements, so that "not significant" is not read as "absent".

Two of this study's central results are failures to reject:

  (a) no fitted model shows a same-signed association with the raw h-day price
      change (Table: raw_price_null, price-h);
  (b) no fitted model out-predicts the parameter-free rule on the transformed
      target (Table: spa).

A failure to reject is not evidence of absence, and neither claim should be
presented as one. This module converts both into statements with an explicit
upper bound: rather than "we found nothing", "the data exclude an effect larger
than x". Three quantities are produced.

  Equivalence bound   The upper end of the two-sided 95% interval. Any effect
                      above it is inconsistent with the data at that level.
  MDE                 Minimum detectable effect: the true effect a two-sided
                      5% test would reject the null for with 80% probability,
                      1.96+0.84 = 2.80 bootstrap standard errors.
  Non-overlap check   For the h-day object, the same test on a subsample taken
                      every h-th anchor, so no two labels share an observation.
                      This trades sample size for independence and shows the
                      overlap correction is not doing the work.

Outputs -> results/analysis/equivalence_bounds.csv
           results/analysis/nonoverlap_price_h.csv
"""
import numpy as np
import pandas as pd

from portfolio_replay import OUT, stationary_bootstrap_idx
from predictive_ability import MODELS, load_predictions

B = 10000
BLOCK = 20
H = 7
Z_MDE = 1.959964 + 0.8416212        # two-sided 5% test, 80% power


def _date_panel(d, hit, extra=None):
    """Collapse to one row per date, averaging assets trading that day."""
    cols = {"h": hit, "date": d.date.values}
    if extra:
        cols.update(extra)
    return pd.DataFrame(cols).groupby("date").mean()


def _boot(g, cols, block=BLOCK, seed=91):
    I = stationary_bootstrap_idx(len(g), B, block, seed)
    return {c: g[c].values[I].mean(axis=1) for c in cols}


def price_object_bounds(P, obj_col, seed=91):
    """Per model: DA - Ps with a two-sided interval, its upper bound, and MDE."""
    rows = []
    for m in sorted(P.model.unique()):
        d = P[P.model == m]
        yt = np.sign(d[obj_col].values)
        yp = np.sign(d.y_pred.values)
        yp[yp == 0] = 1.0
        hit = (yp == yt).astype(float)
        g = _date_panel(d, hit, {"t": (yt > 0).astype(float),
                                 "p": (yp > 0).astype(float)})
        b = _boot(g, ["h", "t", "p"], seed=seed)
        diff = b["h"] - (b["t"] * b["p"] + (1 - b["t"]) * (1 - b["p"]))
        se = float(diff.std(ddof=1))
        py, pz = float((yt > 0).mean()), float((yp > 0).mean())
        point = float(hit.mean() - (py * pz + (1 - py) * (1 - pz)))
        rows.append(dict(
            model=m, n_obs=len(d), n_dates=len(g), point=point, se=se,
            lo=float(np.percentile(diff, 2.5)),
            hi=float(np.percentile(diff, 97.5)),
            mde=Z_MDE * se))
    return pd.DataFrame(rows)


def slope_advantage_bounds(P, seed=77):
    """Per model: (model - SLOPE) directional accuracy on the transformed
    target, with a two-sided interval and MDE.

    The slope forecast is h*(level_t - level_{t-1}); the exported panel carries
    the realised change but not the level, so the rule's per-date hit rate is
    read from the cached model-free sweep, which scores it on the same anchors.
    """
    S = pd.read_csv(OUT / "slope_daily_accuracy.csv", parse_dates=["date"])
    S = S.set_index("date")["hit_trend"]
    rows = []
    for m in sorted(P.model.unique()):
        d = P[P.model == m]
        yp = np.sign(d.y_pred.values)
        yp[yp == 0] = 1.0
        hit = (yp == np.sign(d.y_true.values)).astype(float)
        g = _date_panel(d, hit)
        common = g.index.intersection(S.index)
        adv = g.loc[common, "h"].values - S.loc[common].values
        A = pd.DataFrame({"d": adv}, index=common)
        b = _boot(A, ["d"], seed=seed)
        rows.append(dict(
            model=m, n_dates=len(common), point=float(adv.mean()),
            se=float(b["d"].std(ddof=1)),
            lo=float(np.percentile(b["d"], 2.5)),
            hi=float(np.percentile(b["d"], 97.5)),
            mde=Z_MDE * float(b["d"].std(ddof=1))))
    return pd.DataFrame(rows)


def nonoverlapping_price_h(P, h=H, seed=53):
    """Repeat the price-h test on anchors spaced h apart, so that no two labels
    share an observation. Reported for every offset 0..h-1 so the result cannot
    depend on where the thinning starts."""
    rows = []
    for m in sorted(P.model.unique()):
        d = P[P.model == m]
        for off in range(h):
            keep = []
            for (_, _), gidx in d.groupby(["asset", "fold"]).groups.items():
                sub = d.loc[gidx].sort_values("date")
                keep.append(sub.iloc[off::h])
            s = pd.concat(keep)
            yt = np.sign(s.y_true_price_h.values)
            yp = np.sign(s.y_pred.values)
            yp[yp == 0] = 1.0
            hit = (yp == yt).astype(float)
            py, pz = float((yt > 0).mean()), float((yp > 0).mean())
            ps = py * pz + (1 - py) * (1 - pz)
            g = _date_panel(s, hit, {"t": (yt > 0).astype(float),
                                     "p": (yp > 0).astype(float)})
            b = _boot(g, ["h", "t", "p"], seed=seed + off)
            diff = b["h"] - (b["t"] * b["p"] + (1 - b["t"]) * (1 - b["p"]))
            rows.append(dict(
                model=m, offset=off, n_obs=len(s), n_dates=len(g),
                DA=float(hit.mean()), Ps=ps,
                point=float(hit.mean() - ps),
                lo=float(np.percentile(diff, 2.5)),
                hi=float(np.percentile(diff, 97.5)),
                p_two_sided=float(2 * min((1 + np.sum(diff <= 0)) / (B + 1),
                                          (1 + np.sum(diff >= 0)) / (B + 1)))))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = load_predictions()

    out = []
    for obj, col in (("price-h", "y_true_price_h"), ("price-1", "y_true_price_1")):
        E = price_object_bounds(P, col)
        E.insert(0, "quantity", f"DA - Ps, {obj}")
        out.append(E)
    A = slope_advantage_bounds(P)
    A.insert(0, "quantity", "DA(model) - DA(slope), target")
    out.append(A)
    E = pd.concat(out, ignore_index=True)
    E.to_csv(OUT / "equivalence_bounds.csv", index=False)

    pd.set_option("display.width", 200)
    for q in E.quantity.unique():
        g = E[E.quantity == q]
        print(f"\n== {q} ==")
        print(g[["model", "point", "lo", "hi", "se", "mde"]]
              .round(4).to_string(index=False))
        print(f"  worst-case upper bound over the thirteen models: "
              f"{g.hi.max():+.4f}")
        print(f"  largest MDE (least powerful model):              {g.mde.max():.4f}")
        print(f"  median MDE:                                     "
              f"{g.mde.median():.4f}")

    N = nonoverlapping_price_h(P)
    N.to_csv(OUT / "nonoverlap_price_h.csv", index=False)
    print("\n== price-h on non-overlapping anchors (every 7th) ==")
    agg = N.groupby("model").agg(n_obs=("n_obs", "mean"),
                                 point=("point", "mean"),
                                 lo=("lo", "mean"), hi=("hi", "mean"),
                                 min_p=("p_two_sided", "min"),
                                 sig=("p_two_sided", lambda x: int((x < .05).sum())))
    print(agg.round(4).to_string())
    print(f"\n  model x offset cells significant at 5%: "
          f"{int((N.p_two_sided < 0.05).sum())} of {len(N)}")
    print(f"  mean point estimate: {N.point.mean():+.4f} "
          f"(overlapping full sample gives the value in raw_price_null.csv)")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
