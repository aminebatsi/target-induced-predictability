"""Is there any directional association with the raw price, in either sign?

The three-truths table scores raw-price direction against 0.50. That is not an
adequate null for two reasons.

  1. Class balance. If the realised direction is up on a fraction py of days and
     a forecast calls up on a fraction pz, then under independence the expected
     agreement is Ps = py*pz + (1-py)(1-pz), which is not 0.50 unless one of the
     marginals is balanced. Testing against 0.50 credits or penalises a model
     for the marginals rather than for association.

  2. Orientation. Every next-day accuracy in the study lies BELOW 0.50. A
     one-sided test of "accuracy > 0.50" therefore cannot fail to accept the
     null, and it would also accept it if the forecasts carried strong but
     negatively oriented information. Inverting a forecast with accuracy 0.476
     gives 0.524.

This module answers the question the paper actually needs: is the forecast sign
associated with the raw-price sign at all, in either direction? It reports the
marginals, balanced accuracy, the independence benchmark Ps, the Matthews
correlation (which is signed and zero under independence), the accuracy of the
inverted forecast, and dependence-aware two-sided intervals. It also reruns the
superior predictive ability test against Ps for both orientations.

No model is refitted. Predictions are read from artifacts/.

Outputs -> results/analysis/
"""
import numpy as np
import pandas as pd
from scipy import stats as st

from portfolio_replay import OUT, ROOT, stationary_bootstrap_idx
from predictive_ability import (MODELS, PRED_DIR, _boot_means, load_predictions,
                                panel, spa)

B = 10000
BLOCK = 20
OBJECTS = {"price-h": "y_true_price_h", "price-1": "y_true_price_1",
           "target": "y_true"}


def _holm(p):
    """Holm step-down adjusted p-values, returned in original order."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * ranked)
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def _mcc(a, b):
    """Matthews correlation between two sign vectors; 0 under independence."""
    a, b = (np.asarray(a) > 0), (np.asarray(b) > 0)
    tp = np.sum(a & b); tn = np.sum(~a & ~b)
    fp = np.sum(~a & b); fn = np.sum(a & ~b)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def per_model(P, obj_col, block=BLOCK, seed=91):
    """Marginals, balanced accuracy, Ps, MCC and two-sided intervals, by model."""
    rows = []
    for m in sorted(P.model.unique()):
        d = P[P.model == m]
        yt = np.sign(d[obj_col].values)
        yp = np.sign(d.y_pred.values)
        yp[yp == 0] = 1.0
        hit = (yp == yt).astype(float)

        py = float((yt > 0).mean())          # realised up-rate
        pz = float((yp > 0).mean())          # predicted up-rate
        ps = py * pz + (1 - py) * (1 - pz)   # agreement under independence
        up = yt > 0
        bal = float(0.5 * (hit[up].mean() + hit[~up].mean()))

        # date-blocked two-sided interval for DA - Ps, recomputing Ps per draw
        g = pd.DataFrame({"h": hit, "t": (yt > 0).astype(float),
                          "p": (yp > 0).astype(float),
                          "date": d.date.values}).groupby("date").mean()
        I = stationary_bootstrap_idx(len(g), B, block, seed)
        H = g["h"].values[I].mean(axis=1)
        PY = g["t"].values[I].mean(axis=1)
        PZ = g["p"].values[I].mean(axis=1)
        PS = PY * PZ + (1 - PY) * (1 - PZ)
        diff = H - PS

        rows.append(dict(
            model=m, DA=float(hit.mean()), DA_flipped=float(1.0 - hit.mean()),
            up_rate_actual=py, up_rate_pred=pz, Ps=ps,
            balanced_acc=bal, MCC=_mcc(yp, yt),
            DA_minus_Ps=float(hit.mean() - ps),
            lo=float(np.percentile(diff, 2.5)), hi=float(np.percentile(diff, 97.5)),
            p_two_sided=float(2 * min((1 + np.sum(diff <= 0)) / (B + 1),
                                      (1 + np.sum(diff >= 0)) / (B + 1)))))
    return pd.DataFrame(rows)


def spa_both_orientations(P, obj_col, block=BLOCK):
    """SPA against the independence benchmark Ps, for the forecast as issued and
    for its exact inversion. A model carrying negatively oriented information
    would be caught by the second."""
    models = sorted(P.model.unique())
    Q = P.copy()
    yt = np.sign(Q[obj_col].values)
    yp = np.sign(Q.y_pred.values)
    yp[yp == 0] = 1.0
    Q["hit"] = (yp == yt).astype(float)
    Q["hit_flip"] = ((-yp) == yt).astype(float)

    # Ps per (model, asset), subtracted so the null is independence, not 0.50
    for col, sgn in (("hit", 1.0), ("hit_flip", -1.0)):
        key = f"exc_{col}"
        Q[key] = np.nan
        for (m, a), g in Q.groupby(["model", "asset"]):
            t = (np.sign(g[obj_col].values) > 0).mean()
            p = ((sgn * np.sign(g.y_pred.values)) > 0).mean()
            Q.loc[g.index, key] = g[col].values - (t * p + (1 - t) * (1 - p))

    out = {}
    for col in ("hit", "hit_flip"):
        X = panel(Q, f"exc_{col}")[models].values
        bm = _boot_means(X, stationary_bootstrap_idx(X.shape[0], B, block, 13))
        res, dbar, stat, _, _ = spa(X, bm=bm)
        out[col] = dict(best=models[res["best_model_idx"]],
                        advantage=float(dbar.max()), t_spa=res["t_spa"],
                        p_spa=res["spa_c"])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = load_predictions()

    print("== agreement between the exported target and a local recomputation ==")
    # consistency guard for the ExDA table: DA must be scored on ONE target
    from excess_accuracy import build_cells
    cells = build_cells()
    d = P[P.model == "GRU"]
    worst = 0.0
    for (asset, k), c in cells.items():
        s = d[(d.asset == asset) & (d.fold == k)].set_index("date")["y_true"]
        v = s.reindex(c["dates"]).values
        worst = max(worst, float(np.max(np.abs(v - c["y_true"]))))
    print(f"  max |exported y_true - recomputed| = {worst:.3e}")

    frames = {}
    for name, col in OBJECTS.items():
        print(f"\n== {name}: independence-adjusted directional association ==")
        T = per_model(P, col)
        T.insert(0, "object", name)
        frames[name] = T
        print(T[["model", "DA", "up_rate_actual", "up_rate_pred", "Ps",
                 "balanced_acc", "MCC", "DA_minus_Ps", "lo", "hi",
                 "p_two_sided"]].round(4).to_string(index=False))

    A = pd.concat(frames.values(), ignore_index=True)
    A.to_csv(OUT / "raw_price_null.csv", index=False)

    print("\n== SPA against the independence benchmark, both orientations ==")
    rows = []
    for name, col in OBJECTS.items():
        r = spa_both_orientations(P, col)
        for orient, key in (("as issued", "hit"), ("inverted", "hit_flip")):
            rows.append(dict(object=name, orientation=orient, **r[key]))
    S = pd.DataFrame(rows)
    S["p_holm_raw_family"] = np.nan
    raw = S.object.isin(["price-h", "price-1"])
    S.loc[raw, "p_holm_raw_family"] = _holm(S.loc[raw, "p_spa"].values)
    S.to_csv(OUT / "raw_price_null_spa.csv", index=False)
    print(S.round(4).to_string(index=False))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
