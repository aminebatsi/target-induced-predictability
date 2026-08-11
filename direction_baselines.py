"""Forecasters trained directly for direction, not for squared error.

The thirteen models in the main comparison are regressors: they are supervised
on the h-step change under mean squared error and their outputs are then scored
directionally. A reader may reasonably object that the study diagnoses the
directional-classification literature using systems that were never trained for
direction.

This module answers that objection with two classifiers trained on the sign of
the same target, under the identical protocol: same causal windows, same purged
and embargoed splits, same training-only standardisation, same anchors. They are
additional baselines, not replacements; none of the thirteen regressors is
refitted.

  LR-DIR    l2-penalised logistic regression on the flattened 90-vector
  XGB-DIR   gradient-boosted trees with logistic loss, same hyperparameters as
            the regression XGB except for the objective

Outputs -> results/analysis/direction_baselines.csv
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import ASSETS, FOLDS, H, L
from data import load_asset
from evaluation import fold_indices, norm_end as _norm_end
from portfolio_replay import OUT, stationary_bootstrap_idx
from targets import build_windows, kalman_causal

B = 10000
BLOCK = 20


def _standardise(X, tr):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
    return (X - mu) / sd


def fit_all():
    """One row per (model, asset, fold, anchor): prediction and the three truths."""
    rows = []
    for asset in ASSETS:
        df = load_asset(asset)
        dts = df["date"].values
        lp_f = df["logprice"].values.astype(float)
        for k, (ts, tz) in enumerate(FOLDS):
            cut = np.searchsorted(dts, np.datetime64(tz) + np.timedelta64(2, "D"))
            dates, lp = dts[:cut], lp_f[:cut]
            if len(lp) < 700:
                continue
            # the forecasting run's own convention, imported rather than copied
            ne = _norm_end(dates, ts, tz)
            if ne is None:
                continue
            trend, _ = kalman_causal(lp, ne)
            Xtab, _, anchors = build_windows(lp, trend, min_anchor=L - 1)
            idx = fold_indices(dates[anchors], ts, tz)
            if idx is None:
                continue
            tr, va, te = idx
            d = trend[anchors + H] - trend[anchors]
            y = (d > 0).astype(int)
            Xn = _standardise(Xtab, tr)
            a = anchors[te]

            fits = {"LR-DIR": LogisticRegression(max_iter=2000, C=1.0)}
            try:
                from xgboost import XGBClassifier
                fits["XGB-DIR"] = XGBClassifier(
                    n_estimators=400, max_depth=3, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                    n_jobs=-1, random_state=7, eval_metric="logloss",
                    early_stopping_rounds=30)
            except ImportError:
                pass

            for name, clf in fits.items():
                if name == "XGB-DIR":
                    clf.fit(Xn[tr], y[tr], eval_set=[(Xn[va], y[va])], verbose=False)
                else:
                    clf.fit(Xn[tr], y[tr])
                p = clf.predict_proba(Xn[te])[:, 1]
                rows.append(pd.DataFrame({
                    "model": name, "asset": asset, "fold": k,
                    "date": pd.DatetimeIndex(dates[a]),
                    "p_up": p,
                    "y_true": d[te],
                    "y_true_price_h": lp[a + H] - lp[a],
                    "y_true_price_1": lp[a + 1] - lp[a]}))
    return pd.concat(rows, ignore_index=True)


def score(P, block=BLOCK, seed=97):
    out = []
    for m in sorted(P.model.unique()):
        d = P[P.model == m]
        s = np.where(d.p_up.values >= 0.5, 1.0, -1.0)
        for obj, col in (("target", "y_true"), ("price-h", "y_true_price_h"),
                         ("price-1", "y_true_price_1")):
            yt = np.sign(d[col].values)
            hit = (s == yt).astype(float)
            py, pz = float((yt > 0).mean()), float((s > 0).mean())
            ps = py * pz + (1 - py) * (1 - pz)
            g = pd.DataFrame({"h": hit, "t": (yt > 0).astype(float),
                              "p": (s > 0).astype(float),
                              "date": d.date.values}).groupby("date").mean()
            I = stationary_bootstrap_idx(len(g), B, block, seed)
            H_ = g["h"].values[I].mean(axis=1)
            PY, PZ = g["t"].values[I].mean(axis=1), g["p"].values[I].mean(axis=1)
            diff = H_ - (PY * PZ + (1 - PY) * (1 - PZ))
            out.append(dict(model=m, object=obj, n=len(d), DA=float(hit.mean()),
                            up_rate_pred=pz, Ps=ps,
                            DA_minus_Ps=float(hit.mean() - ps),
                            lo=float(np.percentile(diff, 2.5)),
                            hi=float(np.percentile(diff, 97.5)),
                            p_two_sided=float(
                                2 * min((1 + np.sum(diff <= 0)) / (B + 1),
                                        (1 + np.sum(diff >= 0)) / (B + 1)))))
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = fit_all()
    P.to_csv(OUT / "direction_baseline_predictions.csv", index=False)
    S = score(P)
    S.to_csv(OUT / "direction_baselines.csv", index=False)
    print(f"anchors per model: {P.groupby('model').size().to_dict()}\n")
    print(S.round(4).to_string(index=False))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
