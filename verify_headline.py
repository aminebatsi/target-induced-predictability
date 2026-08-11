"""Reproduce every headline number in the manuscript from cached artefacts.

Nothing is refitted. Forecast predictions, strategy components, and the
model-free sweep are all read from disk.
"""
import numpy as np
import pandas as pd

from portfolio_replay import OUT, ROOT, SIGNS, load_components, portfolio, sharpe
from predictive_ability import MODELS, load_predictions, panel

REV = ROOT / "results" / "analysis"
CHECKS = []


def check(name, got, want, tol):
    ok = (got is not None) and abs(got - want) <= tol
    CHECKS.append((name, got, want, tol, ok))
    flag = "OK " if ok else "FAIL"
    g = "None" if got is None else f"{got:.4f}"
    print(f"  [{flag}] {name:52} got {g:>9}  want {want:.4f} +/- {tol}")


def main():
    print("== headline-number audit ==\n")

    P = load_predictions()
    n_models = P.model.nunique()
    check("13 fitted models", float(n_models), 13.0, 0.0)

    per_model_n = P.groupby("model").size()
    check("6760 OOS forecasts per model", float(per_model_n.min()), 6760.0, 0.0)
    check("  ... and the same for every model", float(per_model_n.max()), 6760.0, 0.0)

    da_trend = P.assign(h=(np.sign(P.y_pred) == np.sign(P.y_true))
                        ).groupby("model").h.mean()
    check("mean DA_trend over 13 models", float(da_trend.mean()), 0.724, 0.002)

    da_p1 = P.assign(h=(np.sign(P.y_pred) == np.sign(P.y_true_price_1))
                     ).groupby("model").h.mean()
    check("pooled DA_price-1 over 13 models", float(da_p1.mean()), 0.481, 0.002)

    T = pd.read_csv(REV / "slope_generality_sweep.csv")
    check("transform cells", float(len(T)), 5610.0, 0.0)

    ship = T[(T.smoother.str.contains("shipped")) & (T.h == 7)
             & (T.asset.isin(["TSLA", "FTSE", "AUDUSD", "COPPER", "LTC"]))]
    # anchor-weighted, matching the manuscript's pooled figure
    check("slope DA_trend, shipped filter, h=7, 5 assets",
          float((ship.da_slope * ship.n).sum() / ship.n.sum()), 0.7682, 0.004)

    from scipy import stats as st
    ok = T.dropna(subset=["rho1", "da_slope"])
    check("rho1 / slope-DA correlation",
          float(st.pearsonr(ok.rho1, ok.da_slope)[0]), 0.957, 0.002)
    res = ok.da_slope - ok.da_gaussian
    check("benchmark MAD", float(res.abs().mean()), 0.032, 0.001)
    check("benchmark mean residual", float(res.mean()), 0.019, 0.001)
    check("benchmark max |residual|", float(res.abs().max()), 0.296, 0.002)

    raw = T[T.smoother.str.startswith("none")]
    check("unsmoothed slope accuracy", float(raw.da_slope.mean()), 0.501, 0.003)

    S = pd.read_csv(REV / "spa_tests.csv")
    check("SPA vs slope (direction)",
          float(S[S.test.str.contains("vs SLOPE")].p_spa_c.iloc[0]), 0.458, 0.001)
    check("SPA vs chance on price-1",
          float(S[S.test.str.contains("next-day")].p_spa_c.iloc[0]), 1.000, 0.001)

    E = pd.read_csv(REV / "exposure_matched.csv")
    mb = E[(E.comparison.str.startswith("MATCH-B")) & (E.funding)]
    check("MATCH-B min dSharpe", float(mb.d_sharpe.min()), 1.892, 0.005)
    check("MATCH-B max dSharpe", float(mb.d_sharpe.max()), 2.634, 0.005)

    store = load_components()
    for m, want in (("TimeFilter", 1.125), ("LR", 1.065),
                    ("GRU", 0.834), ("TimeMixer", 0.763)):
        check(f"PRED Sharpe, {m} (with turn exit)",
              sharpe(portfolio(store[m], SIGNS["PRED"]).values), want, 0.002)

    n_ok = sum(c[4] for c in CHECKS)
    print(f"\n{n_ok}/{len(CHECKS)} checks pass")
    pd.DataFrame(CHECKS, columns=["check", "got", "want", "tol", "ok"]).to_csv(
        REV / "step0_audit.csv", index=False)
    return n_ok == len(CHECKS)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
