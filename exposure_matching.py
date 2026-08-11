"""Exposure-matched signal reversal, cost sensitivity, and selection-corrected
strategy inference. No model is refitted: every arm is a different function of
the cached execution-stack components.

WHY THE OBVIOUS MATCHING DOES NOT WORK. The published PRED/FLIP comparison is
confounded because PRED carries about twice FLIP's gross exposure. The reflex
fix -- rescale FLIP by the gross-exposure ratio -- is a no-op. The book is not
levered against a cash leg, so multiplying every position by a constant k
multiplies the return, the fee and the funding charge by the same k, and the
Sharpe ratio is exactly invariant. Matching MEAN exposure therefore cannot
change the answer; only matching the exposure PATH can.

The gate is what breaks the path. Under the SMA-200 gate, PRED and FLIP are in
the market on complementary days: in a bull regime PRED is long exactly when
FLIP is flat. No rescaling can align two schedules that are disjoint. So the
matched tests here remove or bypass the gate rather than reweighting it.

  MATCH-A  reversal of the shipped exposure path.  pos -> -pos, gate and all.
           Gross and turnover identical by construction, net exactly opposite.
           Answers: does the direction the shipped book actually took pay?
           It does NOT isolate the forecast, because the gate co-determines
           that direction.

  MATCH-B  ungated forecast reversal.  Drop the gate from BOTH arms, keep the
           turn exit and the vol sizing. FLIP's position is then exactly minus
           PRED's on every day: identical gross, identical turnover, opposite
           net. The forecast sign is now the ONLY thing setting direction, so
           this is the test the exposure objection asks for.

Funding is the one asymmetry no position matching can remove: short crypto pays
10%/yr and long crypto does not, so a reversed book pays funding on different
days. Every matched comparison is therefore reported twice, with funding and
with funding switched off in both arms.

Outputs -> results/analysis/
"""
import numpy as np
import pandas as pd
from scipy import stats as st

from portfolio_replay import (OUT, SIGNS, book_scale, exposures, load_components,
                             max_drawdown, paired_tests, portfolio, price_positions,
                             raw_positions, sharpe)

B_BOOT = 10000
COSTS_BPS = (0.0, 5.0, 10.0, 20.0)
FUNDS = (0.05, 0.10, 0.20)
N_MAXSHARPE = 2000
EULER = 0.5772156649015329


# ============================================================ exposure matching
def matched_battery(store):
    rows, boots = [], []
    for model, comps in store.items():
        # One overlay path per comparison, shared by both arms (see docstring).
        for tag, kw_pred, kw_flip in (
                ("gated (published)",
                 dict(gated=True), dict(gated=True)),
                ("MATCH-A: shipped path reversed",
                 dict(gated=True), dict(gated=True, reverse=True)),
                ("MATCH-B: ungated forecast reversal",
                 dict(gated=False), dict(gated=False))):
            for funding in (True, False):
                pred_sign = SIGNS["PRED"]
                flip_sign = SIGNS["PRED"] if "reverse" in kw_flip else SIGNS["FLIP"]

                pm = raw_positions(comps, pred_sign, **kw_pred)
                ks = book_scale(price_positions(comps, pm))
                p = portfolio(comps, pred_sign, funding=funding,
                              shared_scale=ks, **kw_pred)
                f = portfolio(comps, flip_sign, funding=funding,
                              shared_scale=ks, **kw_flip)
                ep = exposures(comps, pred_sign, shared_scale=ks, **kw_pred)
                ef = exposures(comps, flip_sign, shared_scale=ks, **kw_flip)

                al = pd.concat({"f": f, "p": p}, axis=1).dropna()
                t = paired_tests(al["f"].values, al["p"].values, B=B_BOOT)
                rows.append(dict(model=model, comparison=tag, funding=funding,
                                 sharpe_pred=sharpe(p.values), sharpe_flip=sharpe(f.values),
                                 dd_pred=max_drawdown(p.values), dd_flip=max_drawdown(f.values),
                                 gross_pred=ep["gross"], gross_flip=ef["gross"],
                                 net_pred=ep["net"], net_flip=ef["net"],
                                 turn_pred=ep["turnover"], turn_flip=ef["turnover"],
                                 **t))
                if funding:
                    boots.append((model, tag, al))
    return pd.DataFrame(rows), boots


# ============================================================ drift neutralisation
def drift_neutral(store):
    """Net exposure is the other half of the objection: PRED runs +0.16 mean net
    against FLIP's +0.03, and the sample has positive drift. Regressing each arm
    on the model-free LONG book and comparing the residual Sharpes prices the
    direction after the common long exposure has been taken out."""
    rows = []
    for model, comps in store.items():
        long = portfolio(comps, SIGNS["LONG"])
        for arm in ("PRED", "FLIP"):
            s = portfolio(comps, SIGNS[arm])
            al = pd.concat({"l": long, "s": s}, axis=1).dropna()
            beta = np.polyfit(al["l"].values, al["s"].values, 1)[0]
            rows.append(dict(model=model, arm=arm, beta_vs_long=float(beta),
                             sharpe=sharpe(al["s"].values),
                             sharpe_resid=sharpe(al["s"].values - beta * al["l"].values)))
    R = pd.DataFrame(rows).pivot(index="model", columns="arm")
    return R


# ============================================================ cost sensitivity
def cost_sensitivity(store):
    rows = []
    for model, comps in store.items():
        for arm, blend in (("PRED", False), ("BLEND", True)):
            for cbps in COSTS_BPS:
                s = portfolio(comps, SIGNS[arm.replace("BLEND", "PRED")],
                              cost=cbps * 1e-4, blend=blend)
                rows.append(dict(model=model, arm=arm, cost_bps=cbps, funding=0.10,
                                 sharpe=sharpe(s.values),
                                 ann_ret=float(np.mean(s.values) * 365)))
    return pd.DataFrame(rows)


def funding_sensitivity(store):
    """Funding enters only through the crypto legs' SHORT exposure, so it is
    swept by rescaling that charge on the cached components."""
    rows = []
    for model, comps in store.items():
        for fund in FUNDS:
            scaled = {k: dict(c, fund=(fund if c["fund"] > 0 else 0.0))
                      for k, c in comps.items()}
            for arm, blend in (("PRED", False), ("BLEND", True)):
                s = portfolio(scaled, SIGNS["PRED"], blend=blend)
                rows.append(dict(model=model, arm=arm, cost_bps=10.0, funding=fund,
                                 sharpe=sharpe(s.values)))
    return pd.DataFrame(rows)


# ============================================================ selection correction
def deflated_sharpe(returns, n_trials, var_trials, ann=365):
    """Bailey and Lopez de Prado (2014) deflated Sharpe ratio.

    Works in per-period (daily) units throughout; only the reported SR is
    annualised. `var_trials` is the cross-sectional variance of the trial
    Sharpes in the SAME per-period units.
    """
    x = np.asarray(returns, float)
    T = len(x)
    sr = float(np.mean(x) / np.std(x, ddof=1))
    g3 = float(st.skew(x))
    g4 = float(st.kurtosis(x, fisher=False))
    n = max(int(n_trials), 2)
    # expected maximum Sharpe of n independent trials with no skill
    sr0 = np.sqrt(var_trials) * ((1 - EULER) * st.norm.ppf(1 - 1.0 / n)
                                 + EULER * st.norm.ppf(1 - 1.0 / (n * np.e)))
    den = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2, 1e-12))
    z = (sr - sr0) * np.sqrt(T - 1) / den
    return dict(T=T, sharpe_ann=sr * np.sqrt(ann), sr_daily=sr, skew=g3, kurtosis=g4,
                n_trials=n, sr0_daily=float(sr0), sr0_ann=float(sr0 * np.sqrt(ann)),
                psr_z=float(z), dsr=float(st.norm.cdf(z)))


def max_sharpe_null(store, n_draws=N_MAXSHARPE, seed=11):
    """Bootstrap the maximum Sharpe attainable across the four selected models
    under the null of no directional information.

    The sign draws are synchronised across models -- the same random signs are
    fed to every model on the same (fold, asset) cell -- so the four null arms
    are as correlated as the four real arms are, which is what makes their
    maximum the right reference for a maximum that was itself selected.
    """
    models = list(store)
    keys = sorted(store[models[0]], key=lambda k: (k[0], k[1]))
    lens = {k: len(store[models[0]][k]["w"]) for k in keys}
    # id(component dict) -> cell key, so one drawn sign vector can be handed to
    # every model's copy of the same (fold, asset) cell.
    where = {m: {id(c): k for k, c in store[m].items()} for m in models}
    rng = np.random.default_rng(seed)
    per_model, maxes = {m: [] for m in models}, []
    for _ in range(n_draws):
        draw = {k: rng.choice([-1.0, 1.0], size=lens[k]) for k in keys}
        sh = []
        for m in models:
            fn = (lambda c, _w=where[m]: draw[_w[id(c)]])
            sh.append(sharpe(portfolio(store[m], fn).values))
        for m, v in zip(models, sh):
            per_model[m].append(v)
        maxes.append(max(sh))
    return {m: np.array(v) for m, v in per_model.items()}, np.array(maxes)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    store = load_components()

    print("\n=== exposure-matched reversal ===")
    M, boots = matched_battery(store)
    M.to_csv(OUT / "exposure_matched.csv", index=False)
    show = ["model", "comparison", "funding", "sharpe_pred", "sharpe_flip",
            "d_sharpe", "ci_lo", "ci_hi", "p_sharpe", "gross_pred", "gross_flip",
            "net_pred", "net_flip", "turn_pred", "turn_flip"]
    print(M[show].round(4).to_string(index=False))

    print("\n=== drift-neutralised (residual vs the LONG book) ===")
    D = drift_neutral(store)
    D.to_csv(OUT / "drift_neutral.csv")
    print(D.round(3).to_string())

    print("\n=== cost sensitivity ===")
    C = pd.concat([cost_sensitivity(store), funding_sensitivity(store)],
                  ignore_index=True)
    C.to_csv(OUT / "cost_sensitivity.csv", index=False)
    print(C.pivot_table(index=["arm", "model"], columns=["cost_bps", "funding"],
                        values="sharpe").round(3).to_string())

    print("\n=== maximum-Sharpe null across the four selected models ===")
    per_model, maxes = max_sharpe_null(store)
    obs = {m: sharpe(portfolio(store[m], SIGNS["PRED"]).values) for m in store}
    best = max(obs, key=obs.get)
    p_max = float((1 + np.sum(maxes >= obs[best])) / (len(maxes) + 1))
    rows = [dict(model=m, sharpe=obs[m],
                 null_mean=float(per_model[m].mean()),
                 null_p975=float(np.percentile(per_model[m], 97.5)),
                 p_single=float((1 + np.sum(per_model[m] >= obs[m])) / (len(maxes) + 1)))
            for m in store]
    N = pd.DataFrame(rows)
    N["p_family"] = [float((1 + np.sum(maxes >= v)) / (len(maxes) + 1)) for v in N.sharpe]
    N.to_csv(OUT / "max_sharpe_null.csv", index=False)
    print(N.round(4).to_string(index=False))
    print(f"  selected best = {best} at {obs[best]:.3f}; family-wise "
          f"max-Sharpe null 95th pct {np.percentile(maxes, 95):.3f}, p = {p_max:.4f}")

    print("\n=== deflated Sharpe ===")
    daily = {m: portfolio(store[m], SIGNS["PRED"]) for m in store}
    sr_daily = np.array([float(np.mean(v.values) / np.std(v.values, ddof=1))
                         for v in daily.values()])
    var_obs = float(np.var(sr_daily, ddof=1))
    var_null = float(np.var(np.concatenate([per_model[m] for m in store]) / np.sqrt(365),
                            ddof=1))
    drows = []
    for label, var in (("observed spread across the 4 carried models", var_obs),
                       ("random-sign null across the same stack", var_null)):
        for n in (4, 13, 52):
            d = deflated_sharpe(daily[best].values, n, var)
            drows.append(dict(variance_source=label, **d))
    D2 = pd.DataFrame(drows)
    D2.to_csv(OUT / "deflated_sharpe.csv", index=False)
    print(D2.round(4).to_string(index=False))

    print(f"\n-> {OUT}")
    return M, C, N, D2


if __name__ == "__main__":
    main()
