"""Can the turn classifier be removed?

The classifier's meta-label is built from the first-stage model's FITTED sign on
the training rows, not from cross-fitted out-of-sample errors. Rather than
retrain the first stage -- which is forbidden here and expensive anyway -- we
test whether the component is needed at all.

PRE-SPECIFIED SURVIVAL CRITERION, fixed before the numbers were seen:

  A. the no-turn PRED book retains a positive pooled Sharpe ratio;
  B. under MATCH-B, all four carried models have
     dSharpe = Sharpe(PRED) - Sharpe(FLIP) > 0;
  C. the paired stationary-bootstrap 95% interval for the MATCH-B dSharpe lies
     entirely above zero for all four carried models.

These preserve the qualitative claim the ablation exists to support -- that the
forecast sign is not inert under exposure-matched reversal -- and nothing more.
Absolute Sharpe ratios are NOT required to match. No parameter is optimised, and
the turn exit is deleted rather than replaced.

Outputs -> results/analysis/
"""
import numpy as np
import pandas as pd

from portfolio_replay import (ROOT, SIGNS, book_scale, exposures, load_components,
                             max_drawdown, paired_tests, portfolio,
                             price_positions, raw_positions, sharpe)

OUT = ROOT / "results" / "analysis"
B_BOOT = 10000
FOLDS_Y = [2021, 2022, 2023, 2024, 2025]
FOLD_SPANS = [(f"{y}-07-01", f"{y + 1}-07-01") for y in FOLDS_Y]
ARMS = ("PRED", "BLEND", "LONG", "REGIME", "FLIP")


def _by_fold(s):
    out = {}
    for y, (ts, tz) in zip(FOLDS_Y, FOLD_SPANS):
        m = (s.index >= ts) & (s.index < tz)
        out[f"fold_{y}"] = sharpe(s[m].values)
    return out


def arm_series(comps, arm, turn_exit):
    blend = arm == "BLEND"
    key = "PRED" if blend else arm
    return portfolio(comps, SIGNS[key], blend=blend, turn_exit=turn_exit)


def levels(store, turn_exit):
    """Sharpe, drawdown, annualised return and volatility for every arm."""
    rows = []
    for model, comps in store.items():
        for arm in ARMS:
            s = arm_series(comps, arm, turn_exit)
            x = s.values
            rows.append(dict(turn_exit=turn_exit, model=model, arm=arm,
                             sharpe=sharpe(x), maxdd=max_drawdown(x),
                             ann_ret=float(np.mean(x) * 365),
                             ann_vol=float(np.std(x) * np.sqrt(365)),
                             **_by_fold(s)))
    return pd.DataFrame(rows)


def matched(store, turn_exit):
    """The three reversal comparisons, both arms under one shared overlay."""
    rows = []
    for model, comps in store.items():
        for tag, kwp, kwf in (
                ("gated (published)", dict(gated=True), dict(gated=True)),
                ("MATCH-A: shipped path reversed",
                 dict(gated=True), dict(gated=True, reverse=True)),
                ("MATCH-B: ungated forecast reversal",
                 dict(gated=False), dict(gated=False))):
            for funding in (True, False):
                fs = SIGNS["PRED"] if "reverse" in kwf else SIGNS["FLIP"]
                pm = raw_positions(comps, SIGNS["PRED"], turn_exit=turn_exit, **kwp)
                ks = book_scale(price_positions(comps, pm))
                p = portfolio(comps, SIGNS["PRED"], funding=funding,
                              shared_scale=ks, turn_exit=turn_exit, **kwp)
                f = portfolio(comps, fs, funding=funding, shared_scale=ks,
                              turn_exit=turn_exit, **kwf)
                ep = exposures(comps, SIGNS["PRED"], shared_scale=ks,
                               turn_exit=turn_exit, **kwp)
                ef = exposures(comps, fs, shared_scale=ks, turn_exit=turn_exit, **kwf)
                al = pd.concat({"f": f, "p": p}, axis=1).dropna()
                t = paired_tests(al["f"].values, al["p"].values, B=B_BOOT)
                rows.append(dict(turn_exit=turn_exit, model=model, comparison=tag,
                                 funding=funding,
                                 sharpe_pred=sharpe(p.values),
                                 sharpe_flip=sharpe(f.values),
                                 gross_pred=ep["gross"], gross_flip=ef["gross"],
                                 net_pred=ep["net"], net_flip=ef["net"],
                                 turn_pred=ep["turnover"], turn_flip=ef["turnover"],
                                 **t))
    return pd.DataFrame(rows)


def evaluate(L, M):
    """Apply the pre-specified criterion. Returns (passed, report rows)."""
    rep = []
    no = L[(~L.turn_exit)]
    mb = M[(~M.turn_exit) & (M.comparison.str.startswith("MATCH-B")) & (M.funding)]

    pred = no[no.arm == "PRED"].set_index("model")["sharpe"]
    a_ok = bool((pred > 0).all())
    rep.append(("A: no-turn PRED pooled Sharpe > 0",
                f"min {pred.min():+.3f} ({pred.idxmin()})", a_ok))

    b_ok = bool((mb.d_sharpe > 0).all())
    rep.append(("B: MATCH-B dSharpe > 0 for all four",
                f"min {mb.d_sharpe.min():+.3f} ({mb.loc[mb.d_sharpe.idxmin(), 'model']})",
                b_ok))

    c_ok = bool((mb.ci_lo > 0).all())
    rep.append(("C: MATCH-B 95% CI entirely above zero for all four",
                f"min lower bound {mb.ci_lo.min():+.3f} "
                f"({mb.loc[mb.ci_lo.idxmin(), 'model']})", c_ok))

    return all(r[2] for r in rep), rep


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    store = load_components()

    print("PRE-SPECIFIED CRITERION (fixed before the numbers were seen):")
    print("  A. no-turn PRED pooled Sharpe > 0")
    print("  B. MATCH-B dSharpe > 0 for all four carried models")
    print("  C. MATCH-B 95% bootstrap interval above zero for all four\n")

    L = pd.concat([levels(store, True), levels(store, False)], ignore_index=True)
    L.to_csv(OUT / "arm_levels_turn_vs_noturn.csv", index=False)
    piv = L.pivot_table(index=["arm", "model"], columns="turn_exit", values="sharpe")
    piv.columns = ["no turn exit", "with turn exit"]
    piv["delta"] = piv["no turn exit"] - piv["with turn exit"]
    print("Pooled Sharpe by arm:\n", piv.round(3).to_string(), "\n")

    M = pd.concat([matched(store, True), matched(store, False)], ignore_index=True)
    M.to_csv(OUT / "matched_turn_vs_noturn.csv", index=False)
    show = ["turn_exit", "model", "comparison", "sharpe_pred", "sharpe_flip",
            "d_sharpe", "ci_lo", "ci_hi", "p_sharpe"]
    print("Matched reversal (funded):\n",
          M[M.funding][show].round(4).to_string(index=False), "\n")

    passed, rep = evaluate(L, M)
    print("CRITERION:")
    for name, detail, ok in rep:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:52} {detail}")
    print(f"\n==> The turn classifier {'CAN' if passed else 'CANNOT'} be removed "
          f"under the pre-specified criterion.")
    pd.DataFrame(rep, columns=["criterion", "detail", "passed"]).to_csv(
        OUT / "survival_criterion.csv", index=False)
    return passed


if __name__ == "__main__":
    main()
