"""Why can a near-chance next-day hit rate coexist with a large
exposure-matched reversal difference?

Directional accuracy weights every observation equally. A Sharpe ratio does
not: it weights by realised return, and the execution stack further reweights
by volatility sizing and by which days it is in the market at all. The two
statistics can therefore disagree without contradiction -- but that has to be
measured, not asserted. This module measures it.

Everything is computed on the MATCH-B arm, which is the exposure-matched
comparison the paper's directional claim rests on, and on the retained stack
(the turn exit stays; see turn_exit_ablation.py for why).

No model is refitted.

Outputs -> results/analysis/
"""
import numpy as np
import pandas as pd

from portfolio_replay import (ROOT, SIGNS, load_components, nz_sign, portfolio,
                             raw_positions, sharpe, stationary_bootstrap_idx)

OUT = ROOT / "results" / "analysis"
B_BOOT = 10000
BLOCK = 20


# ------------------------------------------------------------------ mechanism
def mechanism(store, block=BLOCK, B=B_BOOT):
    """Per-model day-level decomposition of the MATCH-B forecast arm."""
    rows = []
    for model, comps in store.items():
        pos_map = raw_positions(comps, SIGNS["PRED"], gated=False)
        s_all, r_all, p_all = [], [], []
        for key, c in comps.items():
            s_all.append(nz_sign(c["w"]))
            r_all.append(c["r"])
            p_all.append(pos_map[key])
        s = np.concatenate(s_all)
        r = np.concatenate(r_all)
        p = np.concatenate(p_all)

        active = p != 0.0
        correct = s == np.sign(r)
        pnl = p * r

        rows.append(dict(
            model=model,
            n_asset_days=int(len(s)),
            da_all=float(correct.mean()),
            da_active=float(correct[active].mean()),
            active_frac=float(active.mean()),
            absret_correct=float(np.abs(r[correct]).mean()),
            absret_wrong=float(np.abs(r[~correct]).mean()),
            med_absret_correct=float(np.median(np.abs(r[correct]))),
            med_absret_wrong=float(np.median(np.abs(r[~correct]))),
            absret_ratio=float(np.abs(r[correct]).mean() / np.abs(r[~correct]).mean()),
            signed_ret_bps=float(np.mean(s * r) * 1e4),
            signed_ret_active_bps=float(np.mean((s * r)[active]) * 1e4),
            posweighted_ret_bps=float(np.mean(p * r) * 1e4),
            # The signed return decomposes exactly into a frequency term and a
            # magnitude term, which is the whole point of the diagnostic:
            #   E[s r] = P(correct) E[|r| | correct] - P(wrong) E[|r| | wrong].
            freq_term_bps=float(correct.mean() * np.abs(r[correct]).mean() * 1e4),
            magn_term_bps=float((~correct).mean() * np.abs(r[~correct]).mean() * 1e4),
            # counterfactual: what the signed return would be if correct and
            # wrong days carried the SAME mean absolute move
            signed_ret_equalmag_bps=float(
                (2 * correct.mean() - 1) * np.abs(r).mean() * 1e4),
            pnl_gross_correct=float(pnl[correct].sum()),
            pnl_gross_wrong=float(pnl[~correct].sum()),
            pnl_gross_total=float(pnl.sum())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ residuals
def _boot_reg(y, x, idx):
    """Bootstrap alpha, beta and residual Sharpe over resampled date blocks."""
    a, b, rs = [], [], []
    for row in idx:
        yy, xx = y[row], x[row]
        v = xx.var()
        beta = float(((xx - xx.mean()) * (yy - yy.mean())).mean() / v) if v > 0 else 0.0
        alpha = float(yy.mean() - beta * xx.mean())
        res = yy - beta * xx
        a.append(alpha)
        b.append(beta)
        rs.append(sharpe(res))
    return np.array(a), np.array(b), np.array(rs)


def long_residual(store, block=BLOCK, B=B_BOOT, seed=23):
    """Regress each arm's daily book return on the model-free LONG book and
    price the residual. Uncertainty comes from the same stationary bootstrap
    used everywhere else in the paper."""
    rows = []
    for model, comps in store.items():
        lng = portfolio(comps, SIGNS["LONG"])
        for arm, kw in (("PRED", dict(gated=False)),
                        ("FLIP", dict(gated=False))):
            fn = SIGNS["PRED"] if arm == "PRED" else SIGNS["FLIP"]
            s = portfolio(comps, fn, **kw)
            al = pd.concat({"l": lng, "s": s}, axis=1).dropna()
            y, x = al["s"].values, al["l"].values
            v = x.var()
            beta = float(((x - x.mean()) * (y - y.mean())).mean() / v)
            alpha = float(y.mean() - beta * x.mean())
            res = y - beta * x
            idx = stationary_bootstrap_idx(len(y), B, block, seed)
            ba, bb, brs = _boot_reg(y, x, idx)
            rows.append(dict(
                model=model, arm=arm, sharpe=sharpe(y), beta=beta,
                beta_lo=float(np.percentile(bb, 2.5)),
                beta_hi=float(np.percentile(bb, 97.5)),
                alpha_ann=alpha * 365,
                alpha_lo=float(np.percentile(ba, 2.5) * 365),
                alpha_hi=float(np.percentile(ba, 97.5) * 365),
                sharpe_resid=sharpe(res),
                resid_lo=float(np.percentile(brs, 2.5)),
                resid_hi=float(np.percentile(brs, 97.5)),
                block=block))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    store = load_components()

    print("== MATCH-B day-level mechanism ==")
    M = mechanism(store)
    M.to_csv(OUT / "mechanism_matchb.csv", index=False)
    cols = ["model", "da_all", "da_active", "active_frac", "absret_correct",
            "absret_wrong", "absret_ratio", "signed_ret_bps",
            "signed_ret_equalmag_bps", "posweighted_ret_bps",
            "pnl_gross_correct", "pnl_gross_wrong", "pnl_gross_total"]
    print(M[cols].round(4).to_string(index=False))

    print("\n== residual against the model-free LONG book (ungated arms) ==")
    R = long_residual(store)
    R.to_csv(OUT / "long_residual.csv", index=False)
    print(R[["model", "arm", "sharpe", "beta", "beta_lo", "beta_hi",
             "alpha_ann", "alpha_lo", "alpha_hi",
             "sharpe_resid", "resid_lo", "resid_hi"]].round(3).to_string(index=False))

    print(f"\n-> {OUT}")
    return M, R


if __name__ == "__main__":
    main()
