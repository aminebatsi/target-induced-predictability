"""Audit of the exposure-matched reversal: does the position symmetry the paper
claims actually hold, and if so why are the net Sharpe ratios not exact
negatives?

The manuscript states that under MATCH-A and MATCH-B the PRED and FLIP position
paths are exact negatives, yet Table 17 reports net Sharpe ratios that are not
negatives of each other. That is consistent, but only if the frictions are
applied after the reversal, and it has to be verified rather than asserted.

The identity under test. Positions are exact negatives, so the gross return
negates exactly. The turnover cost does NOT negate: it is charged on
|delta pos|, and |delta(-x)| = |delta x|, so the SAME cost series is subtracted
from both arms. Writing G for the PRED gross return and C >= 0 for the common
cost,

    R_pred = G - C,      R_flip = -G - C = -R_pred - 2C.

Funding adds a second, genuinely asymmetric term, because only short exposure
pays it: F_pred = phi * max(-pos, 0) and F_flip = phi * max(pos, 0), whose sum
is phi * |pos|.

Nothing is refitted and no strategy parameter is touched. Every arm is
recomputed from the cached execution-stack components.

Outputs -> results/analysis/
"""
import numpy as np
import pandas as pd

from portfolio_replay import (ROOT, SIGNS, aggregate, book_scale, load_components,
                             price_positions, raw_positions, sharpe,
                             stationary_bootstrap_idx)

OUT = ROOT / "results" / "analysis"
B_BOOT = 10000
BLOCK = 20
COST = 1e-3
FUND = 0.10
TOL_POS = 1e-12
TOL_RET = 1e-12
TOL_SR = 1e-9
MATCHES = {"MATCH-A": dict(gated=True), "MATCH-B": dict(gated=False)}
FAILURES = []


def check(name, value, tol):
    ok = value < tol
    if not ok:
        FAILURES.append(f"{name} = {value:.3e} (tol {tol:.0e})")
    print(f"  [{'OK ' if ok else 'FAIL'}] {name:52} {value:.3e}")
    return ok


# ---------------------------------------------------------------- arms
def arms(comps, match):
    """Per-cell PRED and FLIP position maps for one matched comparison."""
    kw = MATCHES[match]
    pm = raw_positions(comps, SIGNS["PRED"], **kw)
    if match == "MATCH-A":
        fm = raw_positions(comps, SIGNS["PRED"], reverse=True, **kw)
    else:
        fm = raw_positions(comps, SIGNS["FLIP"], **kw)
    return pm, fm


def scaled(comps, pos_map, ks):
    """Apply the shared book overlay to a position map."""
    out = {}
    for key, c in comps.items():
        sc = ks.reindex(c["dates"]).ffill().fillna(1.0).values
        out[key] = pos_map[key] * sc
    return out


def book(comps, pos_map, cost, fund_on):
    """Portfolio series for a position map under given frictions."""
    out = {}
    for (k, a), c in comps.items():
        pos = pos_map[(k, a)]
        dpos = np.abs(np.diff(pos, prepend=0.0))
        phi = (c["fund"] if fund_on else 0.0) / 365.0
        net = pos * c["r"] - cost * dpos - phi * np.maximum(-pos, 0.0)
        net = net.copy()
        net[-1] -= cost * abs(pos[-1])
        out.setdefault(k, {})[a] = pd.Series(net, index=c["dates"])
    return aggregate(out)


def components(comps, pos_map):
    """Portfolio-level gross, cost and funding series, separately."""
    g, c_, f = {}, {}, {}
    for (k, a), cc in comps.items():
        pos = pos_map[(k, a)]
        dpos = np.abs(np.diff(pos, prepend=0.0))
        cost = COST * dpos
        cost = cost.copy()
        cost[-1] += COST * abs(pos[-1])
        g.setdefault(k, {})[a] = pd.Series(pos * cc["r"], index=cc["dates"])
        c_.setdefault(k, {})[a] = pd.Series(cost, index=cc["dates"])
        f.setdefault(k, {})[a] = pd.Series(
            (cc["fund"] / 365.0) * np.maximum(-pos, 0.0), index=cc["dates"])
    return aggregate(g), aggregate(c_), aggregate(f)


# ------------------------------------------------------------ symmetry checks
def part1_positions(store):
    print("\n== exact position symmetry, before frictions ==")
    rows = []
    for model, comps in store.items():
        for match in MATCHES:
            pm, fm = arms(comps, match)
            ks = book_scale(price_positions(comps, pm))
            pm_s, fm_s = scaled(comps, pm, ks), scaled(comps, fm, ks)
            a = np.concatenate([pm_s[k] for k in sorted(comps)])
            b = np.concatenate([fm_s[k] for k in sorted(comps)])
            da = np.concatenate([np.abs(np.diff(pm_s[k], prepend=0.0))
                                 for k in sorted(comps)])
            db = np.concatenate([np.abs(np.diff(fm_s[k], prepend=0.0))
                                 for k in sorted(comps)])
            m_sum = float(np.max(np.abs(a + b)))
            m_abs = float(np.max(np.abs(np.abs(a) - np.abs(b))))
            m_dp = float(np.max(np.abs(da - db)))
            rows.append(dict(model=model, match=match, max_pos_sum=m_sum,
                             max_abs_diff=m_abs, max_dpos_diff=m_dp,
                             n_asset_days=len(a)))
            check(f"{model} {match} max|pos_pred+pos_flip|", m_sum, TOL_POS)
            check(f"{model} {match} max||pos_pred|-|pos_flip||", m_abs, TOL_POS)
            check(f"{model} {match} max||dpos_p|-|dpos_f||", m_dp, TOL_POS)
    return pd.DataFrame(rows)


def part2_zero_friction(store):
    print("\n== zero-friction unit test ==")
    rows = []
    for model, comps in store.items():
        for match in MATCHES:
            pm, fm = arms(comps, match)
            ks = book_scale(price_positions(comps, pm))
            pm_s, fm_s = scaled(comps, pm, ks), scaled(comps, fm, ks)
            rp = book(comps, pm_s, 0.0, False)
            rf = book(comps, fm_s, 0.0, False)
            al = pd.concat({"p": rp, "f": rf}, axis=1).dropna()
            m_ret = float(np.max(np.abs(al["p"].values + al["f"].values)))
            sp, sf = sharpe(al["p"].values), sharpe(al["f"].values)
            rows.append(dict(model=model, match=match, SR_pred=sp, SR_flip=sf,
                             SR_sum=sp + sf, max_ret_sum=m_ret))
            check(f"{model} {match} max|r_pred+r_flip|", m_ret, TOL_RET)
            check(f"{model} {match} |SR_pred+SR_flip|", abs(sp + sf), TOL_SR)
    return pd.DataFrame(rows)


def part3_decompose(store):
    print("\n== friction decomposition ==")
    rows = []
    for model, comps in store.items():
        for match in MATCHES:
            pm, fm = arms(comps, match)
            ks = book_scale(price_positions(comps, pm))
            pm_s, fm_s = scaled(comps, pm, ks), scaled(comps, fm, ks)
            gp, cp, fp = components(comps, pm_s)
            gf, cf, ff = components(comps, fm_s)
            for label, cost, fon in (("A none", 0.0, False), ("B cost", COST, False),
                                     ("C funding", 0.0, True), ("D both", COST, True)):
                rp = book(comps, pm_s, cost, fon)
                rf = book(comps, fm_s, cost, fon)
                al = pd.concat({"p": rp, "f": rf}, axis=1).dropna()
                rows.append(dict(
                    model=model, match=match, frictions=label,
                    SR_pred=sharpe(al["p"].values), SR_flip=sharpe(al["f"].values),
                    dSR=sharpe(al["p"].values) - sharpe(al["f"].values),
                    ann_ret_pred=float(al["p"].mean() * 365),
                    ann_ret_flip=float(al["f"].mean() * 365),
                    ann_vol_pred=float(al["p"].std() * np.sqrt(365)),
                    ann_vol_flip=float(al["f"].std() * np.sqrt(365)),
                    gross_pred_bps=float(gp.mean() * 1e4),
                    gross_flip_bps=float(gf.mean() * 1e4),
                    cost_bps=float(cp.mean() * 1e4) if cost else 0.0,
                    cost_flip_bps=float(cf.mean() * 1e4) if cost else 0.0,
                    fund_pred_bps=float(fp.mean() * 1e4) if fon else 0.0,
                    fund_flip_bps=float(ff.mean() * 1e4) if fon else 0.0))
            # the algebraic identity, on the cost-only series
            rp_c = book(comps, pm_s, COST, False)
            rf_c = book(comps, fm_s, COST, False)
            al = pd.concat({"p": rp_c, "f": rf_c, "c": cp}, axis=1).dropna()
            resid = al["f"].values + al["p"].values + 2 * al["c"].values
            check(f"{model} {match} max|r_flip+r_pred+2C|",
                  float(np.max(np.abs(resid))), 1e-12)
            # cost series identical across arms
            check(f"{model} {match} max|C_pred-C_flip|",
                  float(np.max(np.abs((cp - cf).dropna().values))), 1e-15)
            # funding identity F_pred + F_flip = phi |pos|
            fs = (fp + ff).dropna()
            phi_abs = {}
            for (k, a), c in comps.items():
                phi_abs.setdefault(k, {})[a] = pd.Series(
                    (c["fund"] / 365.0) * np.abs(pm_s[(k, a)]), index=c["dates"])
            check(f"{model} {match} max|F_p+F_f-phi|pos||",
                  float(np.max(np.abs((fs - aggregate(phi_abs)).dropna().values))),
                  1e-15)
    return pd.DataFrame(rows)


def part4_funding(D):
    print("\n== what funding does to the difference ==")
    rows = []
    for (model, match), g in D.groupby(["model", "match"]):
        d_full = float(g[g.frictions == "D both"].dSR.iloc[0])
        d_cost = float(g[g.frictions == "B cost"].dSR.iloc[0])
        rows.append(dict(model=model, match=match, dSR_with_funding=d_full,
                         dSR_no_funding=d_cost, change=d_full - d_cost))
    F = pd.DataFrame(rows)
    print(F.round(4).to_string(index=False))
    mx = F.change.abs().max()
    print(f"  max |change| = {mx:.4f}  -> claim 'at most 0.011' is "
          f"{'CORRECT' if mx <= 0.011 else 'WRONG'}")
    return F, mx


# ------------------------------------------------------- residual regression
def part7_residual(store):
    print("\n== residual regression under three friction settings ==")
    rows = []
    for model, comps in store.items():
        pm, fm = arms(comps, "MATCH-B")
        ks = book_scale(price_positions(comps, pm))
        pm_s, fm_s = scaled(comps, pm, ks), scaled(comps, fm, ks)
        lm = raw_positions(comps, SIGNS["LONG"])
        kl = book_scale(price_positions(comps, lm))
        for label, cost, fon in (("A none", 0.0, False), ("B cost", COST, False),
                                 ("C full", COST, True)):
            lng = book(comps, scaled(comps, lm, kl), cost, fon)
            fits = {}
            for arm, pmap in (("PRED", pm_s), ("FLIP", fm_s)):
                s = book(comps, pmap, cost, fon)
                al = pd.concat({"l": lng, "s": s}, axis=1).dropna()
                x, y = al["l"].values, al["s"].values
                beta = float(((x - x.mean()) * (y - y.mean())).mean() / x.var())
                alpha = float(y.mean() - beta * x.mean())
                res = y - beta * x
                fits[arm] = (alpha, beta, res)
                rows.append(dict(model=model, frictions=label, arm=arm,
                                 alpha_ann=alpha * 365, beta=beta,
                                 sharpe_resid=sharpe(res)))
            if label == "A none":
                ap, bp, ep = fits["PRED"]
                af, bf, ef = fits["FLIP"]
                check(f"{model} zero-friction |alpha_f+alpha_p|", abs(af + ap), 1e-15)
                check(f"{model} zero-friction |beta_f+beta_p|", abs(bf + bp), 1e-12)
                check(f"{model} zero-friction max|eps_f+eps_p|",
                      float(np.max(np.abs(ef + ep))), 1e-14)
    R = pd.DataFrame(rows)
    print(R.round(4).to_string(index=False))
    return R


# --------------------------------------------------- equal-magnitude control
def part8_equal_magnitude(store, B=B_BOOT, block=BLOCK, seed=71):
    """Equal-magnitude counterfactual using the ACTUAL MATCH-B portfolio weights.

    The realised absolute next-day return is replaced by a common magnitude M
    while its sign is preserved, and the portfolio is then re-aggregated through
    the identical weighting. Direction, active mask, volatility sizing, book
    scale, equal asset weighting and the union calendar are all untouched; only
    the difference in magnitude between correct and wrong calls is removed.

    M is the exposure-weighted mean absolute return over the same observation
    set, sum(|q||r|) / sum(|q|), so the counterfactual holds the arm's total
    gross exposure to absolute movement fixed.
    """
    print("\n== weighted equal-magnitude counterfactual (MATCH-B) ==")
    rows = []
    for model, comps in store.items():
        pm, _ = arms(comps, "MATCH-B")
        ks = book_scale(price_positions(comps, pm))
        q = scaled(comps, pm, ks)

        num = den = 0.0
        for key, c in comps.items():
            num += float(np.sum(np.abs(q[key]) * np.abs(c["r"])))
            den += float(np.sum(np.abs(q[key])))
        M = num / den

        act, eq = {}, {}
        for (k, a), c in comps.items():
            w = q[(k, a)]
            act.setdefault(k, {})[a] = pd.Series(w * c["r"], index=c["dates"])
            eq.setdefault(k, {})[a] = pd.Series(
                w * np.sign(c["r"]) * M, index=c["dates"])
        A, E = aggregate(act), aggregate(eq)
        al = pd.concat({"a": A, "e": E}, axis=1).dropna()
        a_v, e_v = al["a"].values, al["e"].values
        I = stationary_bootstrap_idx(len(a_v), B, block, seed)
        da, de = a_v[I].mean(axis=1) * 1e4, e_v[I].mean(axis=1) * 1e4
        dd = da - de
        rows.append(dict(
            model=model, M_pct=M * 100,
            actual_bps=float(a_v.mean() * 1e4),
            actual_lo=float(np.percentile(da, 2.5)),
            actual_hi=float(np.percentile(da, 97.5)),
            equalmag_bps=float(e_v.mean() * 1e4),
            equalmag_lo=float(np.percentile(de, 2.5)),
            equalmag_hi=float(np.percentile(de, 97.5)),
            equalmag_p=float((1 + np.sum(de <= 0)) / (B + 1)),
            diff_bps=float((a_v - e_v).mean() * 1e4),
            diff_lo=float(np.percentile(dd, 2.5)),
            diff_hi=float(np.percentile(dd, 97.5)),
            diff_p=float((1 + np.sum(dd <= 0)) / (B + 1))))
    T = pd.DataFrame(rows)
    print(T.round(3).to_string(index=False))
    return T


def acceptance_checks():
    """The matched-reversal identities, as (name, ok, detail) triples for
    run_all.py's acceptance report. Cheap: no bootstrap, no regression."""
    store = load_components()
    out = []
    worst_pos = worst_ret = worst_sr = worst_id = 0.0
    for model, comps in store.items():
        for match in MATCHES:
            pm, fm = arms(comps, match)
            ks = book_scale(price_positions(comps, pm))
            pm_s, fm_s = scaled(comps, pm, ks), scaled(comps, fm, ks)
            a = np.concatenate([pm_s[k] for k in sorted(comps)])
            b = np.concatenate([fm_s[k] for k in sorted(comps)])
            worst_pos = max(worst_pos, float(np.max(np.abs(a + b))))

            rp = book(comps, pm_s, 0.0, False)
            rf = book(comps, fm_s, 0.0, False)
            al = pd.concat({"p": rp, "f": rf}, axis=1).dropna()
            worst_ret = max(worst_ret,
                            float(np.max(np.abs(al["p"].values + al["f"].values))))
            worst_sr = max(worst_sr,
                           abs(sharpe(al["p"].values) + sharpe(al["f"].values)))

            _, cp, _ = components(comps, pm_s)
            rpc = book(comps, pm_s, COST, False)
            rfc = book(comps, fm_s, COST, False)
            al2 = pd.concat({"p": rpc, "f": rfc, "c": cp}, axis=1).dropna()
            worst_id = max(worst_id, float(np.max(np.abs(
                al2["f"].values + al2["p"].values + 2 * al2["c"].values))))

    out.append(("matched.positions_exact_negatives", worst_pos < TOL_POS,
                f"max|pos_pred+pos_flip| = {worst_pos:.2e}"))
    out.append(("matched.gross_returns_negate", worst_ret < TOL_RET,
                f"max|r_pred+r_flip| at zero friction = {worst_ret:.2e}"))
    out.append(("matched.sharpe_mirrors_at_zero_friction", worst_sr < TOL_SR,
                f"max|SR_pred+SR_flip| = {worst_sr:.2e}"))
    out.append(("matched.cost_identity", worst_id < 1e-12,
                f"max|r_flip+r_pred+2C| = {worst_id:.2e}"))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    store = load_components()

    P1 = part1_positions(store);      P1.to_csv(OUT / "part1_positions.csv", index=False)
    P2 = part2_zero_friction(store);  P2.to_csv(OUT / "part2_zero_friction.csv", index=False)
    print("\n", P2.round(6).to_string(index=False))
    P3 = part3_decompose(store);      P3.to_csv(OUT / "part3_frictions.csv", index=False)
    print("\nMATCH-B friction ladder:")
    print(P3[P3.match == "MATCH-B"][
        ["model", "frictions", "SR_pred", "SR_flip", "dSR", "gross_pred_bps",
         "cost_bps", "fund_pred_bps", "fund_flip_bps"]].round(4).to_string(index=False))
    P4, mx = part4_funding(P3); P4.to_csv(OUT / "part4_funding.csv", index=False)
    P7 = part7_residual(store); P7.to_csv(OUT / "part7_residual.csv", index=False)
    P8 = part8_equal_magnitude(store); P8.to_csv(OUT / "part8_equal_magnitude.csv", index=False)

    print("\n== acceptance assertions ==")
    if FAILURES:
        print("  FAILED:")
        for f in FAILURES:
            print("   ", f)
    else:
        print(f"  all {0} identities hold at machine precision".replace("0", "checked"))
    print(f"\n-> {OUT}")
    return not FAILURES


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
