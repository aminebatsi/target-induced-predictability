"""Numerical consistency audit of the KBS manuscript.

Every quantitative claim in `paper/manuscript-kbs-final.tex` is re-derived here
from the result files and checked against the text, so a number cannot drift in
the abstract, a table, the discussion or the conclusion without this failing.
Nothing is hard-coded that could instead be computed: the expected values come
from `results/analysis/*.csv` and from the cached predictions, and the check is
that the manuscript contains the string the data implies.

Run after any rerun. Exit status is non-zero if any check fails.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TEX = ROOT.parent / "paper" / "manuscript-kbs-final.tex"
A = ROOT / "results" / "analysis"
PRED = ROOT / "artifacts" / "forecast_predictions"

MODELS = ["LR", "RF", "XGB", "LGBM", "ARIMA", "GRU", "LSTM", "DLinear",
          "PatchTST", "TimeMixer", "TimeFilter", "Crossformer", "FEDformer"]

PASS, FAIL = [], []


def chk(label, cond, detail=""):
    (PASS if cond else FAIL).append(f"{label}{(' -- ' + detail) if detail else ''}")


def plain(tex):
    """Markup-stripped, whitespace-collapsed text, so a phrase split across a
    line break or wrapped in \\textbf still matches."""
    t = re.sub(r"\s+", " ", tex)
    for _ in range(3):
        t = re.sub(r"\\(?:textbf|emph|textit|texttt|mathrm|textsc)\{([^{}]*)\}",
                   r"\1", t)
    return t


# ---------------------------------------------------------------- sources
def three_truths():
    """Pooled directional accuracy of each model against the three objects."""
    rows = []
    for m in MODELS:
        fp = PRED / f"forecast__{m}__predictions.csv"
        if not fp.exists():
            continue
        d = pd.read_csv(fp)
        s = np.sign(d.y_pred.values)
        rows.append(dict(
            model=m, n=len(d),
            target=float(np.mean(s == np.sign(d.y_true.values))),
            price_h=float(np.mean(s == np.sign(d.y_true_price_h.values))),
            price_1=float(np.mean(s == np.sign(d.y_true_price_1.values)))))
    return pd.DataFrame(rows).set_index("model")


def main():
    if not TEX.exists():
        print(f"manuscript not found at {TEX}")
        return 1
    tex = TEX.read_text(encoding="utf-8")
    P = plain(tex)

    # ---------------------------------------------------- design constants
    chk("13 models present", len(MODELS) == 13)
    T = three_truths()
    chk("all 13 models have predictions", len(T) == 13,
        f"found {len(T)}: {sorted(set(MODELS) - set(T.index))} missing")
    chk("6760 forecasts per model", set(T.n) == {6760}, str(sorted(set(T.n))))
    chk("6760 quoted", "6760" in tex)
    for s in ["five families", "fifteen causal transforms", "five horizons",
              "fifteen assets", "five walk-forward folds", "5610"]:
        chk(f"design constant: {s}", s in P)

    # ---------------------------------------------------- three-truths table
    for m in T.index:
        for col, tag in (("target", "target"), ("price_h", "price-h"),
                         ("price_1", "price-1")):
            v = f"{T.loc[m, col]:.4f}"
            chk(f"table value {m} {tag} = {v}", v in tex)
    mt, mh, m1 = T.target.mean(), T.price_h.mean(), T.price_1.mean()
    for v, tag in ((mt, "mean target"), (mh, "mean price-h"), (m1, "mean price-1")):
        chk(f"{tag} {v:.4f} in text", f"{v:.4f}" in tex)
    # abstract / conclusion quote these to 3 decimals
    chk("abstract mean target 3dp", f"{mt:.3f}" in tex)
    chk("abstract mean price-h 3dp", f"{mh:.3f}" in tex)
    chk("abstract mean price-1 3dp", f"{m1:.3f}" in tex)

    # ---------------------------------------------------- purge audit
    if (A / "purge_audit.csv").exists():
        U = pd.read_csv(A / "purge_audit.csv")
        obs = U[U.rule == "observation index"]
        cols = ["train_to_val", "val_to_test", "train_to_test"]
        chk("purge: zero crossings under observation index",
            int(obs[cols].to_numpy().sum()) == 0)
        chk("purge: audit covers 75 cells", len(obs) == 75, str(len(obs)))
        leg = U[U.rule == "calendar days"]
        chk("purge: legacy val->test count quoted",
            str(int(leg.val_to_test.sum())) in tex,
            f"expected {int(leg.val_to_test.sum())}")
        chk("purge: legacy train->val count quoted",
            str(int(leg.train_to_val.sum())) in tex,
            f"expected {int(leg.train_to_val.sum())}")
    else:
        chk("purge audit present", False, "purge_audit.csv missing")

    # ---------------------------------------------------- ExDA
    if (A / "exda_point.csv").exists():
        E = pd.read_csv(A / "exda_point.csv").set_index("model")
        B = pd.read_csv(A / "exda_bootstrap.csv")
        bench = float(E.loc[E.index[0], "benchmark"])
        chk(f"ExDA benchmark {bench:.4f} in text", f"{bench:.4f}" in tex)
        fitted = E.drop(index="SLOPE", errors="ignore")
        chk("ExDA table accuracies match three-truths",
            all(abs(fitted.loc[m, "DA"] - T.loc[m, "target"]) < 5e-5
                for m in fitted.index if m in T.index),
            "Table 4 and the ExDA table disagree on DA_target")
        me = fitted.ExDA.mean()
        chk(f"mean ExDA {me:.4f} in text", f"{abs(me):.4f}" in tex)
        nneg = int((B.hi < 0).sum())
        npos = int((B.lo > 0).sum())
        WORD = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
                11: "eleven", 12: "twelve", 13: "thirteen"}
        chk(f"{WORD[nneg]} models significantly below the reference",
            f"{WORD[nneg]} of the thirteen fitted models perform" in P,
            f"computed {nneg}")
        chk("no model significantly above the reference", npos == 0,
            f"{npos} models have an interval entirely above zero")
    else:
        chk("ExDA results present", False, "exda_point.csv missing")

    # ---------------------------------------------------- raw-price null
    if (A / "raw_price_null.csv").exists():
        N = pd.read_csv(A / "raw_price_null.csv")
        WORD = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
                11: "eleven", 12: "twelve", 13: "thirteen"}
        p1 = N[N.object == "price-1"]
        ph = N[N.object == "price-h"]
        n1 = int((p1.p_two_sided < 0.05).sum())
        nh = int((ph.p_two_sided < 0.05).sum())
        chk(f"price-1: {WORD[n1]} of thirteen intervals exclude zero",
            f"excludes zero for {WORD[n1]} of them" in P, f"computed {n1}")
        # price-h: the manuscript must say either how many intervals exclude
        # zero, or that none does -- whichever the data supports.
        chk(f"price-h: manuscript matches the computed count ({nh})",
            (f"{WORD[nh]} of thirteen intervals" in P) if nh else
            ("no $h$-day interval excludes zero" in P
             or "no interval excludes zero" in P),
            f"computed {nh}")
        chk("MCC negative for all 13 on price-1", bool((p1.MCC < 0).all()))
        for tag, g in (("price-h", ph), ("price-1", p1)):
            u = float(g.up_rate_actual.iloc[0])
            chk(f"{tag} realised up-rate {u:.3f} quoted", f"{u:.3f}" in tex)
    else:
        chk("raw-price null present", False, "raw_price_null.csv missing")

    # ---------------------------------------------------- SPA
    if (A / "raw_price_null_spa.csv").exists():
        S = pd.read_csv(A / "raw_price_null_spa.csv")
        g = S.set_index(["object", "orientation"]).p_spa
        for (obj, ori), v in g.items():
            s = f"{v:.3f}" if v >= 5e-4 else None
            if s is not None:
                chk(f"SPA {obj}/{ori} p={s} in text", s in tex)
        # the contradiction item 2 was about
        ph_inv = g.get(("price-h", "inverted"), np.nan)
        p1_inv = g.get(("price-1", "inverted"), np.nan)
        both_sig = (ph_inv < 0.05) and (p1_inv < 0.05)
        chk("significance statement matches both-horizon result",
            (not both_sig) or ("significant at both raw-price horizons" in P),
            f"price-h inverted p={ph_inv:.4f}, price-1 inverted p={p1_inv:.4f}")
        chk("no 'significant only at the one-day horizon' claim",
            "significant only at the one-day horizon" not in P)
    else:
        chk("SPA results present", False, "raw_price_null_spa.csv missing")

    # ---------------------------------------------------- direction baselines
    if (A / "direction_baselines.csv").exists():
        D = pd.read_csv(A / "direction_baselines.csv").set_index(["model", "object"])
        for (m, o), r in D.iterrows():
            chk(f"dir-baseline {m}/{o} = {r.DA:.4f}", f"{r.DA:.4f}" in tex)
    else:
        chk("direction baselines present", False, "direction_baselines.csv missing")

    # ---------------------------------------------------- synthetic null
    if (A / "synthetic_null.csv").exists():
        G = pd.read_csv(A / "synthetic_null.csv")
        chk("synthetic: 75 cells", len(G) == 75, str(len(G)))
        raw7 = G[(G.family == "no smoothing") & (G.horizon == 7)]
        chk("synthetic control accuracy quoted",
            f"{float(raw7.da_target.iloc[0]):.4f}" in tex)
        chk("synthetic price range quoted",
            f"{G.da_price_1.min():.4f}"[:6] in tex or
            f"{min(G.da_price_1.min(), G.da_price_h.min()):.4f}" in tex)
        chk("synthetic max target accuracy consistent",
            f"{G.da_target.max():.3f}" in tex, f"{G.da_target.max():.4f}")
    else:
        chk("synthetic null present", False, "synthetic_null.csv missing")

    # ---------------------------------------------------- equivalence bounds
    if (A / "equivalence_bounds.csv").exists():
        Q = pd.read_csv(A / "equivalence_bounds.csv")
        for q, tag in [("DA - Ps, price-h", "price-h"),
                       ("DA - Ps, price-1", "price-1"),
                       ("DA(model) - DA(slope), target", "slope")]:
            g = Q[Q.quantity == q]
            chk(f"bound {tag}: largest upper {g.hi.max():+.3f}",
                f"{abs(g.hi.max()):.3f}" in tex)
            chk(f"bound {tag}: median MDE {g.mde.median():.3f}",
                f"{g.mde.median():.3f}" in tex)
        sl = Q[Q.quantity == "DA(model) - DA(slope), target"]
        chk("slope bound: every model below +0.010", bool((sl.hi < 0.010).all()),
            f"max {sl.hi.max():+.4f}")
        chk("slope bound quoted to 4dp", f"{sl.hi.max():.4f}" in tex)
        ph = Q[Q.quantity == "DA - Ps, price-h"]
        chk("price-h: twelve of thirteen below +0.010",
            int((ph.hi < 0.010).sum()) == 12, str(int((ph.hi < 0.010).sum())))
    else:
        chk("equivalence bounds present", False, "equivalence_bounds.csv missing")

    # ---------------------------------------------------- non-overlap check
    if (A / "nonoverlap_price_h.csv").exists():
        N = pd.read_csv(A / "nonoverlap_price_h.csv")
        chk("non-overlap point estimate quoted",
            f"{abs(N.point.mean()):.4f}" in tex, f"{N.point.mean():+.4f}")
        chk("non-overlap significant-cell count",
            f"{int((N.p_two_sided < 0.05).sum())} of the 91" in P
            or f"{int((N.p_two_sided < 0.05).sum())} of 91" in P,
            f"{int((N.p_two_sided<0.05).sum())}/{len(N)}")
    else:
        chk("non-overlap check present", False, "nonoverlap_price_h.csv missing")

    # ---------------------------------------------------- causal wavelet
    if (A / "causal_wavelet.csv").exists():
        W = pd.read_csv(A / "causal_wavelet.csv")
        g = W.groupby(["transform", "h"]).agg(
            rho1=("rho1", "mean"), da=("da_target", "mean"), B=("B", "mean"),
            ph=("da_price_h", "mean"), p1=("da_price_1", "mean"))
        for (t, h) in [("causal wavelet", 1), ("causal wavelet", 7),
                       ("causal wavelet", 30), ("full-sample wavelet", 1),
                       ("full-sample wavelet", 7), ("full-sample wavelet", 30)]:
            r = g.loc[(t, h)]
            for v in (r.rho1, r.da, r.B, r.ph, r.p1):
                chk(f"wavelet {t} h={h} value {v:.3f}", f"{abs(v):.3f}" in tex)
        c7 = g.loc[("causal wavelet", 7)]
        f7 = g.loc[("full-sample wavelet", 7)]
        chk("full-sample wavelet moves raw price, causal does not",
            (f7.ph - c7.ph) > 0.11 and abs(c7.p1 - 0.5) < 0.02,
            f"causal price-h {c7.ph:.3f}, full {f7.ph:.3f}")
    else:
        chk("causal wavelet present", False, "causal_wavelet.csv missing")

    # ---------------------------------------------------- asset robustness
    if (A / "robust_loao.csv").exists():
        R = pd.read_csv(A / "robust_loao.csv")
        Ab = pd.read_csv(A / "robust_by_asset.csv")
        Fo = pd.read_csv(A / "robust_by_fold.csv")
        chk("LOAO gap range quoted",
            f"{R.gap.min():.3f}" in tex and f"{R.gap.max():.3f}" in tex)
        chk("by-asset gap range quoted",
            f"{Ab.gap.min():.3f}" in tex and f"{Ab.gap.max():.3f}" in tex)
        chk("by-fold gap range quoted",
            f"{Fo.gap.min():.3f}" in tex and f"{Fo.gap.max():.3f}" in tex)
        chk("target beats price-1 in every asset and fold",
            bool((Ab.mean_target > Ab.mean_price_1).all()
                 and (Fo.mean_target > Fo.mean_price_1).all()))
        hi = Ab.loc[Ab.mean_price_1.idxmax()]
        chk(f"the one asset above 0.50 on price-1 is disclosed ({hi.asset})",
            f"{hi.mean_price_1:.4f}" in tex, f"{hi.asset} {hi.mean_price_1:.4f}")
    else:
        chk("asset robustness present", False, "robust_loao.csv missing")

    # ---------------------------------------------------- protocol constants
    chk("bootstrap B = 10,000", "10\\,000" in tex or "10,000" in tex)
    chk("mean block 20", "mean block 20" in P)
    chk("fold span stated", "2021-07-01" in tex and "2026" in tex)

    # ---------------------------------------------------- forbidden content
    for f in ["Sharpe", "portfolio", "turn classifier", "MATCH-", "backtest",
              "Author One", "Author Two"]:
        chk(f"absent: {f}", f not in tex)
    # "profitability" may appear only in disclaimers, never as a claim. Check
    # every sentence containing it is a denial.
    prof = [s for s in re.split(r"(?<=[.!?]) ", P) if "profitab" in s.lower()]
    chk("profitability only ever disclaimed",
        all(any(k in s.lower() for k in
                ("no claim", "makes no", "tests no", "not tested",
                 "nothing here", "we do not", "does not"))
            for s in prof),
        f"{len(prof)} sentence(s): {prof}")
    chk("no exploitability claim",
        "nothing here tests whether the inverted signal is" in P
        or "not tested here" in P)
    chk("no unresolved TODO outside metadata",
        not re.search(r"TODO(?! BEFORE SUBMISSION)", tex))

    # ---------------------------------------------------- LaTeX integrity
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", tex))
    chk("no dangling references", not (refs - set(labels)), str(refs - set(labels)))
    dup = [l for l in set(labels) if labels.count(l) > 1]
    chk("no duplicate labels", not dup, str(dup))
    kw = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", tex, re.S)
    nkw = len([k for k in kw.group(1).split("\\sep") if k.strip()]) if kw else 0
    chk("at most 7 keywords", nkw <= 7, f"{nkw} keywords")

    # ---------------------------------------------------- report
    print(f"{len(PASS)} checks passed, {len(FAIL)} failed")
    for f in FAIL:
        print("  FAIL:", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
