"""Supporting strategy tables: the provenance of the per-fold accuracy table,
the unmatched exposure table at a single documented convention, and every
strategy paired proof recomputed at B = 10 000 with the (1+k)/(B+1) estimator.

Outputs -> results/analysis/
"""
import numpy as np
import pandas as pd

from portfolio_replay import (OUT, ROOT, SIGNS, exposures, load_components,
                             paired_tests, portfolio, sharpe)

PRED_DIR = ROOT / "artifacts" / "forecast_predictions"
EVAL = ["TSLA", "FTSE", "AUDUSD", "COPPER", "LTC"]
TOP4 = ["GRU", "LR", "TimeFilter", "TimeMixer"]
FOLD_YEAR = {0: 2021, 1: 2022, 2: 2023, 3: 2024, 4: 2025}


def fold_accuracy():
    """Per-fold accuracy against the three truths, for each candidate source, so
    the manuscript's Table 2 can be attributed to exactly one of them."""
    out = {}
    frames = {}
    for m in ["LR", "RF", "XGB", "LGBM", "ARIMA", "GRU", "LSTM", "DLinear",
              "PatchTST", "TimeMixer", "TimeFilter", "Crossformer", "FEDformer"]:
        fp = PRED_DIR / f"forecast__{m}__predictions.csv"
        if fp.exists():
            frames[m] = pd.read_csv(fp)
    for m, d in frames.items():
        d = d[d.asset.isin(EVAL)]
        g = d.assign(
            trend=(np.sign(d.y_pred) == np.sign(d.y_true)).astype(float),
            price_h=(np.sign(d.y_pred) == np.sign(d.y_true_price_h)).astype(float),
            price_1=(np.sign(d.y_pred) == np.sign(d.y_true_price_1)).astype(float),
        ).groupby("fold")[["trend", "price_h", "price_1"]].mean()
        out[m] = g
    allm = pd.concat(out, names=["model"])
    tab = {m: g for m, g in out.items()}
    tab["MEAN of 13"] = allm.groupby("fold").mean()
    tab["MEAN of top 4"] = pd.concat({m: out[m] for m in TOP4}).groupby("fold").mean()
    R = pd.concat(tab, names=["source", "fold"]).reset_index()
    R["fold"] = R["fold"].map(FOLD_YEAR)
    return R


def fold_accuracy_portfolio():
    """The same three-truths accuracy on the TEN PORTFOLIO assets, per fold.

    The forecast-evaluation export only covers the five comparison assets, so
    the portfolio-universe accuracies are read off the cached strategy
    components, which carry the prediction `w` alongside the realised trend
    change `d_te`, the raw h-day price change `fwd_h` and the next-day return
    `r` on exactly the days the book traded.
    """
    store = load_components()
    rows = []
    for model, comps in store.items():
        acc = {}
        for (k, a), c in comps.items():
            s = np.sign(c["w"])
            s[s == 0] = 1.0
            acc.setdefault(k, []).append(pd.DataFrame({
                "trend": (s == np.sign(c["d_te"])).astype(float),
                "price_h": (s == np.sign(c["fwd_h"])).astype(float),
                "price_1": (s == np.sign(c["r"])).astype(float)}))
        for k, parts in sorted(acc.items()):
            # average within an asset first, then across assets, so a long
            # crypto series does not outvote a shorter equity one
            per_asset = pd.concat([p.mean().to_frame().T for p in parts])
            rows.append(dict(model=model, fold=FOLD_YEAR[k],
                             **per_asset.mean().to_dict(),
                             n_assets=len(parts),
                             n_days=int(sum(len(p) for p in parts))))
    return pd.DataFrame(rows)


def unmatched_exposure():
    """Gross, net and turnover of the PRED and FLIP arms measured on the
    PER-ASSET positions BEFORE the book-level overlay -- the convention the
    manuscript's unmatched exposure table uses."""
    store = load_components()
    rows = []
    for m, comps in store.items():
        e = {arm: exposures(comps, SIGNS[arm], scaled=False) for arm in ("PRED", "FLIP")}
        rows.append(dict(model=m,
                         gross_pred=e["PRED"]["gross"], gross_flip=e["FLIP"]["gross"],
                         net_pred=e["PRED"]["net"], net_flip=e["FLIP"]["net"],
                         turn_pred=e["PRED"]["turnover"], turn_flip=e["FLIP"]["turnover"],
                         gross_ratio=e["PRED"]["gross"] / e["FLIP"]["gross"]))
    return pd.DataFrame(rows).set_index("model").loc[
        ["TimeFilter", "LR", "GRU", "TimeMixer"]]


def paired_proofs(signal="GRU"):
    """Every strategy comparison in the manuscript, at B = 10 000.

    The signal is the one the automatic selection rule picks. Arms are aligned
    on identical days and share the same resample indices.
    """
    store = load_components()
    comps = store[signal]
    arms = {
        "PRED": portfolio(comps, SIGNS["PRED"]),
        "BLEND": portfolio(comps, SIGNS["PRED"], blend=True),
        "LONG": portfolio(comps, SIGNS["LONG"]),
        "REGIME": portfolio(comps, SIGNS["REGIME"]),
        "FLIP": portfolio(comps, SIGNS["FLIP"]),
        "NOSCALE": portfolio(comps, SIGNS["PRED"], scaled=False),
    }
    rows = {}
    for name, (base, new) in {
        "PRED vs LONG": ("LONG", "PRED"),
        "PRED vs REGIME": ("REGIME", "PRED"),
        "PRED vs FLIP": ("FLIP", "PRED"),
        "PRED vs no overlay": ("NOSCALE", "PRED"),
        "BLEND vs PRED": ("PRED", "BLEND"),
        "BLEND vs REGIME": ("REGIME", "BLEND"),
    }.items():
        al = pd.concat({"b": arms[base], "n": arms[new]}, axis=1).dropna()
        rows[name] = paired_tests(al["b"].values, al["n"].values, B=10000)
    P = pd.DataFrame(rows).T
    P.insert(0, "sharpe_new", [sharpe(arms[n].values) for _, n in [
        ("LONG", "PRED"), ("REGIME", "PRED"), ("FLIP", "PRED"),
        ("NOSCALE", "PRED"), ("PRED", "BLEND"), ("REGIME", "BLEND")]])
    P.insert(0, "sharpe_base", [sharpe(arms[b].values) for b, _ in [
        ("LONG", "PRED"), ("REGIME", "PRED"), ("FLIP", "PRED"),
        ("NOSCALE", "PRED"), ("PRED", "BLEND"), ("REGIME", "BLEND")]])
    return P, {k: sharpe(v.values) for k, v in arms.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=== per-fold accuracy by source (five evaluation assets) ===")
    R = fold_accuracy()
    R.to_csv(OUT / "fold_accuracy_by_source.csv", index=False)
    piv = R.pivot_table(index="source", columns="fold",
                        values=["trend", "price_h", "price_1"])
    print(piv.round(3).to_string())

    print("\n=== per-fold accuracy on the TEN portfolio assets ===")
    F = fold_accuracy_portfolio()
    F.to_csv(OUT / "fold_accuracy_portfolio.csv", index=False)
    print(F.pivot_table(index="model", columns="fold",
                        values=["trend", "price_h", "price_1"]).round(3).to_string())
    print("\n  means over folds:")
    print(F.groupby("model")[["trend", "price_h", "price_1"]].mean().round(4).to_string())

    print("\n=== unmatched exposure, per-asset positions before the overlay ===")
    E = unmatched_exposure()
    E.to_csv(OUT / "exposure_unmatched.csv")
    print(E.round(4).to_string())

    print("\n=== paired proofs at B = 10 000 (GRU, the selected signal) ===")
    P, levels = paired_proofs("GRU")
    P.to_csv(OUT / "paired_proofs_B10000.csv")
    print("arm Sharpe:", {k: round(v, 3) for k, v in levels.items()})
    print(P.round(4).to_string())
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
