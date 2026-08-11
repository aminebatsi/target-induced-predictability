"""Where does the Gaussian sign-agreement benchmark go wrong?

The benchmark is exact only for a zero-mean bivariate normal pair. Filtered
financial increments are neither. The sweep shows a mean absolute deviation of
about 0.032 and a mean signed residual of about +0.019, so the approximation is
good but biased upward. This module asks where the bias sits and what drives it.

Three diagnostics, all descriptive. No significance tests are run over the
partitions: with 5610 dependent cells, dozens of tests would say nothing.

  1.  residual by transform family, horizon, asset, asset class, smoothing
      quintile, rho_1 bin and |rho_1| bin;
  2.  the single largest absolute residual, with its full metadata;
  3.  the centering diagnostic -- rho is invariant to centering, so comparing
      sign agreement before and after centering isolates the contribution of
      nonzero location.

Outputs -> results/analysis/
"""
import numpy as np
import pandas as pd

from portfolio_replay import ROOT

OUT = ROOT / "results" / "analysis"
SWEEP = ROOT / "results" / "analysis" / "slope_generality_sweep.csv"


def load():
    T = pd.read_csv(SWEEP)
    T["residual"] = T.da_slope - T.da_gaussian
    T["abs_residual"] = T.residual.abs()
    if "da_slope_centered" in T:
        T["residual_c"] = T.da_slope_centered - T.da_gaussian
        T["abs_residual_c"] = T.residual_c.abs()
    return T


def by(T, col, bins=None, labels=None):
    g = T[col] if bins is None else pd.cut(T[col], bins=bins, labels=labels)
    out = T.groupby(g, observed=True).agg(
        cells=("residual", "size"),
        mean_residual=("residual", "mean"),
        mad=("abs_residual", "mean"),
        worst=("abs_residual", "max"),
        mean_rho1=("rho1", "mean")).round(4)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T = load()
    print(f"cells: {len(T)}")
    print(f"mean residual   {T.residual.mean():+.4f}")
    print(f"MAD             {T.abs_residual.mean():.4f}")
    print(f"worst |residual| {T.abs_residual.max():.4f}\n")

    # ---------------- 5D: where the bias sits ----------------
    parts = {}
    parts["family"] = by(T, "family")
    parts["horizon"] = by(T, "h")
    parts["asset_class"] = by(T, "asset_class")
    parts["asset"] = by(T, "asset").sort_values("mean_residual")
    T["strength_q"] = pd.qcut(T.smooth_ratio, 5,
                              labels=["Q1 smoothest", "Q2", "Q3", "Q4",
                                      "Q5 least smooth"])
    parts["smoothing_quintile"] = by(T, "strength_q")
    parts["rho1_bin"] = by(T, "rho1", bins=[-1, 0, .2, .4, .6, .8, 1.0],
                           labels=["<0", "0-.2", ".2-.4", ".4-.6", ".6-.8", ".8-1"])
    T["abs_rho1"] = T.rho1.abs()
    parts["abs_rho1_bin"] = by(T, "abs_rho1", bins=[0, .2, .4, .6, .8, 1.0],
                               labels=["0-.2", ".2-.4", ".4-.6", ".6-.8", ".8-1"])

    for k, v in parts.items():
        print(f"--- residual by {k}")
        print(v.to_string(), "\n")
    pd.concat(parts, names=["partition"]).to_csv(OUT / "residual_partitions.csv")

    # ---------------- 5E: the worst cell ----------------
    w = T.loc[T.abs_residual.idxmax()]
    keep = ["asset", "asset_class", "fold", "h", "smoother", "family", "strength",
            "n", "rho1", "da_slope", "da_gaussian", "residual", "smooth_ratio",
            "skew_tgt", "kurt_tgt", "mean_tgt_over_sd", "mean_inc_over_sd",
            "da_slope_centered", "residual_c"]
    keep = [c for c in keep if c in T.columns]
    print("--- 5E worst cell")
    print(w[keep].to_string(), "\n")
    T.nlargest(10, "abs_residual")[keep].to_csv(OUT / "worst_cells.csv", index=False)
    print("--- ten largest |residual| cells")
    print(T.nlargest(10, "abs_residual")[
        ["asset", "fold", "h", "smoother", "n", "rho1", "da_slope",
         "da_gaussian", "residual"]].round(4).to_string(index=False), "\n")

    # how unusual is that cell on the features that could explain it?
    print("percentile of the worst cell within the sweep:")
    for c in ("n", "rho1", "abs_rho1", "skew_tgt", "kurt_tgt",
              "mean_tgt_over_sd", "smooth_ratio"):
        if c in T.columns:
            pct = float((T[c] <= w[c]).mean() * 100)
            print(f"  {c:18} value {w[c]:>10.4f}   percentile {pct:5.1f}")

    # ---------------- 5F: centering ----------------
    if "residual_c" in T.columns:
        print("\n--- 5F centering diagnostic (rho is unchanged by centering)")
        rows = [dict(version="original",
                     mean_residual=T.residual.mean(), mad=T.abs_residual.mean(),
                     worst=T.abs_residual.max()),
                dict(version="centered",
                     mean_residual=T.residual_c.mean(), mad=T.abs_residual_c.mean(),
                     worst=T.abs_residual_c.max())]
        C = pd.DataFrame(rows)
        C.to_csv(OUT / "centering_diagnostic.csv", index=False)
        print(C.round(4).to_string(index=False))
        drop = 1 - C.loc[1, "mean_residual"] / C.loc[0, "mean_residual"]
        print(f"  centering removes {drop:.0%} of the mean signed residual")
        print("\n  by horizon (drift grows with h, so this is the natural cut):")
        print(T.groupby("h").agg(
            mean_res=("residual", "mean"), mean_res_centered=("residual_c", "mean"),
            mean_drift=("mean_tgt_over_sd", "mean")).round(4).to_string())

    T.to_csv(OUT / "sweep_with_residuals.csv", index=False)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
