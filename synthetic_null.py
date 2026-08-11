"""Target-induced directional predictability on a process with none.

The empirical sweep shows that the accuracy a parameter-free rule reaches on a
transformed target tracks the persistence the transform induces. On real prices
that argument has an alternative reading: perhaps the smoother is extracting a
genuine slow component, and the accuracy reflects it.

This module removes that reading by running the same measurement on a driftless
Gaussian random walk, where the increments are independent by construction and
the directional predictability of the price is exactly zero in population. Any
transformed-target accuracy above chance here is produced by the transform
alone.

Nothing is fitted. The same fifteen strictly causal transforms as
`transform_sweep.py` are applied to simulated log prices, the same
parameter-free rule is scored against the same three objects, and the same
structural reference B(rho_1) is evaluated. No model is trained anywhere in
this file.

Outputs -> results/analysis/synthetic_null.csv
           results/analysis/synthetic_null_paths.csv
           results/figures/fig__synthetic_null.png
"""
import numpy as np
import pandas as pd

from config import PLOT_RC, RESULTS
from transform_sweep import smoothers

OUT = RESULTS / "analysis"
FIGS = RESULTS / "figures"

N_PATHS = 200        # independent replications
T = 4000             # observations per path, matching the empirical histories
SIGMA = 0.02         # daily log-return scale, typical of the studied assets
HORIZONS = (1, 3, 7, 14, 30)
SEED = 20260810
BURN = 200           # anchors discarded so every filter has warmed up


def paths(n=N_PATHS, T=T, sigma=SIGMA, seed=SEED):
    """Driftless Gaussian random walks in log price, one per row."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, sigma, size=(n, T))
    return np.cumsum(r, axis=1)


def _cell(lp, level, h):
    """Score the parameter-free rule on one (path, transform, horizon)."""
    n = len(lp)
    a = np.arange(max(BURN, 1), n - h)          # anchors
    d = level[a + h] - level[a]                 # transformed target
    s = h * (level[a] - level[a - 1])           # the rule: continue the increment
    ph = lp[a + h] - lp[a]
    p1 = lp[a + 1] - lp[a]
    ok = np.std(d) > 0 and np.std(s) > 0
    rho1 = float(np.corrcoef(d, s)[0, 1]) if ok else np.nan
    B = 0.5 + np.arcsin(np.clip(rho1, -1, 1)) / np.pi if ok else np.nan
    return dict(
        rho1=rho1, B=B,
        da_target=float(np.mean(np.sign(s) == np.sign(d))),
        da_price_h=float(np.mean(np.sign(s) == np.sign(ph))),
        da_price_1=float(np.mean(np.sign(s) == np.sign(p1))),
        strength=float(np.std(np.diff(level)) / np.std(np.diff(lp))),
        n=len(a))


def run():
    S = smoothers()
    P = paths()
    rows = []
    for i, lp in enumerate(P):
        for label, (family, tag, fn) in S.items():
            level = np.asarray(fn(lp, len(lp)), float)
            for h in HORIZONS:
                rows.append(dict(path=i, transform=label, family=family,
                                 horizon=h, **_cell(lp, level, h)))
    return pd.DataFrame(rows)


def summarise(R):
    g = (R.groupby(["family", "transform", "horizon"])
          .agg(strength=("strength", "mean"), rho1=("rho1", "mean"),
               da_target=("da_target", "mean"), B=("B", "mean"),
               da_price_h=("da_price_h", "mean"),
               da_price_1=("da_price_1", "mean"),
               sd_target=("da_target", "std"), n_paths=("path", "size"))
          .reset_index())
    # two-sided 95% interval across independent paths, for the h = 7 column
    g["ci_half"] = 1.96 * g.sd_target / np.sqrt(g.n_paths)
    return g


def figure(G):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(PLOT_RC)

    g7 = G[G.horizon == 7].sort_values("strength")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))

    ax = axes[0]
    ax.plot(g7.rho1, g7.da_target, "o", color="royalblue", ms=5,
            label="transformed target")
    ax.plot(g7.rho1, g7.da_price_1, "s", color="0.55", ms=4,
            label="next-day price")
    x = np.linspace(-0.05, 0.97, 200)
    ax.plot(x, 0.5 + np.arcsin(x) / np.pi, "k-", lw=1.3,
            label=r"$B(\rho_1)$")
    ax.set_xlabel(r"increment correlation $\rho_1$ of the target")
    ax.set_ylabel("directional accuracy")
    ax.set_title("(a) Simulated random walk, $h=7$")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.plot(g7.strength, g7.da_target, "o-", color="royalblue", ms=5,
            label="transformed target")
    ax.plot(g7.strength, g7.da_price_1, "s-", color="0.55", ms=4,
            label="next-day price")
    ax.axhline(0.5, color="k", ls="--", lw=1.0)
    ax.set_xlabel(r"smoothing strength $\mathrm{sd}(\Delta\ell)/\mathrm{sd}(\Delta p)$")
    ax.set_ylabel("directional accuracy")
    ax.set_title("(b) The same cells against smoothing strength")
    ax.legend(loc="center right")

    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig__synthetic_null.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = run()
    R.to_csv(OUT / "synthetic_null_paths.csv", index=False)
    G = summarise(R)
    G.to_csv(OUT / "synthetic_null.csv", index=False)
    figure(G)

    g7 = G[G.horizon == 7].sort_values("strength")
    pd.set_option("display.width", 200)
    print(g7[["transform", "strength", "rho1", "da_target", "B",
              "da_price_h", "da_price_1", "ci_half"]].round(4).to_string(index=False))
    raw = g7[g7.family == "no smoothing"]
    print(f"\nunsmoothed control, h=7: DA_target {float(raw.da_target.iloc[0]):.4f} "
          f"+/- {float(raw.ci_half.iloc[0]):.4f}")
    print(f"strongest smoother, h=7: DA_target {float(g7.da_target.max()):.4f}")
    print(f"price-1 accuracy across all transforms and horizons: "
          f"min {G.da_price_1.min():.4f} max {G.da_price_1.max():.4f} "
          f"mean {G.da_price_1.mean():.4f}")
    print(f"price-h accuracy across all transforms and horizons: "
          f"min {G.da_price_h.min():.4f} max {G.da_price_h.max():.4f} "
          f"mean {G.da_price_h.mean():.4f}")
    print(f"mean |DA_target - B| over all cells: "
          f"{float((G.da_target - G.B).abs().mean()):.4f}   "
          f"signed {float((G.da_target - G.B).mean()):+.4f}")
    print(f"\n-> {OUT / 'synthetic_null.csv'}")


if __name__ == "__main__":
    main()
