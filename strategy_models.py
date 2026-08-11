"""Run the MCAP strategy on each of the three best-DA models and compare.

`forecast.py` ranks models by direction accuracy against the causal Kalman
trend; the top three are LR (0.771), TimeMixer (0.760) and GRU (0.757). This
module puts each of them through the IDENTICAL execution stack -- regime gate,
turn-classifier exit, per-asset vol sizing, 60/40 signal blend, book-level vol
target, 10 bps/side -- and reports what each is worth as a trading signal.

WHY THIS IS NOT A FORMALITY. This project has already found that DA against the
smoothed trend is a poor predictor of execution-stack Sharpe: the README records
`AUDUSD` scoring 0.762 DA in `forecast.py` while returning -1.99 Sharpe through
this stack. A 0.014 DA spread across these three models is well inside the range
where the ordering can reverse. Whichever model wins here, the DA table is not
evidence for it.

Nothing canonical is touched: `strategy.py` keeps using `best_model.json`, and
this module writes to its own directory.

Outputs -> results/strategy_models/
"""
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FOLDS, ORIGINAL_ASSETS, PLOT_RC, RESULTS, SIGNAL_BLEND, VOL_TARGET
from evaluation import max_drawdown, paired_tests, sharpe
from strategy import N_PERM, _components, _nz_sign, _portfolio, _regime_sign

plt.rcParams.update(PLOT_RC)
OUT = RESULTS / "strategy_models"
START_EQUITY = 100.0
YEARS = [pd.Timestamp(ts).year for ts, _ in FOLDS]
CACHE = OUT / "_components.pkl"                # refits are the whole cost
TOP_N = 4


def _top_models(n=TOP_N):
    """The n best models by mean direction accuracy, read from the forecast run.

    Derived rather than hardcoded: adding a model to config.MODELS can change
    the ranking, and a stale hardcoded list would silently compare the wrong
    three. Falls back to LR if the forecast step has not been run.
    """
    fp = RESULTS / "forecast" / "model_comparison.csv"
    if not fp.exists():
        print("  [warn] model_comparison.csv missing -- defaulting to LR")
        return ["LR"]
    comp = pd.read_csv(fp, index_col=0)
    return list(comp.loc["MEAN"].sort_values(ascending=False).head(n).index)


CANDIDATES = _top_models()


def _fit_all(model):
    """Per-(fold, asset) fits for one model, cached: the arms below are free
    once these exist, so adding an ablation arm must not force a refit."""
    store = {}
    if CACHE.exists():
        store = pickle.loads(CACHE.read_bytes())
    if model in store:
        return store[model]
    comps = {}
    for name in ORIGINAL_ASSETS:
        for c in _components(name, model):
            comps[(c["fold"], name)] = c
    store[model] = comps
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(store))
    return comps


def _by_fold(s):
    return {y: sharpe(s[(s.index >= ts) & (s.index < tz)].values)
            for y, (ts, tz) in zip(YEARS, FOLDS)}


def _stats(s):
    x = s.values
    return dict(sharpe=sharpe(x), maxdd=max_drawdown(x),
                ann_ret=float(np.mean(x) * 365), ann_vol=float(np.std(x) * np.sqrt(365)))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"== STRATEGY x MODEL ({', '.join(CANDIDATES)}); same stack, "
          f"blend {SIGNAL_BLEND:.0%}/{1 - SIGNAL_BLEND:.0%}, "
          f"book vol target {VOL_TARGET:.0%}) ==")

    rows, pooled, nulls = [], {}, {}
    for model in CANDIDATES:
        comps = _fit_all(model)
        print(f"  {model}: {len(comps)} (fold, asset) fits", flush=True)

        for arm, fn, bl in (("BLEND", lambda c, r: _nz_sign(c["w"]), True),
                            ("PRED", lambda c, r: _nz_sign(c["w"]), False),
                            ("FLIP", lambda c, r: -_nz_sign(c["w"]), False),
                            ("REGIME", lambda c, r: _regime_sign(c), False)):
            s = _portfolio(comps, fn, blend=bl)
            pooled[(model, arm)] = s
            rows.append({"model": model, "arm": arm, **_stats(s), **_by_fold(s)})

        # permutation null: random +/-1 through the identical stack
        draws = []
        for seed in range(N_PERM):
            rng = np.random.default_rng(seed)
            p = _portfolio(comps,
                           lambda c, r=rng: r.choice([-1.0, 1.0], size=len(c["w"])), rng)
            draws.append(sharpe(p.values))
        nulls[model] = np.array(draws)
        print(f"    RANDOM null mean {nulls[model].mean():+.3f}, "
              f"95% [{np.percentile(nulls[model], 2.5):+.3f}, "
              f"{np.percentile(nulls[model], 97.5):+.3f}]", flush=True)

    T = pd.DataFrame(rows).set_index(["arm", "model"])
    T["min_fold"] = T[YEARS].min(axis=1)
    T["std_fold"] = T[YEARS].std(axis=1)
    T = T.sort_index()
    T.to_csv(OUT / "model_comparison.csv")

    for arm in ("BLEND", "PRED", "FLIP", "REGIME"):
        print(f"\n{arm} -- the shipped stack, signal = each model:")
        print(T.xs(arm)[["sharpe", *YEARS, "min_fold", "std_fold", "maxdd"]]
              .round(3).to_string())

    # PRED vs FLIP is the headline proof: reversing a signal that carries real
    # directional information must hurt.
    print("\nPRED vs FLIP (paired stationary bootstrap, identical days):")
    frows = {}
    for model in CANDIDATES:
        al = pd.concat({"f": pooled[(model, "FLIP")],
                        "p": pooled[(model, "PRED")]}, axis=1).dropna()
        r = paired_tests(al["f"].values, al["p"].values)
        r["perm_p"] = float(np.mean(nulls[model] >= T.loc[("PRED", model), "sharpe"]))
        frows[model] = r
    F = pd.DataFrame(frows).T
    F.to_csv(OUT / "pred_vs_flip.csv")
    print(F.round(3).to_string())

    # REGIME is model-independent in direction, but the turn-exit's tau is
    # selected against each model's own prediction, so it moves a little. Report
    # that spread so it is not mistaken for signal.
    reg = T.xs("REGIME")["sharpe"]
    print(f"\n  [REGIME spread across models: {reg.min():+.3f} to {reg.max():+.3f} "
          f"-- direction is identical, only the turn-exit tau differs. Treat any "
          f"BLEND/PRED\n   difference smaller than this as noise.]")

    # ---------------- paired tests against the incumbent (LR) ----------------
    print("\nPAIRED BOOTSTRAP vs LR on identical days (BLEND arm):")
    prows = {}
    base = pooled[("LR", "BLEND")]
    for model in CANDIDATES[1:]:
        al = pd.concat({"lr": base, "m": pooled[(model, "BLEND")]}, axis=1).dropna()
        prows[f"{model}_vs_LR"] = paired_tests(al["lr"].values, al["m"].values)
    P = pd.DataFrame(prows).T
    P.to_csv(OUT / "paired_vs_LR.csv")
    print(P.round(3).to_string())

    # ---------------- figure ----------------
    colors = dict(zip(CANDIDATES, ("navy", "darkorange", "seagreen")))

    # --- figure 1: equity by model, PRED and FLIP ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    for ax, arm in zip(axes, ("PRED", "FLIP")):
        for model in CANDIDATES:
            eq = START_EQUITY * np.exp(pooled[(model, arm)].cumsum())
            ax.plot(eq.index, eq.values, lw=1.9, color=colors[model],
                    label=f"{model} (Sh {T.loc[(arm, model), 'sharpe']:+.2f})")
        ax.axhline(START_EQUITY, color="k", lw=0.7, ls=":")
        ax.set_xlabel("date")
        ax.set_title(f"({'a' if arm == 'PRED' else 'b'}) {arm} arm")
        ax.legend()
    axes[0].set_ylabel(f"equity (index, {START_EQUITY:.0f} at inception)")
    fig.suptitle("Portfolio equity by signal model under an identical execution stack",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "equity_pred_flip.png")
    plt.close(fig)

    # --- figure 2: PRED vs FLIP ablation + per-fold ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.0),
                                 gridspec_kw={"width_ratios": [2, 3]})
    xs = np.arange(len(CANDIDATES))
    # null band behind the bars, else it washes them out where they overlap
    lo = np.mean([np.percentile(nulls[m], 2.5) for m in CANDIDATES])
    hi = np.mean([np.percentile(nulls[m], 97.5) for m in CANDIDATES])
    a1.axhspan(lo, hi, color="gray", alpha=0.22, zorder=0,
               label=f"permutation null, 95% interval [{lo:.2f}, {hi:.2f}]")
    for i, arm in enumerate(("PRED", "FLIP")):
        vals = [T.loc[(arm, m), "sharpe"] for m in CANDIDATES]
        a1.bar(xs + (i - 0.5) * 0.38, vals, 0.38, label=arm,
               color=("navy" if arm == "PRED" else "crimson"), alpha=0.95, zorder=2)
        for x, v in zip(xs + (i - 0.5) * 0.38, vals):
            a1.text(x, v + (0.03 if v >= 0 else -0.09), f"{v:.2f}",
                    ha="center", fontsize=8, zorder=3)
    a1.axhline(0, color="k", lw=0.9)
    a1.set_xticks(xs, CANDIDATES)
    a1.set_xlabel("signal model")
    a1.set_ylabel("pooled Sharpe ratio")
    a1.set_title("(a) Signal ablation: prediction against its reversal")
    a1.legend()

    xs2 = np.arange(len(YEARS))
    w = 0.8 / len(CANDIDATES)
    for i, model in enumerate(CANDIDATES):
        a2.bar(xs2 + (i - (len(CANDIDATES) - 1) / 2) * w,
               T.loc[("PRED", model), YEARS].values, w,
               label=model, color=colors[model], alpha=0.9)
    a2.axhline(0, color="k", lw=0.9)
    a2.set_xticks(xs2, [str(y) for y in YEARS], fontsize=9)
    a2.set_xlabel("walk-forward test fold")
    a2.set_ylabel("Sharpe ratio")
    a2.set_title("(b) PRED Sharpe ratio by test fold and signal model")
    a2.legend()
    fig.tight_layout()
    fig.savefig(OUT / "ablation_pred_flip.png")
    plt.close(fig)

    best = T.xs("BLEND")["sharpe"].idxmax()
    print(f"\nBEST STRATEGY SIGNAL (BLEND pooled Sharpe) -> {best}")
    fp = RESULTS / "forecast" / "model_comparison.csv"
    if fp.exists():
        da = pd.read_csv(fp, index_col=0).loc["MEAN"]
        print("\nForecast rank vs trading rank -- is direction accuracy a useful "
              "model-selection criterion?")
        rows = [dict(model=m, mean_DA=float(da[m]),
                     BLEND_sharpe=T.loc[("BLEND", m), "sharpe"],
                     PRED_sharpe=T.loc[("PRED", m), "sharpe"])
                for m in CANDIDATES if m in da.index]
        R = pd.DataFrame(rows).sort_values("mean_DA", ascending=False)
        R.to_csv(OUT / "da_vs_sharpe.csv", index=False)
        print(R.round(3).to_string(index=False))
        if len(R) > 1:
            print(f"  DA spread {R.mean_DA.max() - R.mean_DA.min():.3f} maps to a "
                  f"Sharpe spread of "
                  f"{R.BLEND_sharpe.max() - R.BLEND_sharpe.min():.3f}")
    print(f"\n-> {OUT}")
    return T, P


if __name__ == "__main__":
    run()
