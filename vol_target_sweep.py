"""Is the book-level vol target real, or is it a parameter the backtest was
fitted to? A standalone audit of the overlay added in strategy.py.

The overlay was chosen after searching several families of execution changes
against the SAME five test folds, which is a multiple-comparison problem.
Reporting only the winner's Sharpe would be exactly the mistake this file
exists to catch. So the deciding evidence is not the peak number, it is the
SHAPE of the parameter surface plus a set of controls that a spurious result
cannot pass.

Two candidate overlays are swept side by side:

  REJECTED  trend-quality filter -- stand down in an asset whose trailing
            |sum r| / sum|r| is below a threshold. Its best cell (win 30,
            thr 0.25) posts a pooled Sharpe of ~1.40, the highest of anything
            tried. Its surface is a lone spike: immediate neighbours drop to
            ~0.9-1.25 and the rest of the grid sits at or below the 0.99
            baseline with no coherent gradient. That is the signature of
            reading the maximum off a ~56-cell grid scored on the test folds,
            so it is NOT shipped.

  SHIPPED   book vol target -- size the whole book against its own trailing
            realized vol. Every cell of the (win x target) grid beats the
            baseline, monotonically in both axes: a plateau, not a spike. The
            choice of (0.15, 60) is therefore not load-bearing.

Controls on the shipped overlay (`controls.csv`):

  constant scale   the mean throttle applied as a CONSTANT. Sharpe is
                   scale-invariant, so this MUST reproduce the baseline
                   exactly. It does -- proving the gain is not "trade smaller".
  shuffled scale   the same throttle values, permuted in time: the size
                   distribution is preserved and only the timing is destroyed.
                   The gain does not survive, so timing is the whole mechanism.
  per-fold reset   the vol estimate restarted at each fold boundary instead of
                   carrying across it. This is a HANDICAP, not a fairer test --
                   the five test folds are contiguous calendar time, so a live
                   book would carry that history -- but it bounds how much of
                   the effect depends on the cross-boundary window, and the
                   answer is: a lot. Reported rather than hidden.

Outputs -> results/vol_target/
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (COST, ORIGINAL_ASSETS, PLOT_RC, RESULTS, VOL_TARGET,
                    VOL_TARGET_WIN)
from evaluation import net_daily, sharpe
from strategy import (_aggregate, _best_model, _components, _nz_sign, _pos_net,
                      _position, book_scale)

plt.rcParams.update(PLOT_RC)
OUT = RESULTS / "vol_target"

TGT_GRID = [0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30]
WIN_GRID = [30, 45, 60, 90, 126]
TQ_WINS = [15, 20, 25, 30, 35, 40, 50, 60]
TQ_THRS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def _trend_quality(r, win):
    """Trailing |sum r| / sum|r|, shifted one day so day t uses data to t-1."""
    s = pd.Series(r)
    return (s.rolling(win).sum().abs() / (s.abs().rolling(win).sum() + 1e-12)
            ).shift(1).bfill().values


def _pooled(comps, tq=None, scale_fn=None):
    """Pooled PRED book. `tq`=(win, thr) applies the rejected filter; `scale_fn`
    maps the un-throttled book to a daily scale series."""
    pos = {}
    for key, c in comps.items():
        p = _position(c, _nz_sign(c["w"]))
        if tq is not None:
            p = p * (_trend_quality(c["r"], tq[0]) >= tq[1])
        pos[key] = p

    def price(scale=None):
        per_fold = {}
        for (k, a), p in pos.items():
            c = comps[(k, a)]
            sc = 1.0 if scale is None else scale.reindex(c["dates"]).ffill().fillna(1.0).values
            per_fold.setdefault(k, {})[a] = pd.Series(
                net_daily(p * sc, c["r"], COST, c["fund"]), index=c["dates"])
        return _aggregate(per_fold)

    raw = price()
    return raw if scale_fn is None else price(scale_fn(raw))


def _vt(target, win):
    return lambda pooled: book_scale(pooled, target, win)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    model = _best_model()
    print(f"== VOL-TARGET AUDIT (parameter stability + controls; model = {model}) ==")
    comps = {}
    for name in ORIGINAL_ASSETS:
        for c in _components(name, model):
            comps[(c["fold"], name)] = c
        print(f"  {name} done", flush=True)

    base = sharpe(_pooled(comps).values)
    print(f"\nbaseline (no overlay): pooled Sharpe {base:+.3f}")

    # ---------------- surface 1: SHIPPED book vol target ----------------
    S1 = pd.DataFrame(index=WIN_GRID, columns=TGT_GRID, dtype=float)
    for w in WIN_GRID:
        for t in TGT_GRID:
            S1.loc[w, t] = sharpe(_pooled(comps, scale_fn=_vt(t, w)).values)
    S1.index.name = "win"
    S1.to_csv(OUT / "surface_vol_target.csv")
    print(f"\nBOOK VOL TARGET -- pooled Sharpe over (win x target)\n{S1.round(3).to_string()}")
    print(f"  cells beating baseline: {int((S1.values > base).sum())}/{S1.size}"
          f"   range [{S1.values.min():+.2f}, {S1.values.max():+.2f}]")

    # ---------------- surface 2: REJECTED trend-quality filter ----------------
    S2 = pd.DataFrame(index=TQ_WINS, columns=TQ_THRS, dtype=float)
    for w in TQ_WINS:
        for t in TQ_THRS:
            S2.loc[w, t] = sharpe(_pooled(comps, tq=(w, t)).values)
    S2.index.name = "win"
    S2.to_csv(OUT / "surface_trend_quality.csv")
    print(f"\nTREND-QUALITY FILTER (rejected) -- pooled Sharpe over (win x thr)"
          f"\n{S2.round(3).to_string()}")
    print(f"  cells beating baseline: {int((S2.values > base).sum())}/{S2.size}"
          f"   range [{S2.values.min():+.2f}, {S2.values.max():+.2f}]")
    iw, it = np.unravel_index(np.nanargmax(S2.values), S2.shape)
    nb = [S2.values[i, j] for i in (iw - 1, iw, iw + 1) for j in (it - 1, it, it + 1)
          if 0 <= i < S2.shape[0] and 0 <= j < S2.shape[1] and (i, j) != (iw, it)]
    print(f"  best cell (win={S2.index[iw]}, thr={S2.columns[it]}) = {S2.values[iw, it]:+.2f}; "
          f"immediate neighbours mean {np.mean(nb):+.2f} -> spike, not plateau")

    # ---------------- controls ----------------
    rows = [dict(control="baseline (no overlay)", sharpe=base)]
    shipped = _pooled(comps, scale_fn=_vt(VOL_TARGET, VOL_TARGET_WIN))
    rows.append(dict(control=f"book vol target {VOL_TARGET:.0%}/{VOL_TARGET_WIN}d (SHIPPED)",
                     sharpe=sharpe(shipped.values)))

    kbar = float(book_scale(_pooled(comps), VOL_TARGET, VOL_TARGET_WIN).mean())
    rows.append(dict(control=f"CTRL constant scale ({kbar:.3f}) -- must equal baseline",
                     sharpe=sharpe(_pooled(
                         comps, scale_fn=lambda p: pd.Series(kbar, index=p.index)).values)))

    def shuffled(seed):
        def f(p):
            k = book_scale(p, VOL_TARGET, VOL_TARGET_WIN).values.copy()
            np.random.default_rng(seed).shuffle(k)
            return pd.Series(k, index=p.index)
        return f

    sh = [sharpe(_pooled(comps, scale_fn=shuffled(s)).values) for s in range(8)]
    rows.append(dict(control="CTRL scale shuffled in time (mean of 8 seeds)",
                     sharpe=float(np.mean(sh))))

    fold_dates = {}
    for (k, _), c in comps.items():
        fold_dates.setdefault(k, c["dates"])

    def per_fold_reset(p):
        parts = []
        for k in sorted(fold_dates):
            seg = p.reindex(fold_dates[k]).dropna()
            parts.append(book_scale(seg, VOL_TARGET, VOL_TARGET_WIN))
        return pd.concat(parts).sort_index()

    rows.append(dict(control="CTRL vol estimate reset each fold (handicap)",
                     sharpe=sharpe(_pooled(comps, scale_fn=per_fold_reset).values)))

    C = pd.DataFrame(rows).set_index("control")
    C.to_csv(OUT / "controls.csv")
    print(f"\nCONTROLS\n{C.round(3).to_string()}")

    # ---------------- figure: the two surfaces side by side ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for ax, (S, title, xl) in zip(axes, [
            (S1, f"(a) Book volatility target: all {S1.size} cells exceed the\n"
                 f"baseline of {base:+.2f} (plateau)", "annualised volatility target"),
            (S2, "(b) Trend-quality filter: isolated maximum at (30, 0.25),\n"
                 "no coherent gradient (spike)", "trend-efficiency threshold")]):
        v = np.nanmax(np.abs(S.values - base))
        im = ax.imshow(S.values.astype(float), cmap="RdYlGn", aspect="auto",
                       vmin=base - v, vmax=base + v)
        ax.set_xticks(range(S.shape[1]), S.columns)
        ax.set_yticks(range(S.shape[0]), S.index)
        ax.set_xlabel(xl)
        ax.set_ylabel("trailing window (days)")
        ax.set_title(title, fontsize=10)
        ax.grid(False)
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                ax.text(j, i, f"{S.values[i, j]:.2f}", ha="center", va="center", fontsize=7.5)
        fig.colorbar(im, ax=ax, label="pooled Sharpe ratio", shrink=0.85)
    fig.suptitle("Parameter-stability surfaces for two candidate execution overlays",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "parameter_surfaces.png")
    plt.close(fig)
    print(f"\n-> {OUT}")
    return S1, S2, C


if __name__ == "__main__":
    run()
