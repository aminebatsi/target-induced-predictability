"""Forecasting experiment: causal Kalman-trend target, five assets.

For every model in `config.MODELS` and every asset in `config.ASSETS`, a full
five-fold walk-forward forecast of the h-step change of the causal Kalman trend
(the filter's level state). Metrics: direction accuracy, R^2 against a
random walk, Diebold-Mariano against the same random walk, and
Pesaran-Timmermann.

The model set spans five families so that "which model wins" is not a
comparison within one hypothesis class:
  tabular on the flattened window   LR, RF, XGB, LGBM
  classical state space             ARIMA
  recurrent                         GRU, LSTM
  transformer-class sequence        DLinear, PatchTST, TimeMixer, TimeFilter,
                                    Crossformer, FEDformer
See models.py for the output-scaling rule that applies to the last group.

The model with the highest mean direction accuracy across the five assets is
written to results/forecast/best_model.json; the strategy backtest uses it as
its signal. That selection reads only the evaluation universe, which is
disjoint from the portfolio universe it is then traded on.

Outputs -> results/forecast/  (per-model CSV tables + PNG figures + best_model.json)
"""
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (ASSETS, FOLDS, H, L, MODELS, NORM_TRAIN_ONLY, PLOT_RC,
                    RESULTS)
from data import load_asset
from evaluation import dir_acc, dm_test, fold_indices, norm_end, pt_test, r2_vs_rw
from models import predict
from targets import build_windows, kalman_causal

OUT = RESULTS / "forecast"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(PLOT_RC)

MIN_ANCHOR = L - 1


def _series(name, test_end):
    """Series truncated 2 days past the fold's test end (no peeking further)."""
    df = load_asset(name)
    dates = df["date"].values
    lp = df["logprice"].values.astype(float)
    cut = np.searchsorted(dates, np.datetime64(test_end) + np.timedelta64(2, "D"))
    return dates[:cut], lp[:cut]


def _fold_arrays(name, ts, tz):
    """Build (Xtab, Xseq, d, anchors, dates, idx) for one asset/fold, or None."""
    dates, lp = _series(name, tz)
    if len(lp) < 700:
        return None
    ne = norm_end(dates, ts, tz)
    if ne is None:
        return None
    trend, _ = kalman_causal(lp, ne)
    Xtab, Xseq, anchors = build_windows(lp, trend, min_anchor=MIN_ANCHOR)
    idx = fold_indices(dates[anchors], ts, tz)
    if idx is None:
        return None
    d = trend[anchors + H] - trend[anchors]
    return Xtab, Xseq, d, anchors, dates, idx


def _seed_tag(model):
    """The seed set this model is currently configured to average over.

    Deterministic models report "-". Used to invalidate a cached summary when
    the seed configuration changes underneath it.
    """
    from models import seeds_for
    s = seeds_for(model)
    return "-" if not s else ",".join(str(x) for x in s)


def _cache_is_current(model):
    """True when the summary on disk was produced with today's seed setting."""
    fp = OUT / f"{model}_summary.csv"
    if not fp.exists():
        return False, "absent"
    try:
        got = pd.read_csv(fp).get("seeds")
    except Exception:
        return False, "unreadable"
    want = _seed_tag(model)
    if got is None:
        # Written before summaries carried a seed stamp. Deterministic models
        # are unaffected by seeds so their old files are still valid; anything
        # seeded has to be refitted because we cannot tell what produced it.
        return (want == "-"), ("pre-stamp, deterministic" if want == "-"
                               else "pre-stamp, seeds unknown")
    got = str(got.iloc[0])
    return got == want, f"cached seeds {got} vs configured {want}"


def _cell(args):
    """Fit one (model, asset, fold) and return everything the caller needs.

    Module level and picklable so a process pool can run cells concurrently.
    The unit of work is deliberately the cell rather than the model: within a
    model the 25 cells are independent, which is where the parallelism is.
    """
    model, name, k = args
    import torch
    torch.set_num_threads(1)          # workers must not each grab every core

    ts, tz = FOLDS[k]
    fa = _fold_arrays(name, ts, tz)
    if fa is None:
        return None
    Xtab, Xseq, d, anchors, dates, (tr, va, te) = fa
    _, dp = predict(model, Xtab, Xseq, d, tr, va, te)
    a_te = anchors[te]
    _, lp_full = _series(name, tz)
    return dict(
        model=model, asset=name, fold=k, ts=ts, tz=tz,
        dir=dir_acc(d[te], dp), r2=r2_vs_rw(d[te], dp),
        dates=dates[a_te], anchors=a_te, logprice=lp_full[a_te],
        y_true=d[te], y_pred=dp,
        price_h=lp_full[a_te + H] - lp_full[a_te],
        price_1=lp_full[a_te + 1] - lp_full[a_te])


def _run_cells(model, workers):
    """All 25 cells for one model, in parallel when asked."""
    jobs = [(model, name, k) for name in ASSETS for k in range(len(FOLDS))]
    if workers <= 1:
        return [r for r in (_cell(j) for j in jobs) if r is not None]
    ctx = mp.get_context("spawn")     # CUDA cannot survive fork
    out = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        for r in ex.map(_cell, jobs):
            if r is not None:
                out.append(r)
    return out


def run_model(model, workers=None):
    """Walk-forward forecast for one model over the five assets.

    `workers` > 1 fits the 25 (asset, fold) cells concurrently. Cells share no
    state and each is independently seeded, so the results are identical to the
    serial path; only the wall clock changes. Defaults to FORECAST_WORKERS, or
    serial when that is unset, so existing behaviour is unchanged unless asked.
    """
    if workers is None:
        workers = int(os.environ.get("FORECAST_WORKERS", "1"))
    if workers <= 0:
        from sweep import suggest_workers
        workers = suggest_workers()
    if workers > 1:
        print(f"  [{model}] fitting {len(ASSETS)*len(FOLDS)} cells "
              f"on {workers} workers", flush=True)

    cells = _run_cells(model, workers)
    by_key = {(c["asset"], c["fold"]): c for c in cells}

    rows, sig_rows, pred_rows = [], [], []
    for name in ASSETS:
        e_m, e_rw, d_all, p_all = [], [], [], []
        for k in range(len(FOLDS)):
            c = by_key.get((name, k))
            if c is None:
                continue
            dp, dtrue = c["y_pred"], c["y_true"]
            rows.append(dict(asset=name, fold=k, dir=c["dir"], r2_vs_rw=c["r2"]))
            # Row-level record: everything needed to recompute any metric, or to
            # build new plots later, without refitting a single model.
            pred_rows.append(pd.DataFrame({
                "model": model, "asset": name, "fold": k,
                "fold_start": c["ts"], "fold_end": c["tz"],
                "date": pd.DatetimeIndex(c["dates"]),
                "anchor": c["anchors"],
                "logprice": c["logprice"],
                "y_true": dtrue,                    # realised h-step trend change
                "y_pred": dp,                       # model forecast of the same
                "y_true_price_h": c["price_h"],
                "y_true_price_1": c["price_1"],
            }))
            e_m.append(dtrue - dp)
            e_rw.append(dtrue)                 # random walk predicts no change
            d_all.append(dtrue); p_all.append(dp)
        if not e_m:
            continue
        e_m, e_rw = map(np.concatenate, (e_m, e_rw))
        d_all, p_all = np.concatenate(d_all), np.concatenate(p_all)
        dm, p_rw = dm_test(e_m, e_rw)
        z, pz = pt_test(np.sign(d_all), np.sign(p_all))
        sig_rows.append(dict(asset=name, DM_vs_RW=dm, p_RW=p_rw, PT_z=z, PT_p=pz))
        print(f"  [{model}] {name:7} dir="
              f"{np.mean([r['dir'] for r in rows if r['asset'] == name]):.3f} "
              f"DMvRW={dm:+.1f} PTp={pz:.3f}", flush=True)

    F = pd.DataFrame(rows)
    S = pd.DataFrame(sig_rows)
    F.to_csv(OUT / f"{model}_folds.csv", index=False)
    S.to_csv(OUT / f"{model}_significance.csv", index=False)
    if pred_rows:
        pd.concat(pred_rows, ignore_index=True).to_csv(
            OUT / f"{model}_predictions.csv", index=False)
    g = F.groupby("asset").agg(dir_mean=("dir", "mean"), dir_sd=("dir", "std"),
                               r2_mean=("r2_vs_rw", "mean")).reset_index()
    # Stamp the seed set into the summary. `resume` compares this against the
    # current configuration, so raising DL_SEEDS from one seed to three
    # invalidates the cached SOTA runs instead of silently mixing a one-seed
    # result into a three-seed table.
    g["seeds"] = _seed_tag(model)
    g.to_csv(OUT / f"{model}_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    gg = g.sort_values("dir_mean", ascending=False)
    ax.bar(range(len(gg)), gg["dir_mean"], yerr=gg["dir_sd"], capsize=3,
           color="royalblue", alpha=0.85)
    ax.axhline(0.5, color="k", ls="--", lw=1.1, label="coin flip (50%)")
    ax.set_xticks(range(len(gg)), gg["asset"], rotation=30, ha="right")
    ax.set_xlabel("asset")
    ax.set_ylabel(f"direction accuracy at $h={H}$")
    ax.set_title(f"{model}: direction accuracy against the causal Kalman trend "
                 f"(mean {g['dir_mean'].mean():.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"{model}_accuracy.png")
    plt.close(fig)
    return g


def build_comparison():
    """Assemble the cross-model table from the per-model summaries on disk.

    Kept separate from `run_all_models` so a long sweep is RESUMABLE: each model
    writes its own `<model>_summary.csv` as it finishes, and this rebuilds the
    comparison, `best_model.json` and the figure from whatever is present. A run
    interrupted after model 7 of 10 therefore loses nothing -- rerun the missing
    models individually and call this.
    """
    per_model, missing = {}, []
    for model in MODELS:
        fp = OUT / f"{model}_summary.csv"
        if fp.exists():
            per_model[model] = pd.read_csv(fp).set_index("asset")["dir_mean"]
        else:
            missing.append(model)
    if not per_model:
        raise RuntimeError("no per-model summaries found; run the models first")
    if missing:
        print(f"  [warn] no results yet for {', '.join(missing)} -- "
              f"comparison covers {len(per_model)}/{len(MODELS)} models")

    comp = pd.DataFrame(per_model)
    comp.loc["MEAN"] = comp.mean()
    comp.to_csv(OUT / "model_comparison.csv")

    mean_da = comp.loc["MEAN"]
    best = str(mean_da.idxmax())
    json.dump(dict(best_model=best, mean_da=float(mean_da.max()),
                   mean_da_by_model={m: float(v) for m, v in mean_da.items()}),
              open(OUT / "best_model.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    order = mean_da.sort_values(ascending=False)
    ax.bar(range(len(order)), order.values,
           color=["seagreen" if m == best else "#8aa" for m in order.index], alpha=0.9)
    for i, (m, v) in enumerate(order.items()):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="k", ls="--", lw=1.1, label="coin flip (50%)")
    ax.set_xticks(range(len(order)), order.index, rotation=20, ha="right")
    ax.set_xlabel("model")
    ax.set_ylabel(f"mean direction accuracy at $h={H}$")
    ax.set_title("Direction accuracy against the causal Kalman trend, "
                 f"averaged over {len(ASSETS)} assets and {len(FOLDS)} walk-forward folds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png")
    plt.close(fig)

    print("\nMODEL COMPARISON (mean DA across assets):")
    print(comp.round(3).to_string())
    print(f"\nBEST MODEL -> {best}  (mean DA {mean_da.max():.3f})")
    return comp, best


def run_all_models(resume=False):
    """Run every model in MODELS, then build the comparison.

    `resume=True` skips models whose summary is already on disk AND was produced
    with the current seed configuration, so an interrupted sweep continues
    without repeating hours of fitting, while a change to DL_SEEDS correctly
    forces the affected models to be refitted.
    """
    for model in MODELS:
        if resume:
            ok, why = _cache_is_current(model)
            if ok:
                print(f"== forecast: {model} (cached, skipped) ==", flush=True)
                continue
            if why != "absent":
                print(f"== forecast: {model} (cache stale: {why}) ==", flush=True)
        print(f"== forecast: {model} ==", flush=True)
        run_model(model)
    return build_comparison()


def plot_windows(model, asset="LTC", n=12, nrow=3, ncol=4):
    """Prediction-vs-truth panels for one asset on the last fold.

    `n` windows evenly spaced across the test fold, so the sample is systematic
    rather than a hand-picked selection of favourable cases.
    """
    ts, tz = FOLDS[-1]
    fa = _fold_arrays(asset, ts, tz)
    if fa is None:
        print(f"  plot_windows skipped ({asset}: insufficient data)")
        return
    Xtab, Xseq, d, anchors, dates, (tr, va, te) = fa
    dates2, lp = _series(asset, tz)
    norm_end = _norm_end(dates2, ts, tz)
    if norm_end is None:
        return
    trend, _ = kalman_causal(lp, norm_end)
    _, dp = predict(model, Xtab, Xseq, d, tr, va, te)
    a_te = anchors[te]
    lvl = trend[a_te] + dp
    ok = np.sign(dp) == np.sign(d[te])
    pos = np.linspace(0, len(a_te) - 1, n).astype(int)

    # the panels themselves, as a table, so the figure can be re-drawn or audited
    pd.DataFrame({"date": pd.DatetimeIndex(dates[a_te[pos]]),
                  "trend_now": trend[a_te[pos]],
                  "trend_true_t_plus_h": trend[a_te[pos] + H],
                  "trend_pred_t_plus_h": lvl[pos],
                  "y_true": d[te][pos], "y_pred": dp[pos],
                  "correct": ok[pos]}).to_csv(
        OUT / f"windows_{asset}_{model}.csv", index=False)

    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.6 * nrow))
    for ax, kk in zip(axes.ravel(), pos):
        t = a_te[kk]
        seg = np.arange(t - 21, t + H + 1)
        ax.plot(dates[seg], lp[seg], color=".6", lw=0.8)
        sp = seg[seg <= t]
        ax.plot(dates[sp], trend[sp], color="crimson", lw=1.3)
        ax.scatter([dates[t + H]], [trend[t + H]], color="crimson", s=28, zorder=5)
        ax.scatter([dates[t + H]], [lvl[kk]], color="royalblue", marker="D", s=30, zorder=6)
        ax.plot([dates[t], dates[t + H]], [trend[t], lvl[kk]], "--", color="royalblue", lw=1.2)
        ax.axvline(dates[t], color="gray", ls=":", lw=0.8)
        ax.set_title(f"{pd.Timestamp(dates[t]).date()} "
                     f"({'correct' if ok[kk] else 'incorrect'})",
                     fontsize=7, color="green" if ok[kk] else "firebrick")
        ax.tick_params(labelsize=6)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(f"{asset}: {model} forecasts of the $h={H}$-step Kalman-trend change "
                 f"(blue) against the realised trend (red)\n"
                 f"{n} windows evenly spaced across the final test fold; "
                 f"direction accuracy over the whole fold {ok.mean():.1%}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / f"pred{n}_{asset}_{model}.png")
    plt.close(fig)
    print(f"  pred{n}_{asset}_{model}.png (dir {ok.mean():.3f})")


def main(resume=False):
    comp, best = run_all_models(resume=resume)
    plot_windows(best, n=12)
    return comp, best


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg in MODELS:                       # one model only (resumable sweeps)
        run_model(arg)
    elif arg == "compare":                  # rebuild the table from what exists
        build_comparison()
    elif arg == "resume":                   # skip models already on disk
        main(resume=True)
    else:
        main()
