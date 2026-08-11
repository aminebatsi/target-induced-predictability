"""The full study, generalised over trend transforms and horizons, in parallel.

forecast.py answers the question for one transform (the causal Kalman) at one
horizon (h = 7). horizons.py widens the grid but only with SLOPE and LR, which
is cheap but leaves the obvious objection open: maybe the bigger models would
have behaved differently. This module closes that by running the *whole* model
set over every causal transform and every horizon.

    causal filters (4) x horizons (3) x models (11) x assets (5) x folds (5)

The leaky filters are deliberately excluded here. leakage_suite.py already shows
what they do, and spending GPU hours fitting eleven models against a target that
reads the future would only produce impressive numbers we would then have to
explain away.

WHY THIS IS A SEPARATE RUNNER

The unit of work is one (filter, horizon, model, asset, fold) cell, and cells
are completely independent: no shared state, no ordering, nothing to reduce
until the end. That makes the sweep embarrassingly parallel, which matters
because a single fit uses very little of a modern GPU.

A T4 has 15 GB and one of these fits occupies roughly 0.6 GB. The models are
small (d_model 64, a 30-step window) and the batch is 64 rows, so each kernel is
tiny and the card spends most of its time waiting for the next launch rather
than computing. Raising the batch size would fix occupancy but would also change
the optimisation and therefore the numbers, so it is not an option here. Running
many independent fits at once costs nothing in fidelity and is the right fix:
each worker is its own process with its own CUDA context, and the card
interleaves them.

Every cell writes its own small CSV under results/sweep/cells/, so the run is
resumable at cell granularity and workers never contend for a file. Kill it at
any point and rerun; it picks up exactly where it stopped.
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import ASSETS, FOLDS, HORIZONS, L, MODELS, RESULTS

OUT = RESULTS / "sweep"
CELLS = OUT / "cells"
CELLS.mkdir(parents=True, exist_ok=True)


def _cell_file(filt, h, model, asset, fold):
    return CELLS / f"{filt}__h{h}__{model}__{asset}__f{fold}.csv"


# --------------------------------------------------------------- one cell
def run_cell(item):
    """Fit and score a single (filter, horizon, model, asset, fold).

    Imports live inside the function because each worker is a fresh spawned
    process; importing torch at module scope would pay that cost even in the
    parent, which never fits anything.
    """
    filt, h, model, asset, fold = item
    fp = _cell_file(filt, h, model, asset, fold)
    if fp.exists():
        return ("cached", item, 0.0)

    t0 = time.time()
    try:
        import torch
        torch.set_num_threads(1)          # N workers must not each grab all cores

        from evaluation import fold_indices
        from filters import apply_filter
        from forecast import _norm_end, _series
        from models import predict
        from targets import build_windows

        ts, tz = FOLDS[fold]
        dates, lp = _series(asset, tz)
        if len(lp) < 700:
            return ("skip", item, time.time() - t0)
        norm_end = _norm_end(dates, ts, tz, h=h)
        if norm_end is None:
            return ("skip", item, time.time() - t0)
        trend = apply_filter(filt, lp, norm_end)
        Xtab, Xseq, anchors = build_windows(lp, trend, min_anchor=L - 1, h=h)
        idx = fold_indices(dates[anchors], ts, tz, h=h)
        if idx is None:
            return ("skip", item, time.time() - t0)
        tr, va, te = idx

        a = anchors
        d = trend[a + h] - trend[a]
        _, dp = predict(model, Xtab, Xseq, d, tr, va, te)

        a_te = a[te]
        pd.DataFrame({
            "filter": filt, "h": h, "model": model, "asset": asset, "fold": fold,
            "date": pd.DatetimeIndex(dates[a_te]),
            "y_true": d[te], "y_pred": dp,
            "y_slope": (h * (trend[a] - trend[a - 1]))[te],
            "y_true_price_h": lp[a_te + h] - lp[a_te],
            "y_true_price_1": lp[a_te + 1] - lp[a_te],
        }).to_csv(fp, index=False)
        return ("ok", item, time.time() - t0)
    except Exception as e:                                  # noqa: BLE001
        return (f"FAIL {type(e).__name__}: {str(e)[:80]}", item, time.time() - t0)


def _init_worker():
    """Keep each worker to one CPU thread and one visible GPU."""
    import torch
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")


# ------------------------------------------------------------- aggregation
def aggregate():
    """Fold every finished cell into the three-truths table."""
    files = sorted(CELLS.glob("*.csv"))
    if not files:
        raise RuntimeError("no cells on disk yet")
    D = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    def _acc(g, pred, truth):
        return float(np.mean(np.sign(g[pred]) == np.sign(g[truth])))

    rows = []
    for (filt, h, model), g in D.groupby(["filter", "h", "model"]):
        rows.append(dict(
            filter=filt, h=h, model=model, n=len(g),
            DA_trend=_acc(g, "y_pred", "y_true"),
            DA_price_h=_acc(g, "y_pred", "y_true_price_h"),
            DA_price_1=_acc(g, "y_pred", "y_true_price_1"),
        ))
    # SLOPE is recorded alongside every cell, so read it off any one model
    for (filt, h), g in D[D.model == D.model.iloc[0]].groupby(["filter", "h"]):
        rows.append(dict(
            filter=filt, h=h, model="SLOPE", n=len(g),
            DA_trend=_acc(g, "y_slope", "y_true"),
            DA_price_h=_acc(g, "y_slope", "y_true_price_h"),
            DA_price_1=_acc(g, "y_slope", "y_true_price_1"),
        ))
    T = pd.DataFrame(rows)
    T["gap"] = T.DA_trend - T.DA_price_1
    T = T.sort_values(["filter", "h", "DA_trend"], ascending=[True, True, False])
    T.to_csv(OUT / "sweep_three_truths.csv", index=False)
    D.to_csv(OUT / "sweep_predictions.csv.gz", index=False, compression="gzip")
    return T


def report(T):
    print("\n" + "=" * 74)
    print("GENERALISED RESULT: every model, every causal transform, every horizon")
    print("=" * 74)
    for h in sorted(T.h.unique()):
        s = T[T.h == h]
        print(f"\n  h = {h}")
        print(f"    mean DA on the filtered target : {s.DA_trend.mean():.4f}")
        print(f"    mean DA on the next-day return : {s.DA_price_1.mean():.4f}")
        print(f"    cells beating 0.5 on price     : "
              f"{int((s.DA_price_1 > 0.5).sum())}/{len(s)}")
    print("\n  does the free SLOPE rule beat the fitted models on the target?")
    for (filt, h), g in T.groupby(["filter", "h"]):
        sl = g[g.model == "SLOPE"]
        fit = g[g.model != "SLOPE"]
        if sl.empty or fit.empty:
            continue
        best = fit.loc[fit.DA_trend.idxmax()]
        v = float(sl.DA_trend.iloc[0])
        flag = "SLOPE" if v >= best.DA_trend else best.model
        print(f"    {filt:11} h={h:<3} SLOPE {v:.4f} vs best fitted "
              f"{best.DA_trend:.4f} ({best.model}) -> {flag}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--filters", default="causal",
                    help="'causal' (default), 'all', or a comma-separated list")
    ap.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--workers", type=int, default=0,
                    help="0 auto-sizes from GPU memory, or CPU count")
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args(argv)

    if a.aggregate_only:
        T = aggregate()
        report(T)
        return T

    from filters import CAUSAL, FILTERS
    if a.filters == "causal":
        filts = CAUSAL
    elif a.filters == "all":
        filts = list(FILTERS)
    else:
        filts = a.filters.split(",")
    hs = [int(x) for x in a.horizons.split(",")]
    mdls = a.models.split(",")

    items = [(f, h, m, asset, k)
             for f in filts for h in hs for m in mdls
             for asset in ASSETS for k in range(len(FOLDS))]
    todo = [i for i in items if not _cell_file(*i).exists()]
    workers = a.workers or suggest_workers()

    print(f"filters  {filts}")
    print(f"horizons {hs}")
    print(f"models   {len(mdls)}: {mdls}")
    print(f"cells    {len(items)} total, {len(items)-len(todo)} cached, "
          f"{len(todo)} to run")
    print(f"workers  {workers}")
    if not todo:
        T = aggregate()
        report(T)
        return T

    t0, done, failed = time.time(), 0, []
    ctx = mp.get_context("spawn")          # required: CUDA cannot survive fork
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                             initializer=_init_worker) as ex:
        futs = {ex.submit(run_cell, it): it for it in todo}
        for fut in as_completed(futs):
            status, item, dt = fut.result()
            done += 1
            if status.startswith("FAIL"):
                failed.append((item, status))
            if done % 10 == 0 or done == len(todo):
                el = time.time() - t0
                rate = done / el
                eta = (len(todo) - done) / rate / 60 if rate else float("nan")
                print(f"  {done}/{len(todo)} cells  "
                      f"{el/60:.1f} min elapsed  ETA {eta:.1f} min", flush=True)
    if failed:
        print(f"\n  {len(failed)} cells failed:")
        for item, why in failed[:10]:
            print(f"    {item}  {why}")
    print(f"\ntotal {(time.time()-t0)/60:.1f} min")
    T = aggregate()
    report(T)
    return T


def suggest_workers():
    """How many concurrent fits this machine should run.

    On a GPU the limit is memory: a CUDA context is ~300 MB and one of these
    fits peaks near 0.6 GB, so budget 0.8 GB per worker and keep 2 GB spare.
    On CPU the limit is cores, and each worker is pinned to one thread.
    """
    try:
        import torch
        if torch.cuda.is_available():
            free = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            return max(1, min(12, int((free - 2.0) / 0.8)))
    except Exception:                                       # noqa: BLE001
        pass
    return max(1, (os.cpu_count() or 2) - 1)


if __name__ == "__main__":
    raise SystemExit(0 if main() is not None else 1)
