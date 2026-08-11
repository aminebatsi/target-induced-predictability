"""Measure how much of the GPU one fit actually uses, and size the worker pool.

The observation that prompted this: on a T4 a training run sits at roughly
0.6 GB of 15 GB. That is not a memory problem to be solved by making the model
bigger. It is a symptom of how this workload is shaped. The nets are small
(d_model 64 over a 30-step window) and the batch is 64 rows, so every kernel is
tiny and the card spends most of its time between launches rather than inside
them. Enlarging the batch would raise occupancy but would also change the
optimisation, and therefore every number in the study, so it is off the table.

The fix that costs nothing in fidelity is concurrency: run many independent
fits at once. This script measures the per-fit footprint on the actual device
and turns it into a worker count for sweep.py.
"""
import time

import numpy as np


def probe(model="TimeMixer", asset="LTC", fold=2):
    import torch
    if not torch.cuda.is_available():
        print("No CUDA device. sweep.py will size its pool from CPU cores "
              f"({__import__('os').cpu_count()} available).")
        return None

    from config import FOLDS, L
    from evaluation import fold_indices
    from filters import apply_filter
    from forecast import _norm_end, _series
    from models import predict
    from targets import build_windows

    dev = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(dev).total_memory / 1024 ** 3
    print(f"device : {torch.cuda.get_device_name(dev)}")
    print(f"memory : {total:.1f} GB total")

    ts, tz = FOLDS[fold]
    dates, lp = _series(asset, tz)
    norm_end = _norm_end(dates, ts, tz)
    if norm_end is None:
        raise RuntimeError("probe fold does not meet the purged training requirement")
    trend = apply_filter("kalman", lp, norm_end)
    Xtab, Xseq, anchors = build_windows(lp, trend, min_anchor=L - 1)
    tr, va, te = fold_indices(dates[anchors], ts, tz)
    d = trend[anchors + 7] - trend[anchors]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time()
    predict(model, Xtab, Xseq, d, tr, va, te)
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated(dev) / 1024 ** 3
    reserved = torch.cuda.max_memory_reserved(dev) / 1024 ** 3

    print(f"\none {model} cell (all seeds): {dt:.1f} s")
    print(f"  peak allocated : {peak:.2f} GB")
    print(f"  peak reserved  : {reserved:.2f} GB")
    print(f"  utilisation    : {100*reserved/total:.1f}% of the card")

    per_worker = max(0.8, reserved + 0.35)      # + CUDA context per process
    n = max(1, min(12, int((total - 2.0) / per_worker)))
    print(f"\nbudget {per_worker:.2f} GB per worker "
          f"(fit + ~0.35 GB CUDA context), keeping 2 GB spare")
    print(f"suggested workers: {n}")
    print(f"\n  python sweep.py --workers {n}")
    print("\nExpect sub-linear scaling. Concurrency hides launch latency, so the")
    print("first few workers help a lot and later ones less as the SMs fill.")
    return n


if __name__ == "__main__":
    probe()
