"""Target transform and dataset construction.

kalman_causal : local-linear-trend Kalman filter [level, velocity] -- the sole
                causal trend/target transform used across this package
                (forecast.py, leakage.py, strategy.py, strategy_crypto.py,
                strategy_mcap5.py).

Strictness rule: the Kalman's volatility-adaptive measurement noise is
normalized by a constant computed from the TRAINING partition only
(`norm_end`). A full-sample median is a global leak, while a median pinned to
the test start still lets validation-period volatility affect preprocessing.
Callers that fit or validate models therefore resolve the purged split first
and pin `norm_end` to the final training observation.
"""
import numpy as np
import pandas as pd

from config import H, KALMAN, L


def kalman_causal(p, norm_end=None, q_level=None, q_slope=None, r=None, return_cache=False):
    """Causal Kalman [level, velocity]. `norm_end`: vol-normalization uses p[:norm_end] only."""
    q_level = KALMAN["q_level"] if q_level is None else q_level
    q_slope = KALMAN["q_slope"] if q_slope is None else q_slope
    r = KALMAN["r"] if r is None else r
    p = np.asarray(p, float)
    n = len(p)
    if norm_end is None:
        norm_end = n
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Hm = np.array([[1.0, 0.0]])
    Q = np.diag([q_level, q_slope])
    ret = np.diff(p, prepend=p[0])
    rv = pd.Series(ret ** 2).ewm(span=20).mean().values
    rv = rv / (np.median(rv[20:max(norm_end, 40)]) + 1e-18)
    x = np.array([p[0], 0.0])
    P = np.eye(2)
    level = np.empty(n)
    slope = np.empty(n)
    xs_pred, Ps_pred, xs_filt, Ps_filt = [], [], [], []
    for t in range(n):
        x = F @ x
        P = F @ P @ F.T + Q
        xs_pred.append(x.copy()); Ps_pred.append(P.copy())
        R_t = r * rv[t]
        S = float((Hm @ P @ Hm.T)[0, 0] + R_t)
        K = (P @ Hm.T) / S
        x = x + K.flatten() * (p[t] - float((Hm @ x)[0]))
        P = (np.eye(2) - K @ Hm) @ P
        xs_filt.append(x.copy()); Ps_filt.append(P.copy())
        level[t], slope[t] = x[0], x[1]
    if return_cache:
        return level, slope, (F, xs_pred, Ps_pred, xs_filt, Ps_filt)
    return level, slope


def build_windows(lp, trend, min_anchor=None, extra_chans=None, h=None):
    """Causal supervised set: L-day window x [trend, logprice, return] channels
    (+ optional extra channels, e.g. for the turn classifier).

    Anchor convention: anchors start at L-1; callers that need SMA-200 warm-up
    filter the TEST anchors (te >= 200) after the fold split, so the pre-test pool
    keeps its early anchors. Targets are built by the caller as
    trend[anchors+H] - trend[anchors] (stationary CHANGE)."""
    h = H if h is None else h
    ret = np.diff(lp, prepend=lp[0])
    chans = [trend, lp, ret] + (list(extra_chans) if extra_chans else [])
    chans = np.stack(chans, axis=1).astype(np.float32)
    start = L - 1 if min_anchor is None else max(L - 1, min_anchor)
    anchors = np.arange(start, len(lp) - h)
    Xseq = np.stack([chans[t - L + 1:t + 1] for t in anchors])
    return Xseq.reshape(len(anchors), -1).astype(np.float64), Xseq, anchors
