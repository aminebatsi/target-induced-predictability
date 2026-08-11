"""Replay the portfolio from cached execution-stack components.

Everything here reads the cached per-(fold, asset) components written by
`strategy_models.py`, which hold the forecast `w`, the regime state `bull`, the
turn probability `c_te`, the validation-selected threshold `tau`, the per-asset
volatility scale `vsc`, the realised next-day return `r` and the funding flag.

No model is refitted by any module that imports this one. Every arm is a
different function of the same cached ingredients, which is what makes the
comparisons like-for-like: two arms can differ only in the directional signal
or in a friction, never in the fit underneath them.

The shipped cache was written under numpy >= 2, which renamed `numpy.core` to
`numpy._core`. `install_numpy2_shim()` aliases the old module tree onto the new
names so the pickle also loads under numpy 1.x.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import SIGNAL_BLEND

ROOT = Path(__file__).resolve().parent
# The manuscript's numbers come from the Colab run; prefer that cache and fall
# back to a local one so the battery still runs in a fresh checkout.
CACHES = [ROOT / "artifacts" / "strategy_components.pkl",
          ROOT / "results" / "strategy_models" / "_components.pkl"]
OUT = ROOT / "results" / "analysis"


def install_numpy2_shim():
    for name in ("numpy._core", "numpy._core.multiarray", "numpy._core.umath",
                 "numpy._core.numeric", "numpy._core._multiarray_umath"):
        if name in sys.modules:
            continue
        legacy = name.replace("numpy._core", "numpy.core")
        try:
            __import__(legacy)
            sys.modules[name] = sys.modules[legacy]
        except ImportError:
            pass


def load_components(path=None):
    """-> {model: {(fold, asset): component dict}}"""
    install_numpy2_shim()
    for fp in ([Path(path)] if path else CACHES):
        if fp.exists():
            print(f"[replay] components cache: {fp}")
            return pickle.loads(fp.read_bytes())
    raise FileNotFoundError("no _components.pkl found; run strategy_models.py first")


# ------------------------------------------------------------------ arms
def nz_sign(x):
    s = np.sign(np.asarray(x, float))
    s[s == 0] = 1.0
    return s


def gate(s, bull):
    return np.where(bull, np.maximum(s, 0.0), np.minimum(s, 0.0))


def regime_sign(c):
    return np.where(c["bull"], 1.0, -1.0)


SIGNS = {
    "PRED": lambda c: nz_sign(c["w"]),
    "FLIP": lambda c: -nz_sign(c["w"]),
    "LONG": lambda c: np.ones(len(c["w"])),
    "REGIME": regime_sign,
}


def position(c, s, scale=None, gross=None):
    """The execution stack: regime gate -> turn exit -> per-asset vol sizing ->
    optional book scale -> optional gross-exposure multiplier.

    `gross` is the only addition relative to strategy.py's `_position`. It is a
    per-arm constant, so it can never change WHEN the arm is in the market or on
    which side; it only rescales how large the arm's book is. Sharpe is
    invariant to a constant leverage factor on a fully invested book, so this
    matters only through the transaction-cost and funding terms, which is
    precisely the channel the exposure-matched comparison has to price.
    """
    g = gate(s, c["bull"])
    g = np.where((g != 0) & (c["c_te"] >= c["tau"]), 0.0, g)
    pos = g * c["vsc"]
    if gross is not None:
        pos = pos * gross
    return pos if scale is None else pos * scale


def net_daily(pos, r, cost, fund_ann):
    """Daily net return: fee on every position CHANGE, funding on short
    exposure, final position liquidated (fee charged). Identical to
    evaluation.net_daily; duplicated so this module stands alone."""
    pos = np.asarray(pos, float)
    dpos = np.abs(np.diff(pos, prepend=0.0))
    net = pos * r - cost * dpos - (fund_ann / 365.0) * np.maximum(-pos, 0.0)
    net = net.copy()
    net[-1] -= cost * abs(pos[-1])
    return net


def aggregate(per_fold):
    """Equal-weight the assets within each fold, then chain the folds in time."""
    series = [pd.concat(dd, axis=1, sort=True).mean(axis=1, skipna=True).dropna()
              for _, dd in sorted(per_fold.items())]
    return pd.concat(series).sort_index()


def book_scale(pooled, target=0.15, win=60):
    rv = pooled.rolling(win).std().shift(1) * np.sqrt(365)
    return (target / (rv + 1e-12)).clip(0.0, 1.0).fillna(1.0)


def raw_positions(comps, sign_fn, gated=True, blend=False, reverse=False, gross=None,
                  turn_exit=True):
    """Per-(fold, asset) position path BEFORE the book-level overlay.

    `gated=False` drops the SMA-200 regime gate, which is the component that
    makes PRED and FLIP carry different exposure: under the gate the two arms
    are in the market on complementary days, so no rescaling can align them.
    Without it the arms hold the identical |position| path every day and differ
    only in sign, which is what an exposure-matched reversal requires.

    `reverse=True` negates the finished position instead of the signal. That
    keeps the shipped stack's exposure schedule byte for byte and reverses only
    its direction.

    `turn_exit=False` deletes the turn-classifier exit outright: no probability,
    no threshold, no fallback. It is not replaced by anything.
    """
    out = {}
    for key, c in comps.items():
        s = sign_fn(c)
        if gated:
            g = gate(s, c["bull"])
        else:
            g = np.asarray(s, float)
        if turn_exit:
            g = np.where((g != 0) & (c["c_te"] >= c["tau"]), 0.0, g)
        pos = g * c["vsc"]
        if blend:
            gr = gate(regime_sign(c), c["bull"])
            if turn_exit:
                gr = np.where((gr != 0) & (c["c_te"] >= c["tau"]), 0.0, gr)
            pos = SIGNAL_BLEND * pos + (1.0 - SIGNAL_BLEND) * gr * c["vsc"]
        if reverse:
            pos = -pos
        if gross is not None:
            pos = pos * gross
        out[key] = pos
    return out


def price_positions(comps, pos_map, cost=1e-3, scale_ser=None, funding=True):
    out = {}
    for (k, a), c in comps.items():
        pos = pos_map[(k, a)]
        if scale_ser is not None:
            pos = pos * scale_ser.reindex(c["dates"]).ffill().fillna(1.0).values
        fund = c["fund"] if funding else 0.0
        out.setdefault(k, {})[a] = pd.Series(
            net_daily(pos, c["r"], cost, fund), index=c["dates"])
    return aggregate(out)


def portfolio(comps, sign_fn, cost=1e-3, gross=None, scaled=True, funding=True,
              gated=True, blend=False, reverse=False, shared_scale=None,
              turn_exit=True):
    """Equal-weight book under one directional signal, book-vol-targeted.

    Two passes, as in strategy.py: the book scale is estimated on the unscaled
    book, then every leg is re-priced at the throttled size so the resizing pays
    the same per-side fee as any other position change.

    `shared_scale` forces both arms of a matched comparison to run the SAME
    overlay path. Each arm estimating its own overlay from its own returns would
    reintroduce an exposure difference through the back door, which is exactly
    what the matched test exists to remove.
    """
    pos_map = raw_positions(comps, sign_fn, gated, blend, reverse, gross, turn_exit)
    pooled = price_positions(comps, pos_map, cost, None, funding)
    if not scaled:
        return pooled
    ks = book_scale(pooled) if shared_scale is None else shared_scale
    return price_positions(comps, pos_map, cost, ks, funding)


def exposures(comps, sign_fn, gross=None, scaled=True, gated=True, blend=False,
              reverse=False, shared_scale=None, turn_exit=True):
    """Mean gross, net and turnover of an arm's per-asset positions, priced at
    the same book scale the arm's returns are priced at. Turnover counts the
    opening trade, because the cost model charges it."""
    pos_map = raw_positions(comps, sign_fn, gated, blend, reverse, gross, turn_exit)
    ks = None
    if scaled:
        ks = shared_scale if shared_scale is not None else \
            book_scale(price_positions(comps, pos_map))

    g, n, t = [], [], []
    for (k, a), c in comps.items():
        pos = pos_map[(k, a)]
        if ks is not None:
            pos = pos * ks.reindex(c["dates"]).ffill().fillna(1.0).values
        g.append(np.abs(pos))
        n.append(pos)
        t.append(np.abs(np.diff(pos, prepend=0.0)))
    return dict(gross=float(np.mean(np.concatenate(g))),
                net=float(np.mean(np.concatenate(n))),
                turnover=float(np.mean(np.concatenate(t))))


# ------------------------------------------------------------------ statistics
def sharpe(x, ann=365):
    x = np.asarray(x, float)
    sd = np.std(x)
    return float(np.mean(x) / sd * np.sqrt(ann)) if sd > 1e-15 else 0.0


def ddev(x):
    return float(np.sqrt(np.mean(np.minimum(np.asarray(x, float), 0.0) ** 2)))


def cvar5(x):
    x = np.asarray(x, float)
    q = np.quantile(x, 0.05)
    return float(np.mean(x[x <= q]))


def max_drawdown(x):
    cum = np.cumsum(np.asarray(x, float))
    return float((cum - np.maximum.accumulate(cum)).min())


def stationary_bootstrap_idx(T, B, mean_block=20, seed=7):
    """Politis-Romano stationary bootstrap index matrix (B, T), vectorised over
    the B replicates so B = 10 000 is affordable."""
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    idx = np.empty((B, T), np.int64)
    idx[:, 0] = rng.integers(0, T, size=B)
    restart = rng.random((B, T)) < p
    fresh = rng.integers(0, T, size=(B, T))
    for t in range(1, T):
        idx[:, t] = np.where(restart[:, t], fresh[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def paired_tests(x_base, x_new, ann=365, B=10000, mean_block=20, seed=7):
    """Paired stationary-bootstrap comparison on identical days.

    p-values use the (1 + k) / (B + 1) estimator rather than k / B. The plain
    mean can return exactly zero, which is not a probability any finite
    resampling test can produce; its true floor is 1 / (B + 1). At B = 800 that
    floor is 0.00125, so no result from that setting could support a claim of
    p < 0.001. B = 10 000 puts the floor at 1e-4.
    """
    x_base, x_new = np.asarray(x_base, float), np.asarray(x_new, float)
    I = stationary_bootstrap_idx(len(x_base), B, mean_block, seed)
    nb, nn = x_base[I], x_new[I]

    def _sh(a):
        sd = a.std(axis=1)
        return np.where(sd > 1e-15, a.mean(axis=1) / np.maximum(sd, 1e-300) * np.sqrt(ann), 0.0)

    def _dd(a):
        return np.sqrt((np.minimum(a, 0.0) ** 2).mean(axis=1))

    def _cv(a):
        q = np.quantile(a, 0.05, axis=1, keepdims=True)
        m = a <= q
        return (a * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1)

    dsh, ddd, dcv = _sh(nn) - _sh(nb), _dd(nn) - _dd(nb), _cv(nn) - _cv(nb)
    return dict(d_sharpe=float(dsh.mean()),
                se_sharpe=float(dsh.std(ddof=1)),
                ci_lo=float(np.percentile(dsh, 2.5)),
                ci_hi=float(np.percentile(dsh, 97.5)),
                p_sharpe=float((1 + np.sum(dsh <= 0)) / (B + 1)),
                p_ddev=float((1 + np.sum(ddd >= 0)) / (B + 1)),
                p_cvar=float((1 + np.sum(dcv <= 0)) / (B + 1)),
                B=B)
