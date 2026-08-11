"""Evaluation utilities: purged splits, forecast metrics + significance tests,
trading accounting, and the paired stationary bootstrap.

Everything here is deliberately standard and citable:
  Diebold-Mariano (1995) with Newey-West variance + Harvey-Leybourne-Newbold (1997)
  small-sample correction; Pesaran-Timmermann (1992); Politis-Romano (1994)
  stationary bootstrap.
"""
import numpy as np
from scipy import stats as st

from config import (COST, EMBARGO_DAYS, H, L, MIN_TEST, MIN_TRAIN,
                    NORM_TRAIN_ONLY)

RNG = np.random.default_rng(7)


# ---------------------------------------------------------------- splits
def fold_indices(anchor_dates, test_start, test_end, h=H, legacy=False):
    """Purged chronological split into train, validation and test.

    Purging is done in the asset's OWN observation index, not in calendar days.
    An anchor at position t carries the label trend[t+h] - trend[t], so its
    label endpoint is anchor position t+h regardless of how many calendar days
    those h observations happen to span. Both boundaries are then exact:

        every training anchor t   : t + h < first validation anchor
        every validation anchor t : t + h < first test anchor

    A calendar embargo cannot enforce this across trading calendars. Seven
    observations span seven calendar days on a crypto series but a median of
    nine or ten on an exchange-traded one, so an eight-day embargo leaves the
    last two validation anchors of every non-crypto fold with labels that close
    inside the test window. Validation drives early stopping, so that is a path
    from test-period prices back into model selection, and it is removed here
    rather than bounded.

    Anchors are consecutive observations (`targets.build_windows`), so anchor
    positions and observation indices differ by a fixed offset and the rule can
    be applied to positions directly.

    `legacy=True` restores the calendar-day rule, kept only so the size of the
    correction can be measured rather than asserted.
    """
    ts, tz = np.datetime64(test_start), np.datetime64(test_end)
    te = np.where((anchor_dates >= ts) & (anchor_dates < tz))[0]
    if len(te) < MIN_TEST:
        return None

    if legacy:
        emb = np.timedelta64(EMBARGO_DAYS, "D")
        pre = np.where(anchor_dates <= ts - emb)[0]
        if len(pre) < MIN_TRAIN:
            return None
        i_va = int(0.85 * len(pre))
        tr, va = pre[:i_va], pre[i_va:]
        if len(va) > EMBARGO_DAYS + 10:
            va = va[EMBARGO_DAYS:]
        return tr, va, te

    # dtype pinned: these index torch tensors downstream, and the platform
    # default for np.arange is 32-bit on Windows.
    pre = np.arange(0, max(int(te[0]) - h, 0), dtype=np.intp)
    if len(pre) < MIN_TRAIN:
        return None
    i_va = int(0.85 * len(pre))
    tr, va = pre[:i_va], pre[i_va + h:]           # label endpoint before val
    if len(tr) == 0 or len(va) == 0:
        return None
    assert tr[-1] + h < va[0], "training label crosses the validation boundary"
    assert va[-1] + h < te[0], "validation label crosses the test boundary"
    assert tr[-1] + h < te[0], "training label crosses the test boundary"
    return tr, va, te


def norm_end(dates, ts, tz, h=H, train_only=NORM_TRAIN_ONLY):
    """Index bounding the data the filter's normalising constant may see.

    The constant is a median of pre-window volatility, so it is a preprocessing
    parameter and belongs to the training segment alone. Pinning it to the test
    start, as an earlier version did, let it see the validation block: features
    and labels then depended on validation-period volatility, and early stopping
    selected on data that had influenced its own preprocessing. None of that
    touched the test set, but it is not a defensible split, so the constant is
    cut at the end of training.

    The anchor grid depends only on the series length, the window and the
    horizon, never on the filtered values, so the split can be resolved before
    the filter is run and there is no circularity.

    This lives here, beside `fold_indices`, rather than in `forecast.py`,
    because every analysis module that recomputes the target needs it and must
    use exactly the convention the forecasting run used. Importing it from a
    module that pulls in the model zoo would make a torch installation a
    prerequisite for scoring cached predictions.

    `train_only=False` restores the old behaviour, kept so the size of the
    change can be measured rather than asserted.
    """
    if not train_only:
        return int(np.searchsorted(dates, np.datetime64(ts)))
    anchors = np.arange(L - 1, len(dates) - h)
    idx = fold_indices(dates[anchors], ts, tz, h=h)
    if idx is None or len(idx[0]) == 0:
        return None
    return int(anchors[idx[0][-1]]) + 1


# ---------------------------------------------------------------- forecast metrics
def dir_acc(d_true, d_pred):
    return float(np.mean(np.sign(d_pred) == np.sign(d_true)))


def rmse(d_true, d_pred):
    """Root mean squared forecast error."""
    return float(np.sqrt(np.mean((np.asarray(d_true) - np.asarray(d_pred)) ** 2)))


def mae(d_true, d_pred):
    """Mean absolute forecast error."""
    return float(np.mean(np.abs(np.asarray(d_true) - np.asarray(d_pred))))


def mase(d_true, d_pred, train_naive_scale):
    """Mean absolute scaled error using a train-period naive scale."""
    scale = float(train_naive_scale)
    return mae(d_true, d_pred) / scale if scale > 0 else np.nan


def r2_vs_rw(d_true, d_pred):
    ss = float(np.sum(d_true ** 2))
    return float(1 - np.sum((d_true - d_pred) ** 2) / ss) if ss > 0 else np.nan


def dm_test(e1, e2, h=H):
    """Diebold-Mariano, squared loss, NW(h-1) variance + HLN correction.
    Negative statistic => model 1 (first argument) has LOWER loss."""
    d = e1 ** 2 - e2 ** 2
    T = len(d)
    dbar = d.mean()
    dc = d - dbar
    var = float(np.mean(dc * dc))
    for lag in range(1, h):
        var += 2 * (1 - lag / h) * float(np.mean(dc[lag:] * dc[:-lag]))
    var = max(var, 1e-20) / T
    stat = dbar / np.sqrt(var) * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return float(stat), float(2 * st.t.sf(abs(stat), df=T - 1))


def pt_test(truth_sign, pred_sign):
    """Pesaran-Timmermann directional-accuracy test (z, one-sided p)."""
    yt = (truth_sign > 0).astype(float)
    yp = (pred_sign > 0).astype(float)
    n = len(yt)
    P = np.mean(yt == yp)
    py, pz = yt.mean(), yp.mean()
    Ps = py * pz + (1 - py) * (1 - pz)
    vP = Ps * (1 - Ps) / n
    vPs = ((2 * pz - 1) ** 2 * py * (1 - py) / n + (2 * py - 1) ** 2 * pz * (1 - pz) / n
           + 4 * py * pz * (1 - py) * (1 - pz) / n ** 2)
    den = vP - vPs
    if den <= 0:
        return np.nan, np.nan
    z = (P - Ps) / np.sqrt(den)
    return float(z), float(st.norm.sf(z))


# ---------------------------------------------------------------- trading accounting
def net_daily(pos, r, cost=COST, fund_ann=0.0):
    """Daily net returns: fee on every position CHANGE + funding on short exposure;
    final position closed at the end (fee charged)."""
    pos = np.asarray(pos, float)
    dpos = np.abs(np.diff(pos, prepend=0.0))
    net = pos * r - cost * dpos - (fund_ann / 365.0) * np.maximum(-pos, 0.0)
    net[-1] -= cost * abs(pos[-1])
    return net


def sharpe(x, ann=365):
    sd = np.std(x)
    return float(np.mean(x) / sd * np.sqrt(ann)) if sd > 1e-15 else 0.0


def ddev(x):
    """Downside deviation (std of the negative part)."""
    return float(np.sqrt(np.mean(np.minimum(x, 0.0) ** 2)))


def cvar5(x):
    """Mean of the worst 5% days (less negative = better)."""
    q = np.quantile(x, 0.05)
    return float(np.mean(x[x <= q]))


def max_drawdown(x):
    cum = np.cumsum(x)
    return float((cum - np.maximum.accumulate(cum)).min())


# ---------------------------------------------------------------- paired bootstrap
def stationary_bootstrap_idx(T, B=800, mean_block=20, rng=RNG):
    """Politis-Romano stationary bootstrap index matrix (B, T)."""
    p = 1.0 / mean_block
    idx = np.empty((B, T), np.int64)
    for b in range(B):
        i = rng.integers(0, T)
        for t in range(T):
            if t > 0:
                i = rng.integers(0, T) if rng.random() < p else (i + 1) % T
            idx[b, t] = i
    return idx


def paired_tests(x_base, x_new, ann=365, B=800):
    """Paired comparison on identical days (same resample indices both legs).
    Returns dict: dSharpe + CI, and one-sided p-values that the new strategy
    REDUCES downside deviation / IMPROVES CVaR5."""
    I = stationary_bootstrap_idx(len(x_base), B)
    dsh = np.array([sharpe(x_new[I[b]], ann) - sharpe(x_base[I[b]], ann) for b in range(B)])
    ddd = np.array([ddev(x_new[I[b]]) - ddev(x_base[I[b]]) for b in range(B)])
    dcv = np.array([cvar5(x_new[I[b]]) - cvar5(x_base[I[b]]) for b in range(B)])
    return dict(d_sharpe=float(np.mean(dsh)),
                ci_lo=float(np.percentile(dsh, 2.5)), ci_hi=float(np.percentile(dsh, 97.5)),
                p_ddev=float(np.mean(ddd >= 0)), p_cvar=float(np.mean(dcv <= 0)))
