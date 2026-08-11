"""Trend-extraction filters, causal and not, for the target-transform audit.

The paper's argument is that a high direction accuracy can be a property of the
target rather than of the model. That argument only generalises if it holds for
the other smoothers this literature actually uses, so this module collects them
behind one interface and labels each with whether it is *expected* to be causal.

Every filter has the signature

    f(p, norm_end=None) -> level        (same length as p)

`norm_end` is honoured by the filters that need a normalising constant and
ignored by the rest; it exists so that callers can pin every transform to the
same pre-test boundary without special-casing.

The `causal` flag is a PREDICTION, not an assertion. leakage.py measures each
filter independently and the prediction is only there so a disagreement between
what we expect and what we measure is visible. The leaky entries are included
deliberately: a smoother that looks perfectly reasonable in a methods section
and silently reads the future is the failure mode this paper is about.
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_coeffs, savgol_filter

from targets import kalman_causal

SG_WIN, SG_POLY = 21, 2
MA_WIN = 21
EMA_SPAN = 20
HP_LAMBDA = 129600          # the usual daily-data setting
WAVELET, WAVE_LEVEL = "db4", 4
BUTTER_ORDER, BUTTER_WN = 3, 0.05


# ------------------------------------------------------------------ causal
def f_kalman(p, norm_end=None):
    """Local linear trend, [level, velocity]. The paper's primary target."""
    return kalman_causal(p, norm_end)[0]


def f_kalman_level(p, norm_end=None):
    """Local level only, no velocity state. A strictly simpler state space."""
    p = np.asarray(p, float)
    n = len(p)
    if norm_end is None:
        norm_end = n
    ret = np.diff(p, prepend=p[0])
    rv = pd.Series(ret ** 2).ewm(span=20).mean().values
    rv = rv / (np.median(rv[20:max(norm_end, 40)]) + 1e-18)
    q, r0 = 1e-5, 1e-3
    x, P = p[0], 1.0
    out = np.empty(n)
    for t in range(n):
        P += q
        R = r0 * rv[t]
        K = P / (P + R)
        x += K * (p[t] - x)
        P *= (1 - K)
        out[t] = x
    return out


def f_ema(p, norm_end=None):
    """Exponential moving average. Causal by construction, adjust=False."""
    return pd.Series(np.asarray(p, float)).ewm(
        span=EMA_SPAN, adjust=False).mean().values


def f_sg_causal(p, norm_end=None):
    """Savitzky-Golay evaluated at the RIGHT EDGE of a trailing window.

    Same polynomial smoothing as the textbook version, but the fit is read at
    the last point of the window instead of the middle, so no future sample
    enters. This is the causal way to use SG and it is rarely what papers do.
    """
    p = np.asarray(p, float)
    c = savgol_coeffs(SG_WIN, SG_POLY, pos=SG_WIN - 1, use="dot")
    out = np.empty(len(p))
    for t in range(len(p)):
        if t < SG_WIN - 1:
            out[t] = p[:t + 1].mean()
        else:
            out[t] = float(c @ p[t - SG_WIN + 1:t + 1])
    return out


# -------------------------------------------------------------- not causal
def f_ma_centred(p, norm_end=None):
    """Centred moving average. Reads (MA_WIN-1)/2 bars into the future."""
    return pd.Series(np.asarray(p, float)).rolling(
        MA_WIN, center=True, min_periods=1).mean().values


def f_sg_centred(p, norm_end=None):
    """Textbook Savitzky-Golay, centred window. The common leaky choice."""
    return savgol_filter(np.asarray(p, float), SG_WIN, SG_POLY, mode="nearest")


def f_hp(p, norm_end=None):
    """Hodrick-Prescott trend. A two-sided smoother of the whole sample."""
    from statsmodels.tsa.filters.hp_filter import hpfilter
    return np.asarray(hpfilter(np.asarray(p, float), lamb=HP_LAMBDA)[1])


def f_wavelet(p, norm_end=None):
    """Wavelet soft-threshold denoising over the full series.

    This is the transform used by the framework the introduction cites. The
    decomposition is computed on the whole sample, so every reconstructed point
    depends on every observation, including later ones.
    """
    import pywt
    p = np.asarray(p, float)
    coeffs = pywt.wavedec(p, WAVELET, level=WAVE_LEVEL, mode="periodization")
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thr = sigma * np.sqrt(2 * np.log(len(p)))
    out = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
    rec = pywt.waverec(out, WAVELET, mode="periodization")
    return rec[:len(p)]


def f_butter_zerophase(p, norm_end=None):
    """Zero-phase Butterworth via filtfilt. Runs the filter forwards then
    backwards, so the backward pass carries future information into every
    point. Popular precisely because it does not lag."""
    b, a = butter(BUTTER_ORDER, BUTTER_WN)
    return filtfilt(b, a, np.asarray(p, float))


# ------------------------------------------------------------------ registry
FILTERS = {
    "kalman":      dict(fn=f_kalman,           causal=True,
                        label="Kalman local linear trend"),
    "kalman_lvl":  dict(fn=f_kalman_level,     causal=True,
                        label="Kalman local level"),
    "ema":         dict(fn=f_ema,              causal=True,
                        label=f"EMA (span {EMA_SPAN})"),
    "sg_causal":   dict(fn=f_sg_causal,        causal=True,
                        label=f"Savitzky-Golay, trailing ({SG_WIN},{SG_POLY})"),
    "ma_centred":  dict(fn=f_ma_centred,       causal=False,
                        label=f"Centred moving average ({MA_WIN})"),
    "sg_centred":  dict(fn=f_sg_centred,       causal=False,
                        label=f"Savitzky-Golay, centred ({SG_WIN},{SG_POLY})"),
    "hp":          dict(fn=f_hp,               causal=False,
                        label="Hodrick-Prescott"),
    "wavelet":     dict(fn=f_wavelet,          causal=False,
                        label=f"Wavelet denoise ({WAVELET})"),
    "butter":      dict(fn=f_butter_zerophase, causal=False,
                        label="Zero-phase Butterworth"),
}

CAUSAL = [k for k, v in FILTERS.items() if v["causal"]]
LEAKY = [k for k, v in FILTERS.items() if not v["causal"]]


def apply_filter(name, p, norm_end=None):
    """Run one filter by name and return its level series."""
    return np.asarray(FILTERS[name]["fn"](p, norm_end), float)
