"""The transform family the cited studies use, made causal and measured.

The related-work table lists studies that denoise a price series with a wavelet
transform and then forecast the denoised object. We cannot reanalyse those
studies: their code and their exact preprocessed series are not available, and
inventing a reconstruction of someone else's pipeline would not be evidence
about it. What we can do is apply the same transform *class*, under the same
protocol as the rest of this paper, and report what a rule that fits nothing
scores on the target it produces.

Two variants are built, and the difference between them is the point.

  causal wavelet   Soft-threshold denoising of a trailing window ending at t,
                   read at its last point. Uses no observation after t.
  full-sample      The same denoising applied once to the entire series, which
                   is the common implementation. The value at t then depends on
                   observations after t.

The causal variant isolates target-induced predictability; the full-sample one
adds look-ahead leakage on top of it. Reporting both separates the two effects
that the introduction distinguishes, on the transform family where the question
most often arises.

Outputs -> results/analysis/causal_wavelet.csv
"""
import numpy as np
import pandas as pd
import pywt

from config import ASSETS, FOLDS, L, ORIGINAL_ASSETS, RESULTS
from data import load_asset
from evaluation import fold_indices

OUT = RESULTS / "analysis"
UNIVERSE = list(dict.fromkeys(list(ASSETS) + list(ORIGINAL_ASSETS)))
WAVELET, LEVEL = "db4", 4
WIN = 256                      # trailing window for the causal variant
HORIZONS = (1, 3, 7, 14, 30)


def _denoise(x, wavelet=WAVELET, level=LEVEL):
    """Universal-threshold soft denoising of one segment."""
    lev = min(level, pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len))
    if lev < 1:
        return np.asarray(x, float)
    c = pywt.wavedec(x, wavelet, level=lev)
    sigma = np.median(np.abs(c[-1])) / 0.6745 if len(c[-1]) else 0.0
    thr = sigma * np.sqrt(2.0 * np.log(max(len(x), 2)))
    c = [c[0]] + [pywt.threshold(d, thr, mode="soft") for d in c[1:]]
    return np.asarray(pywt.waverec(c, wavelet)[:len(x)], float)


def causal_wavelet(p, win=WIN):
    """Denoise a trailing window ending at t and keep its last value.

    Strictly causal by construction: the value at t is a function of
    p[t-win+1 .. t] only. Verified below rather than asserted.
    """
    p = np.asarray(p, float)
    out = np.empty(len(p))
    for t in range(len(p)):
        a = max(0, t - win + 1)
        seg = p[a:t + 1]
        out[t] = _denoise(seg)[-1] if len(seg) >= 8 else p[t]
    return out


def full_sample_wavelet(p):
    """The common implementation: one pass over the whole series."""
    return _denoise(np.asarray(p, float))


def audit_causality(p, t0=1500, seed=3):
    """Randomise everything after t0 and confirm the causal variant is
    unchanged up to t0, while the full-sample variant is not."""
    rng = np.random.default_rng(seed)
    q = p.copy()
    q[t0 + 1:] = p[t0 + 1:] + rng.normal(0, 5 * np.std(np.diff(p)), len(p) - t0 - 1)
    c1, c2 = causal_wavelet(p)[:t0 + 1], causal_wavelet(q)[:t0 + 1]
    f1, f2 = full_sample_wavelet(p)[:t0 + 1], full_sample_wavelet(q)[:t0 + 1]
    return float(np.max(np.abs(c1 - c2))), float(np.max(np.abs(f1 - f2)))


def run():
    rows = []
    for asset in UNIVERSE:
        df = load_asset(asset)
        dts = df["date"].values
        lp_full = df["logprice"].values.astype(float)
        for k, (ts, tz) in enumerate(FOLDS):
            cut = int(np.searchsorted(dts, np.datetime64(tz)
                                      + np.timedelta64(2, "D")))
            if cut < 700:
                continue
            dates, lp = dts[:cut], lp_full[:cut]
            levels = {"causal wavelet": causal_wavelet(lp),
                      "full-sample wavelet": full_sample_wavelet(lp)}
            for h in HORIZONS:
                anchors = np.arange(L - 1, cut - h)
                idx = fold_indices(dates[anchors], ts, tz, h=h)
                if idx is None:
                    continue
                a = anchors[idx[2]]
                for name, lev in levels.items():
                    tgt = lev[a + h] - lev[a]
                    inc = lev[a] - lev[a - 1]
                    if np.std(tgt) < 1e-15 or np.std(inc) < 1e-15:
                        continue
                    s = np.sign(inc)
                    rho1 = float(np.corrcoef(tgt, inc)[0, 1])
                    rows.append(dict(
                        transform=name, asset=asset, fold=k, h=h, n=len(a),
                        rho1=rho1,
                        B=0.5 + np.arcsin(np.clip(rho1, -1, 1)) / np.pi,
                        da_target=float(np.mean(s == np.sign(tgt))),
                        da_price_h=float(np.mean(
                            s == np.sign(lp[a + h] - lp[a]))),
                        da_price_1=float(np.mean(
                            s == np.sign(lp[a + 1] - lp[a]))),
                        strength=float(np.std(np.diff(lev)) / np.std(np.diff(lp)))))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lp = load_asset("TSLA")["logprice"].values.astype(float)
    dc, df_ = audit_causality(lp)
    print("== causality audit (randomise everything after the anchor) ==")
    print(f"  causal wavelet     max |change| before the anchor: {dc:.3e}")
    print(f"  full-sample wavelet max |change| before the anchor: {df_:.3e}")

    R = run()
    R.to_csv(OUT / "causal_wavelet.csv", index=False)
    g = (R.groupby(["transform", "h"])
          .agg(strength=("strength", "mean"), rho1=("rho1", "mean"),
               da_target=("da_target", "mean"), B=("B", "mean"),
               da_price_h=("da_price_h", "mean"),
               da_price_1=("da_price_1", "mean"), cells=("n", "size"))
          .reset_index())
    pd.set_option("display.width", 200)
    print("\n== parameter-free rule on a wavelet-denoised target ==")
    print(g.round(4).to_string(index=False))
    for t in g["transform"].unique():
        s = g[g["transform"] == t]
        print(f"\n  {t}: DA_target {s.da_target.min():.4f}-{s.da_target.max():.4f}"
              f"  price-1 {s.da_price_1.min():.4f}-{s.da_price_1.max():.4f}"
              f"  price-h {s.da_price_h.min():.4f}-{s.da_price_h.max():.4f}")
    print(f"\n-> {OUT / 'causal_wavelet.csv'}")


if __name__ == "__main__":
    main()
