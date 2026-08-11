"""Quick, visual leakage check for the causal Kalman trend (kalman-trend-pred).

One question: does kalman_causal(p)[t0] contain any information about the
FUTURE (prices at t+1, t+2, ...)? A causal filter must answer no. The one
subtlety carried over from the original study: the filter's volatility-
adaptive measurement noise is normalized by a constant computed from
PRE-CUT data only (`norm_end=t0+1`) -- normalizing over the full sample
(including future data) would be a real, if subtle, leak, so every check here
pins `norm_end` to the anchor under test, exactly as every fold-based caller
in this package does.

Two clear tests, one figure:

  A  influence kernel:  d kalman_causal[t0] / d p[t0+k] for k in [-W, +W].
     A causal filter reacts to past/present bars (k<=0) and is EXACTLY 0 for
     every future bar (k>0). We plot the kernel with the future region shaded.

  B  corrupt-the-future overlay:  randomize every price after t0, recompute
     the whole Kalman series (same norm_end), and overlay it on the original.
     The two curves must be bit-identical up to t0 and only differ afterwards.

Output: results/leakage/kalman_leakage.png + results/leakage/audit.json
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import PLOT_RC, RESULTS
from data import load_asset
from targets import kalman_causal

OUT = RESULTS / "leakage"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(PLOT_RC)
WINDOW = 30


def influence_kernel(lp, t0, ks, eps=1e-2):
    """d kalman_causal[t0] / d p[t0+k] via finite differences (one bump at a
    time); norm_end is pinned to t0+1 (pre-cut only) both before and after the
    bump, isolating the filter's own causality from the normalization step."""
    base = kalman_causal(lp, norm_end=t0 + 1)[0][t0]
    g = []
    for k in ks:
        p2 = lp.copy()
        p2[t0 + k] += eps
        g.append((kalman_causal(p2, norm_end=t0 + 1)[0][t0] - base) / eps)
    return np.asarray(g)


def corrupt_future(lp, t0, seed=0):
    """Recompute the Kalman trend after randomizing every price strictly after
    t0, keeping norm_end pinned to t0+1 in both runs."""
    rng = np.random.default_rng(seed)
    p2 = lp.copy()
    p2[t0 + 1:] += rng.normal(0, 0.05, len(lp) - t0 - 1)
    return kalman_causal(lp, norm_end=t0 + 1)[0], kalman_causal(p2, norm_end=t0 + 1)[0]


def main():
    lp = load_asset("LTC")["logprice"].values.astype(float)
    n = len(lp)
    t0 = n - 300
    ks = np.arange(-WINDOW, WINDOW + 1)
    fut = ks > 0

    kernel = influence_kernel(lp, t0, ks)
    future_gain = float(np.max(np.abs(kernel[fut])))     # must be 0.0

    s_orig, s_corr = corrupt_future(lp, t0)
    pre_diff = float(np.max(np.abs(s_orig[:t0 + 1] - s_corr[:t0 + 1])))   # must be 0.0

    passed = (future_gain == 0.0) and (pre_diff == 0.0)
    audit = dict(target="kalman_causal", t0=int(t0),
                 max_future_influence=future_gain,
                 max_pre_cut_diff_after_future_corruption=pre_diff,
                 PASSED=bool(passed))
    json.dump(audit, open(OUT / "audit.json", "w"), indent=2)

    # -------------------------------------------------- figure
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.6))

    ax0.axvspan(0.5, ks[-1] + 0.5, color="orange", alpha=0.15,
                label=r"future region ($k>0$)")
    ax0.stem(ks[~fut], kernel[~fut], linefmt="seagreen", markerfmt="o", basefmt="k-",
             label=r"past and present ($k \leq 0$)")
    ax0.stem(ks[fut], kernel[fut], linefmt="crimson", markerfmt="x", basefmt="k-",
             label=r"future ($k>0$)")
    ax0.axhline(0, color="k", lw=0.8)
    ax0.set_xlabel(r"lag $k$ of the perturbed observation $p_{t_0+k}$")
    ax0.set_ylabel(r"$\partial \ell_{t_0} / \partial p_{t_0+k}$")
    ax0.set_title(r"(a) Influence kernel: $\max_{k>0}|\partial \ell_{t_0}/"
                  rf"\partial p_{{t_0+k}}| = {future_gain:.1e}$")
    ax0.legend()

    seg = np.arange(t0 - 40, min(t0 + 40, n))
    ax1.plot(seg, lp[seg], color=".7", lw=0.9, label=r"log price $p_t$")
    ax1.plot(seg, s_orig[seg], color="seagreen", lw=1.6,
             label=r"filtered level $\ell_t$, observed future")
    ax1.plot(seg, s_corr[seg], color="crimson", lw=1.2, ls="--",
             label=r"filtered level $\ell_t$, randomised future")
    ax1.axvline(t0, color="navy", ls=":", lw=1.2, label=r"audit anchor $t_0$")
    ax1.set_xlabel("observation index")
    ax1.set_ylabel("log price")
    ax1.set_title(r"(b) Future-randomisation invariance: "
                  rf"$\max_{{t \leq t_0}}|\ell_t - \ell'_t| = {pre_diff:.1e}$")
    ax1.legend()

    fig.suptitle("Causality audit of the local-linear-trend Kalman target",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "kalman_leakage.png")
    plt.close(fig)

    print(f"kalman future influence      = {future_gain:.2e}  (must be 0)")
    print(f"pre-t0 diff after corruption = {pre_diff:.2e}  (must be 0)")
    print(f"LEAKAGE CHECK {'PASSED' if passed else 'FAILED'} -> {OUT}/kalman_leakage.png")
    return passed


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
