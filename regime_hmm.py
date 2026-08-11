"""Why does the strategy lose money in 2022 and 2025? A per-asset HMM regime read.

`year_analysis.py` established WHAT happened in the losing folds: the long leg
earned almost nothing and cross-asset correlation was at its highest. It did not
establish WHAT KIND OF MARKET that was. This module labels every test day of
every asset with a hidden market regime and asks which regimes the strategy
makes and loses money in, so the paper can state a condition under which a
trend-following stack of this kind should be expected to work.

METHOD. For each (asset, fold) a Gaussian HMM with `N_STATES` states is fitted
on the daily log returns of the PRE-TEST span only, using exactly the training
window the forecasting models saw. Test days are then labelled by FILTERED state
probability: the forward recursion at day t uses parameters from train plus
observations up to t, and nothing after it. This matters. `hmmlearn`'s
`predict` and `predict_proba` return Viterbi and smoothed posteriors, which
condition on the whole test sequence and would let a day be labelled using
information from later days. That would be a look-ahead leak of exactly the kind
this project audits elsewhere, and it would also make the resulting condition
untradeable. The filtered labels used here are available in real time.

Two observables per day drive the fit: the daily log return, and its absolute
value. Returns alone separate drift poorly when volatility dominates; adding
|r| lets the model distinguish a quiet drift from a violent one.

States are unidentified up to permutation, so after fitting they are sorted by
fitted volatility and named LOW, MID and HIGH. The drift within each is reported
rather than assumed.

Outputs -> results/regime_hmm/
"""
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from config import COST, FOLDS, ORIGINAL_ASSETS, PLOT_RC, RESULTS, SIGNAL_BLEND
from data import load_asset
from evaluation import net_daily, sharpe
from strategy import _aggregate, _blended, _nz_sign, _pos_net, book_scale

plt.rcParams.update(PLOT_RC)
OUT = RESULTS / "regime_hmm"
N_STATES = 3
SEED = 7
NAMES = ["low volatility", "mid volatility", "high volatility"]
YEARS = [pd.Timestamp(ts).year for ts, _ in FOLDS]


# ---------------------------------------------------------------- HMM
def _fit_hmm(train_obs):
    """Gaussian HMM on the training span, states relabelled by volatility."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = GaussianHMM(n_components=N_STATES, covariance_type="diag",
                        n_iter=200, random_state=SEED, tol=1e-4)
        m.fit(train_obs)
    # covars_ is returned as full matrices even for covariance_type="diag",
    # so the variance of the return channel is the [0, 0] entry
    order = np.argsort(np.sqrt(m.covars_[:, 0, 0]))       # ascending return vol
    return m, order


def _filtered_states(m, order, obs):
    """Forward recursion: the label at t uses observations up to t only.

    hmmlearn exposes smoothed posteriors, which look at the whole sequence. For
    a regime label that could be acted on, and for consistency with the rest of
    this project's causality discipline, the filter is run explicitly.
    """
    logB = m._compute_log_likelihood(obs)                  # (T, n_states)
    logA = np.log(m.transmat_ + 1e-300)
    log_alpha = np.log(m.startprob_ + 1e-300) + logB[0]
    out = np.empty((len(obs), m.n_components))
    out[0] = np.exp(log_alpha - _logsumexp(log_alpha))
    for t in range(1, len(obs)):
        prior = _logsumexp_axis(log_alpha[:, None] + logA)
        log_alpha = prior + logB[t]
        out[t] = np.exp(log_alpha - _logsumexp(log_alpha))
    rank = np.empty(m.n_components, int)
    rank[order] = np.arange(m.n_components)                # old index -> vol rank
    return out[:, order], rank[np.argmax(out, axis=1)]


def _logsumexp(v):
    mx = v.max()
    return mx + np.log(np.exp(v - mx).sum())


def _logsumexp_axis(M):
    mx = M.max(axis=0)
    return mx + np.log(np.exp(M - mx).sum(axis=0))


# ---------------------------------------------------------------- pipeline
def _book_scales(comps):
    """The book-level volatility scale the shipped strategy actually traded at.

    Omitting this is not a small approximation. The scale is a time-varying
    multiplier derived from the book's own trailing volatility, so leaving it
    out changes which days carry weight. On the 2024 fold it moves Sharpe from
    +0.68 to +0.08, i.e. it would have misattributed a profitable fold as flat.
    """
    raw = {}
    for (k, a), c in comps.items():
        raw.setdefault(k, {})[a] = _pos_net(c, _nz_sign(c["w"]), blend=True)
    return book_scale(_aggregate(raw))


def _positions(c, k_ser, blend=True):
    """Replay the shipped stack for one (fold, asset) component, at book size."""
    sc = k_ser.reindex(c["dates"]).ffill().fillna(1.0).values
    return _blended(c, _nz_sign(c["w"]), sc, blend=blend)


def run(components_pkl=None, model="GRU"):
    OUT.mkdir(parents=True, exist_ok=True)
    src = components_pkl or (RESULTS / "strategy_models" / "_components.pkl")
    comps = pickle.load(open(src, "rb"))[model]
    print(f"== HMM REGIME ANALYSIS ({N_STATES} states, filtered labels; "
          f"signal {model}) ==")
    k_ser = _book_scales(comps)

    rows, params, daily = [], [], []
    for (k, a), c in sorted(comps.items()):
        df = load_asset(a).set_index("date")["logprice"]
        r_all = df.diff().dropna()
        test_dates = pd.DatetimeIndex(c["dates"])
        train_r = r_all[r_all.index < test_dates[0]]
        if len(train_r) < 300:
            continue

        X_tr = np.column_stack([train_r.values, np.abs(train_r.values)])
        m, order = _fit_hmm(X_tr)

        # filter over train+test so the state at the first test day is not a
        # cold start; only the test slice is kept
        full = r_all[r_all.index <= test_dates[-1]]
        X_full = np.column_stack([full.values, np.abs(full.values)])
        _, lab_full = _filtered_states(m, order, X_full)
        lab = pd.Series(lab_full, index=full.index).reindex(test_dates).values

        mu = m.means_[order, 0]
        sd = np.sqrt(m.covars_[order, 0, 0])
        for j in range(N_STATES):
            params.append(dict(fold=k, asset=a, state=j, mean_ret=mu[j],
                               vol_ret=sd[j]))

        pos = _positions(c, k_ser)
        net = net_daily(pos, c["r"], COST, c["fund"])
        daily.append(pd.DataFrame({"fold": k, "asset": a, "date": test_dates,
                                   "state": lab, "net": net,
                                   "pos": pos, "r": c["r"]}))
        for j in range(N_STATES):
            msk = lab == j
            if msk.sum() == 0:
                continue
            rj, pj = c["r"][msk], pos[msk]
            live = np.abs(pj) > 1e-9
            rows.append(dict(
                fold=k, year=YEARS[k], asset=a, state=j,
                days=int(msk.sum()), frac=float(msk.mean()),
                asset_ret=float(rj.sum()),
                net_pnl=float(net[msk].sum()),
                gross_pnl=float((pos * c["r"])[msk].sum()),
                mean_abs_pos=float(np.abs(pj).mean()),
                # does the state actually trend? 1 = one clean move, 0 = pure chop
                trend_eff=float(abs(rj.sum()) / (np.abs(rj).sum() + 1e-12)),
                # is the position on the right side of the next day's move?
                hit=float(np.mean(np.sign(pj[live]) == np.sign(rj[live])))
                if live.sum() > 5 else np.nan,
                live_days=int(live.sum())))
        rows[-1]["switches"] = int(np.sum(np.diff(lab) != 0))
        # per-day labels, for the agreement calculation below
        pd.DataFrame({"date": test_dates, "state": lab}).assign(
            fold=k, asset=a).to_csv(OUT / f"_labels_{k}_{a}.csv", index=False)
        print(f"  fold {k} {a:6} vol/state = "
              + ", ".join(f"{s:.4f}" for s in sd), flush=True)

    R = pd.DataFrame(rows)
    P = pd.DataFrame(params)
    R.to_csv(OUT / "regime_pnl_by_asset.csv", index=False)
    P.groupby("state")[["mean_ret", "vol_ret"]].mean().to_csv(
        OUT / "state_parameters.csv")

    # ---------------- what each regime is, and how the stack fares in it -------
    S = P.groupby("state")[["mean_ret", "vol_ret"]].mean()
    beh = R.groupby("state").apply(
        lambda g: pd.Series({
            "share of days": g.frac.mean(),
            "trend efficiency": g.trend_eff.mean(),
            "hit rate": np.nansum(g.hit * g.live_days) / g.live_days.sum(),
            "exposure": g.mean_abs_pos.mean(),
            "net bps/day": 1e4 * g.net_pnl.sum() / g.days.sum()}),
        include_groups=False)
    S = pd.concat([S, beh], axis=1)
    S.index = NAMES
    S.to_csv(OUT / "state_profile.csv")
    print("\nWHAT EACH REGIME IS, AND HOW THE STACK BEHAVES IN IT")
    print("(pooled over all 50 asset-fold fits; returns in % per day)")
    disp = S.copy()
    disp["mean_ret"] *= 100
    disp["vol_ret"] *= 100
    print(disp.rename(columns={"mean_ret": "mean %/d", "vol_ret": "vol %/d"})
          .round(3).to_string())

    # ---------------- book-level attribution ----------------
    # The book is the equal-weighted mean over the assets AVAILABLE on each
    # date, and the calendars differ: crypto trades weekends, equities do not.
    # So an asset's contribution to the book on day t is net_i(t) / n(t), not
    # net_i(t) / 10. Summing per-asset P&L and dividing by ten does not
    # reproduce the book, and would misstate every number below.
    D = pd.concat(daily, ignore_index=True)
    D["n_avail"] = D.groupby(["fold", "date"])["net"].transform("size")
    D["contrib"] = D["net"] / D["n_avail"]
    D["year"] = D["fold"].map(dict(enumerate(YEARS)))
    D.to_csv(OUT / "daily_regime_attribution.csv", index=False)

    book = D.groupby(["year", "state"]).contrib.sum().unstack()
    book_days = D.groupby(["year", "state"]).size().unstack()
    book.columns = [NAMES[c] for c in book.columns]
    book["fold total"] = book.sum(axis=1)
    book.to_csv(OUT / "book_pnl_by_regime.csv")
    print("\nCONTRIBUTION TO THE BOOK'S LOG RETURN, BY REGIME")
    print("(rows sum to the fold's realised log return)")
    print(book.round(4).to_string())
    print("\nsame, as per cent of the fold's compounded return")
    print((np.exp(book) - 1).mul(100).round(2).to_string())

    # ---------------- regime mix and P&L by fold ----------------
    agg = R.groupby(["year", "state"]).agg(
        frac=("frac", "mean"), net=("net_pnl", "sum"),
        gross=("gross_pnl", "sum"), exposure=("mean_abs_pos", "mean")).reset_index()
    mix = agg.pivot(index="year", columns="state", values="frac")
    pnl = agg.pivot(index="year", columns="state", values="net")
    mix.columns = [NAMES[c] for c in mix.columns]
    pnl.columns = [NAMES[c] for c in pnl.columns]
    mix.to_csv(OUT / "regime_mix_by_fold.csv")
    pnl.to_csv(OUT / "regime_pnl_by_fold.csv")

    print("\nSHARE OF TEST DAYS IN EACH REGIME, averaged over the ten assets")
    print((mix * 100).round(1).to_string())
    print("\nNET P&L BY REGIME (sum of per-asset log P&L, so book units x10)")
    print(pnl.round(3).to_string())

    # P&L per day in regime, which is the fair comparison when regimes differ
    # in how many days they occupy
    dens = R.groupby(["year", "state"]).apply(
        lambda g: 1e4 * g.net_pnl.sum() / g.days.sum(), include_groups=False
    ).unstack()
    teff = R.groupby(["year", "state"]).trend_eff.mean().unstack()
    hit = R.groupby(["year", "state"]).apply(
        lambda g: np.nansum(g.hit * g.live_days) / g.live_days.sum(),
        include_groups=False).unstack()
    for D, nm in ((dens, "net_bps_per_day"), (teff, "trend_efficiency"),
                  (hit, "hit_rate")):
        D.columns = [NAMES[c] for c in D.columns]
        D.to_csv(OUT / f"{nm}_by_fold.csv")
    print("\nNET BASIS POINTS PER DAY IN REGIME (per asset)")
    print(dens.round(1).to_string())
    print("\nTREND EFFICIENCY WITHIN REGIME (1 = one clean move, 0 = pure chop)")
    print(teff.round(3).to_string())
    print("\nHIT RATE WITHIN REGIME (position on the right side of the next move)")
    print(hit.round(3).to_string())

    # ---------------- cross-asset regime agreement ----------------
    lab_files = sorted(OUT.glob("_labels_*.csv"))
    L = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in lab_files])
    daily = L.pivot_table(index=["fold", "date"], columns="asset", values="state")
    agree, hi_frac = {}, {}
    for k in sorted(daily.index.get_level_values(0).unique()):
        d = daily.xs(k, level=0).dropna(how="all")
        # share of assets sitting in the modal regime that day, then averaged
        agree[YEARS[k]] = float(d.apply(
            lambda row: row.value_counts(normalize=True).max(), axis=1).mean())
        hi_frac[YEARS[k]] = float((d == N_STATES - 1).mean(axis=1).mean())
    A = pd.DataFrame({"modal_agreement": agree, "high_vol_share": hi_frac})
    A.index.name = "year"
    A.to_csv(OUT / "regime_agreement.csv")
    for f in lab_files:
        f.unlink()

    print("\nCROSS-ASSET REGIME AGREEMENT")
    print("modal_agreement = share of assets in the same regime on a given day")
    print(A.round(3).to_string())

    # ---------------- tie it to fold performance ----------------
    fold_net = R.groupby("year")["net_pnl"].sum() / R.asset.nunique()
    summary = pd.concat([mix.add_prefix("share "), A,
                         fold_net.rename("book_log_pnl")], axis=1)
    summary["switches_per_asset"] = R.groupby("year")["switches"].mean()
    summary.to_csv(OUT / "fold_regime_summary.csv")
    corr = summary.corr(numeric_only=True)["book_log_pnl"].drop("book_log_pnl")
    corr.to_frame("corr_with_fold_pnl").to_csv(OUT / "regime_drivers.csv")
    print("\nFOLD SUMMARY")
    print(summary.round(3).to_string())
    print("\nCORRELATION WITH FOLD P&L (n=5 folds, descriptive only)")
    print(corr.round(3).to_string())

    _figure(mix, pnl, A, summary)
    print(f"\n-> {OUT}")
    return mix, pnl, A, summary


def _figure(mix, pnl, A, summary):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    a0, a1, a2 = axes
    cols = ["#4F9D69", "#E8963C", "#C0392B"]
    xs = np.arange(len(mix))

    bot = np.zeros(len(mix))
    for j, n in enumerate(mix.columns):
        a0.bar(xs, mix[n] * 100, 0.62, bottom=bot, label=n, color=cols[j])
        bot += mix[n].values * 100
    a0.set_xticks(xs, mix.index)
    a0.set_xlabel("test fold")
    a0.set_ylabel("share of test days (%)")
    a0.set_title("(a) Regime mix by fold")
    a0.legend(fontsize=8)

    w = 0.26
    for j, n in enumerate(pnl.columns):
        a1.bar(xs + (j - 1) * w, pnl[n], w, label=n, color=cols[j])
    a1.axhline(0, color="k", lw=0.9)
    a1.set_xticks(xs, pnl.index)
    a1.set_xlabel("test fold")
    a1.set_ylabel("net log P&L, summed over assets")
    a1.set_title("(b) Where the money is made and lost")
    a1.legend(fontsize=8)

    a2.plot(xs, A.modal_agreement, "o-", color="#3B6EA5",
            label="cross-asset regime agreement")
    a2.plot(xs, A.high_vol_share, "s-", color="#C0392B",
            label="share of asset-days in high volatility")
    a2.set_xticks(xs, A.index)
    a2.set_xlabel("test fold")
    a2.set_ylabel("value")
    a2.set_title("(c) Regime synchronisation across assets")
    a2.legend(fontsize=8)

    fig.suptitle("Hidden-regime read of the losing folds: states fitted per asset "
                 "on pre-test data, test days labelled by filtered probability",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "regime_analysis.png")
    plt.close(fig)


if __name__ == "__main__":
    run()
