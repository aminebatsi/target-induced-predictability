"""Models for kalman-trend-pred. Ten regressors of the h-step trend CHANGE,
plus the turn-classifier used as the strategy's exit trigger.

  Tabular (flattened causal window)
    LR           Ridge regression, fixed alpha=1.0 (deterministic)
    RF           random forest
    XGB          gradient-boosted trees (XGBoost, early-stopped on validation)
    LGBM         gradient-boosted trees (LightGBM, early-stopped on validation)

  Classical time series
    ARIMA        ARIMA on the Kalman-trend series itself, h-step ahead

  Sequence / deep (take the (n, L, C) tensor)
    GRU          single-layer GRU-64 -> 2-layer head (local)
    PatchTST     )
    TimeMixer    )  the OFFICIAL Time-Series-Library implementations,
    Crossformer  )  vendored unmodified under third_party/
    FEDformer    )

Public API:
  predict(name, Xtab, Xseq, d, tr, va, te, dir_weighted=False)      -> (pred_val, pred_test)
  predict_full(name, Xtab, Xseq, d, tr, va, te, dir_weighted=False) -> pred for ALL anchor rows
  fit_turn_clf(Xtab, y_turn, tr, va, te)        -> (P(turn) val, P(turn) test)
  SUPPORTS_WEIGHT                               -> names accepting sample_weight

`dir_weighted`: sample_weight = 1/(|d_train|+q25). Note this weights SMALL
trend changes MOST -- earlier docs in this repo described it as upweighting
larger moves, which is the opposite of what the expression does. The behaviour
is kept because it is what every published result here was produced with, and
because both alternatives were measured and are worse: dropping the weight
entirely gives pooled Sharpe 0.97 and inverting it to `|d|+q25` gives 0.96,
against 1.21 as written (see README, "Negative results"). Supported only for
the tabular models in `SUPPORTS_WEIGHT`; the deep nets and ARIMA raise, since
weighting a minibatch loss / a state-space likelihood is out of scope here.

THE FOUR TRANSFORMER-CLASS MODELS ARE UPSTREAM CODE. They come from
thuml/Time-Series-Library, vendored unmodified at `third_party/` and pinned to
commit 4e938a1; `tslib_adapter.py` only supplies the config object each
`Model(configs)` expects. Nothing in that tree is edited.

What IS ours is the task adaptation, and it matters when reading the numbers:
those models are built for multivariate long-horizon forecasting, and here they
are configured `seq_len=30, pred_len=1` with the first output slot trained by
MSE against our scalar target. That keeps the comparison against the other six
models exact -- same inputs, same target, same loss, same early stopping -- but
it means these are NOT benchmark reproductions. See `tslib_adapter` for the
consequences of the short window (Crossformer's hardcoded seg_len=12 leaves 3
segments; FEDformer's mode selection is vacuous at L=30) and for the capacity
knob that keeps them comparable to the GRU rather than at library defaults.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from xgboost import XGBRegressor

from config import ARIMA_ORDER, DL_SEEDS, L, RIDGE_ALPHA, SEEDS
from tslib_adapter import TSLIB_MODELS
from tslib_adapter import factory as tslib_factory

# Models whose .fit accepts sample_weight (i.e. honour `dir_weighted`).
SUPPORTS_WEIGHT = {"LR", "RF", "XGB", "LGBM"}

# Neural models run on GPU when one is available (Colab, workstation) and fall
# back to CPU otherwise. This changes wall-clock time by roughly an order of
# magnitude for the transformer-class models and does NOT change the protocol,
# but note that CUDA and CPU kernels differ in floating-point reduction order,
# so neural results are reproducible on the same device rather than bitwise
# identical across devices. Deterministic models (LR/RF/XGB/LGBM/ARIMA) are
# unaffected.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _standardize(X, tr):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    return (X - mu) / sd


# ================================================================ deep nets
class TinyGRU(nn.Module):
    """Single-layer GRU-64 -> 2-layer head. Small on purpose: anything bigger
    overfits this task (a finding carried over from the pilot study)."""

    def __init__(self, n_ch=3, n_steps=L, hidden=64):
        super().__init__()
        self.rnn = nn.GRU(n_ch, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o[:, -1]).squeeze(-1)


class TinyLSTM(nn.Module):
    """Single-layer LSTM-64 -> 2-layer head.

    Deliberately identical to TinyGRU except for the recurrent cell: same hidden
    width, same head, same trainer, same seeds. The pair therefore isolates the
    gating mechanism (LSTM's separate cell state and output gate versus the GRU's
    coupled update gate) rather than confounding it with capacity."""

    def __init__(self, n_ch=3, n_steps=L, hidden=64):
        super().__init__()
        self.rnn = nn.LSTM(n_ch, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o[:, -1]).squeeze(-1)


# net factories: name -> (callable(n_ch, n_steps) -> Module, seeds, head).
# Every net now averages over the same seed set: GRU and LSTM over SEEDS, the
# SOTA models over DL_SEEDS, which config sets equal to SEEDS. That removes the
# last asymmetry in the comparison. The SOTA models come from the OFFICIAL
# Time-Series-Library code vendored under third_party/ -- see tslib_adapter for
# the task adaptation.
#
# The `head` is NOT tuned. It follows one structural property, readable in each
# upstream file: does the model de-normalise its output back onto the input
# window's own level? PatchTST does (`x_enc - means`, `/= stdev`, both re-added
# to the output), TimeMixer does (`normalize_layers(..., 'denorm')`), FEDformer
# does (adds `x_enc.mean(1)` through `trend_init`) -- those three need "delta".
# Crossformer contains no such rescaling, so its output is free and it needs
# "direct", as does the GRU. That rule predicted all four measured outcomes on
# LTC's last fold: the three coupled models each gained ~0.14-0.16 direction
# accuracy moving to "delta", and Crossformer lost 0.21.
_HEADS = {"PatchTST": "delta",      # x_enc - means, /= stdev, both re-added
          "TimeMixer": "delta",     # normalize_layers(..., 'denorm')
          "FEDformer": "delta",     # adds x_enc.mean(1) via trend_init
          "TimeFilter": "delta",    # layers.StandardNorm.Normalize (RevIN)
          "Crossformer": "direct",  # no rescaling of the output
          "DLinear": "direct"}      # decomposition only, output level is free
_NETS = {
    "GRU": (TinyGRU, SEEDS, "direct"),
    "LSTM": (TinyLSTM, SEEDS, "direct"),
    **{name: (tslib_factory(name), DL_SEEDS, _HEADS[name]) for name in TSLIB_MODELS},
}


def seeds_for(model):
    """Seeds this model averages over, or () if it is deterministic.

    Exposed so forecast.py can stamp the seed set into each cached summary and
    invalidate it when the configuration changes.
    """
    return tuple(_NETS[model][1]) if model in _NETS else ()


def _torch_seed_full(cls, Xseq, d, tr, va, seed, epochs=45, patience=7, bs=64,
                     head="direct"):
    """Fit one net (early-stopped on va) and predict EVERY anchor row.

    Shared by every sequence model so they differ only in architecture: same
    optimizer, same schedule, same early stopping, same evaluated quantity.

    `head` decides how the raw network output is read as a prediction of the
    h-step change. Both options are supervised by MSE on the SAME quantity (the
    scaled change), so the comparison is unaffected -- only the parameterisation
    differs, to match what each architecture is built to emit:

      "direct"  output IS the standardised change. Right for the GRU, whose
                flatten head has no built-in notion of level.
      "delta"   output is a CONTINUATION of the input series, and the prediction
                is output minus the last observed value of channel 0. Required
                for the Time-Series-Library models: TSLib's PatchTST, TimeMixer
                and FEDformer all de-normalise their output back onto the input
                window's own mean/std (RevIN-style, or by adding the input mean
                to a trend component), so their output is pinned to the input's
                LEVEL. Supervising such a model on a zero-centred change fights
                that inductive bias -- measured on LTC's last fold, it cost
                PatchTST/TimeMixer/FEDformer ~0.17-0.19 direction accuracy,
                while Crossformer, the one model in the set without level
                coupling, was unaffected. Reading the implied change instead
                turns the coupling from a handicap into a sensible prior: the
                output starts near the input level, so the implied change starts
                near zero.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_steps, C = Xseq.shape[1], Xseq.shape[-1]
    cmu = Xseq[tr].reshape(-1, C).mean(0)
    csd = Xseq[tr].reshape(-1, C).std(0) + 1e-8
    Xs = ((Xseq - cmu) / csd).astype(np.float32)

    if head == "direct":
        tmu, tsd = d[tr].mean(), d[tr].std() + 1e-12
        base = None                                        # no level reference
        scale = tsd
    elif head == "delta":
        # channel 0 is the Kalman trend; the change in standardised ch-0 units
        tmu = 0.0
        base = Xs[:, -1, 0].astype(np.float32)
        v = d / csd[0]
        scale = float(v[tr].std() + 1e-12) * csd[0]        # so pred*scale is in d units
        tsd = float(v[tr].std() + 1e-12)
    else:
        raise ValueError(head)
    y = (((d - tmu) / scale) if head == "direct" else (d / csd[0] / tsd)).astype(np.float32)

    net = cls(n_ch=C, n_steps=n_steps).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = nn.MSELoss()
    Xt = torch.tensor(Xs, device=DEVICE)
    yt = torch.tensor(y, device=DEVICE)
    bt = None if base is None else torch.tensor(base, device=DEVICE)

    def implied(idx):
        """Network output read as the scaled change, for rows `idx`."""
        o = net(Xt[idx])
        return o if bt is None else (o - bt[idx]) / tsd

    tr_t = torch.as_tensor(np.asarray(tr), device=DEVICE)
    va_t = torch.as_tensor(np.asarray(va), device=DEVICE)
    best, bstate, bad = 1e18, None, 0
    for _ in range(epochs):
        net.train()
        perm = tr_t[torch.randperm(len(tr_t), device=DEVICE)]
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = lf(implied(idx), yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            v_loss = lf(implied(va_t), yt[va_t]).item()
        if v_loss < best - 1e-9:
            best, bstate, bad = v_loss, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if bstate:
        net.load_state_dict(bstate)
    net.eval()
    with torch.no_grad():
        out = []
        for s in range(0, len(Xs), 1024):                  # chunked: attention is O(n^2) in memory
            idx = torch.arange(s, min(s + 1024, len(Xs)), device=DEVICE)
            out.append(implied(idx).cpu().numpy())
    return np.concatenate(out).astype(float) * scale + (tmu if head == "direct" else 0.0)


# ================================================================ ARIMA
def _series_from_windows(Xseq, ch=0):
    """Rebuild a contiguous channel series from the overlapping causal windows.

    ARIMA needs the underlying series, not a design matrix. `build_windows`
    stacks channels as [trend, logprice, return] over consecutive anchors, so
    channel 0 column -1 is trend[anchors] and the first window supplies the
    L-1 values before it. Returned series index j maps to anchor i as
    j = (L - 1) + i.
    """
    return np.concatenate([Xseq[0, :, ch], Xseq[1:, -1, ch]]).astype(float)


def _arima_full(Xseq, tr, h):
    """ARIMA fitted on the TRAIN span of the Kalman-trend series, then h-step
    forecasts at every anchor.

    Estimated once per fold on train only; the fitted parameters are then
    applied to the full series with `refit=False`, so each anchor's forecast
    uses the filtered state at that anchor -- data up to t only. The h-step
    forecast is propagated analytically through the state-space transition,
    `Z @ T^h @ state[t]`, which costs one filter pass instead of one refit per
    anchor.
    """
    import warnings

    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    from statsmodels.tsa.arima.model import ARIMA

    s = _series_from_windows(Xseq, ch=0)
    off = L - 1                                            # series index of anchor 0
    train_end = off + int(tr[-1]) + 1                      # exclusive; train anchors only

    try:
        with warnings.catch_warnings():
            # The Kalman level is near-integrated, so MLE routinely stops on its
            # iteration cap with a usable optimum. Expected here, and silenced so
            # it does not fire 50x per sweep; stationarity/invertibility are not
            # enforced for the same reason.
            warnings.simplefilter("ignore", ConvergenceWarning)
            res = ARIMA(s[:train_end], order=ARIMA_ORDER, trend="n",
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit(method="statespace")
            full = res.apply(s, refit=False)
    except Exception as exc:                               # noqa: BLE001 -- fall back, never crash a fold
        print(f"    [ARIMA] fit failed ({type(exc).__name__}: {exc}); "
              f"falling back to random-walk (zero change)")
        return np.zeros(len(Xseq))

    ssm = full.model.ssm
    T = np.asarray(ssm["transition"])
    Z = np.asarray(ssm["design"])
    if T.ndim == 3:
        T = T[:, :, 0]
    if Z.ndim == 3:
        Z = Z[:, :, 0]
    Th = np.linalg.matrix_power(T, h)
    state = np.asarray(full.filtered_state)                # (k_states, nobs), causal
    level_h = (Z @ Th @ state).ravel()                     # E[y_{t+h} | data <= t]

    anchors_idx = off + np.arange(len(Xseq))
    return level_h[anchors_idx] - s[anchors_idx]           # predicted h-step CHANGE


# ================================================================ full-anchor fit
def predict_full(name, Xtab, Xseq, d, tr, va, te, dir_weighted=False):
    """Fit on `tr` (early-stop / ensemble as appropriate) and return predictions
    for ALL anchor rows, aligned to Xtab / Xseq."""
    if dir_weighted and name not in SUPPORTS_WEIGHT:
        raise NotImplementedError(f"dir_weighted is not defined for {name}")

    if name in _NETS:
        cls, seeds, head = _NETS[name]
        return np.mean([_torch_seed_full(cls, Xseq, d, tr, va, s, head=head)
                        for s in seeds], 0)

    if name == "ARIMA":
        from config import H
        return _arima_full(Xseq, tr, H)

    Xn = _standardize(Xtab, tr)
    w = None
    if dir_weighted:
        w = 1.0 / (np.abs(d[tr]) + np.quantile(np.abs(d[tr]), 0.25))

    if name == "LR":
        # The lagged channels are strongly collinear. A fixed ridge penalty
        # stabilises the baseline across BLAS/LAPACK implementations without
        # selecting anything on validation or test.
        m = Ridge(alpha=RIDGE_ALPHA).fit(Xn[tr], d[tr], sample_weight=w)
    elif name == "RF":
        m = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=20,
                                  max_features="sqrt", n_jobs=-1, random_state=7)
        m.fit(Xn[tr], d[tr], sample_weight=w)
    elif name == "XGB":
        m = XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         n_jobs=-1, random_state=7, early_stopping_rounds=30,
                         eval_metric="rmse")
        m.fit(Xn[tr], d[tr], sample_weight=w, eval_set=[(Xn[va], d[va])], verbose=False)
    elif name == "LGBM":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=400, num_leaves=15, learning_rate=0.03,
                              min_child_samples=20, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=1.0,
                              n_jobs=-1, random_state=7, verbose=-1)
        m.fit(Xn[tr], d[tr], sample_weight=w, eval_set=[(Xn[va], d[va])],
              eval_metric="rmse",
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    else:
        raise ValueError(f"unknown model {name!r}")
    return m.predict(Xn)


def predict(name, Xtab, Xseq, d, tr, va, te, dir_weighted=False):
    """-> (pred_val, pred_test) of the h-step change."""
    allp = predict_full(name, Xtab, Xseq, d, tr, va, te, dir_weighted)
    return allp[va], allp[te]


# ================================================================ turn classifier
def fit_turn_clf(Xtab, y_turn, tr, va, te):
    """-> (P(turn) on val, on test). Linear (logistic) on the standardized window.

    KNOWN LIMITATION, documented rather than hidden. `y_turn` is built by the
    caller as sign(d_t) != sign(w_t), and on the TRAINING rows `w_t` is a fitted
    value of the first-stage forecaster, not an out-of-sample forecast. So this
    second stage learns from in-sample errors. That is not look-ahead -- every
    input is known at the close of day t, and validation/test stay purged and
    chronologically separated -- but it is not the right way to build a
    meta-label. The clean construction is an inner walk-forward inside the
    training block: fit the forecaster on the earlier part, predict the later
    part, and build y_turn from those out-of-sample signs. That costs k refits
    of the first stage per (model, asset, fold) cell, which is a full retraining
    campaign for the neural signals, and it has not been run.

    The direction of the resulting bias is NOT known. The in-sample and
    out-of-sample error distributions differ, which can change both how often
    and when the exit fires; do not assume it flatters or penalises the backtest.
    """
    Xn = _standardize(Xtab, tr)
    clf = LogisticRegression(max_iter=500, class_weight="balanced", C=0.1)
    clf.fit(Xn[tr], y_turn[tr])
    return clf.predict_proba(Xn[va])[:, 1], clf.predict_proba(Xn[te])[:, 1]
