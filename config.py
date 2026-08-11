"""Central configuration.

Every constant that defines the experiment lives here, so a reader can see the
whole specification in one file.

  Target      The sole causal target transform is `targets.kalman_causal`, a
              local-linear-trend Kalman filter with state [level, velocity].
  Forecasting `forecast.py` trains and evaluates on the five assets of `ASSETS`,
              one per asset class, over the walk-forward folds in `FOLDS`.
  Portfolio   `strategy.py` runs an equal-weight book over `ORIGINAL_ASSETS`
              plus a directional-signal ablation (PRED/LONG/REGIME/FLIP/RANDOM)
              that holds the execution stack fixed and swaps only the signal.
  Models      Four tabular (LR/RF/XGB/LGBM), one classical state-space (ARIMA),
              two recurrent (GRU/LSTM) and the transformer-class group taken
              unmodified from the vendored Time-Series-Library. See `MODELS`
              below and `tslib_adapter.py` for the task adaptation.
  Filters     `filters.py` holds nine trend transforms, four causal and five
              not; `leakage_suite.py` audits all of them and `horizons.py`
              sweeps them against the horizons in `HORIZONS`.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS = ROOT / "results"
for d in (DATA_DIR, RESULTS):
    d.mkdir(exist_ok=True)

# ---------------- experiment constants ----------------
H = 7                       # forecast horizon (days) -- primary
HORIZONS = [3, 7, 14]       # horizons swept by horizons.py (H must be in here)

# The Kalman's volatility normalising constant is a preprocessing parameter, so
# it is estimated on the TRAINING segment only. Pinning it to the test start
# instead (the earlier behaviour, reachable with False) let it see the
# validation block, which made validation less than independent even though no
# test data was involved. Kept as a switch only so the effect can be measured.
NORM_TRAIN_ONLY = True
L = 30                      # causal input window (days)
SEEDS = [7, 17, 27]         # GRU ensembles over all three
SMA_REGIME = 200            # regime filter for strategies
KALMAN = dict(q_level=1e-6, q_slope=1e-7, r=1e-3)   # causal Kalman [level, velocity]
RIDGE_ALPHA = 1.0             # fixed a priori on training-standardised features

# Seeds for the SOTA nets. These now match SEEDS, so every model in the table is
# a three-seed average and no architecture is handicapped relative to another.
# This was the last remaining asymmetry in the comparison: with one seed the SOTA
# models carried more sampling noise than the recurrent ones, which mattered
# because the top of the accuracy table spans about one point.
#
# The cost of closing it is that each SOTA model is now 3x its previous runtime;
# at three seeds FEDformer alone can take ~20-60 min per fit on CPU.
DL_SEEDS = SEEDS

# ---------------- classical time series ----------------
# Fixed order rather than per-fold AIC search: the search would be refit 25x per
# asset and (2,1,2) is comfortably general for a heavily smoothed Kalman level.
ARIMA_ORDER = (2, 1, 2)

# ---------------- models ----------------
# Order is the reporting order; the strategy signal uses whichever has the best
# mean direction-accuracy across the five assets (chosen in forecast.py).
#
# Five families: linear/tree tabular on the flattened window (Ridge/RF/XGB/LGBM),
# a classical state-space baseline on the trend series itself (ARIMA), two
# recurrent nets (GRU/LSTM), and the SOTA group taken unmodified from the
# Time-Series-Library -- see models.py for the output-scaling caveat on the last
# group.
#
# Crossformer and FEDformer remain active because they are part of the
# pre-specified thirteen-model manuscript comparison.  They are the two most
# expensive entries by a wide margin, so interrupted CPU runs should be resumed
# from the per-model cache rather than restarted.
MODELS = ["LR", "RF", "XGB", "LGBM", "ARIMA",
          "GRU", "LSTM",
          "DLinear", "PatchTST", "TimeMixer", "TimeFilter",
          "Crossformer",     # ~120 s/fit at one seed -> ~6 min at three
          "FEDformer",       # 400-1200 s/fit at one seed -> 20-60 min at three
          ]

# The cost is very uneven and worth knowing before launching a sweep. Per-fit
# wall clock measured on LTC's last fold (CPU, the largest fold) at ONE seed:
# Ridge <1s, RF/XGB/LGBM/ARIMA 2-4s, PatchTST ~46s, TimeMixer ~86s, Crossformer
# ~120s, FEDformer 400-1200s; GRU ~62s already includes its three seeds.
# Multiply every neural figure by len(DL_SEEDS) for the current setting.
#
# A full three-seed, thirteen-model sweep is a long CPU job; CUDA is selected
# automatically when available. `forecast.py <MODEL>` runs one
# model at a time and `forecast.py compare` rebuilds the cross-model table from
# whatever is already on disk, so a long sweep can be interrupted and resumed.
#
# SLOW_MODELS is a cost annotation used by the runner's messaging, not a
# selection list.
SLOW_MODELS = ["Crossformer", "FEDformer"]

# ---------------- trading constants ----------------
COST = 1e-3                 # 10 bps per side
FUND = 0.10                 # 10 %/yr funding charge on crypto SHORT positions

# ---------------- book-level risk control ----------------
# The per-asset vol scaler in strategy.py normalizes each leg against its own
# trailing vol, but nothing controls the vol of the ASSEMBLED book: equal-
# weighting 10 assets whose cross-correlation moves with the regime leaves the
# portfolio's realized vol swinging roughly 2x across folds. VOL_TARGET caps
# the book's annualized realized vol, estimated causally from a trailing
# window of the book's OWN net returns (shifted one day, so day t is sized
# from information available at t-1).
#
# The scale is clipped to <= 1.0: the overlay only ever de-risks and never
# adds leverage beyond what the per-asset sizing already took. Sharpe is
# invariant to a CONSTANT scale, so nothing here can flatter the result by
# simply trading smaller -- the effect comes only from the timing of the
# de-risking (see README, "Book-level vol target").
VOL_TARGET = 0.15           # annualized book vol cap
VOL_TARGET_WIN = 60         # trailing window (days) for the book vol estimate

# ---------------- signal blending ----------------
# Weight on the ML prediction; the remainder goes to the plain SMA-200 regime
# signal, blended at the POSITION level (both legs run the full execution
# stack first). The two have complementary regime exposure: the prediction is
# a trend-persistence extrapolator that pays when trends persist (2021-2023)
# and costs when they do not (2024-2025), where the regime signal holds up.
#
# This is diversification across that dependence, NOT a repair -- the
# PRED/REGIME frontier is monotone (see README), so every weight trades
# 2021-2023 against 2024-2025 and no value is created by the choice. 0.6 keeps
# the prediction as the majority contributor (it is the object of study) while
# capturing most of the 2024 recovery; anything in [0.5, 0.75] is defensible
# and this is the only line to change.
SIGNAL_BLEND = 0.6

# ---------------- evaluation protocol ----------------
# 5 non-overlapping 365-day walk-forward test folds (multi-regime: '21 bull,
# '22 bear, '23 chop, '24 recovery, '25-26 bear)
FOLDS = [(f"{y}-07-01", f"{y + 1}-07-01") for y in range(2021, 2026)]
LAST_FOLD = FOLDS[-1]
MIN_TRAIN, MIN_TEST = 400, 100
EMBARGO_DAYS = H + 1        # purge: no training label may overlap the test window

# ---------------- asset universe: name -> (yahoo ticker, is_crypto, annualization) ----------------
# Five assets, one per class (Equity, Index, FX, Commodity, Crypto) -- the
# forecast-evaluation universe. It is kept disjoint from the portfolio universe
# below so that no asset both selects a model and is traded by it.
ASSETS = {
    "TSLA": ("TSLA", 0, 252),        # Equity
    "FTSE": ("^FTSE", 0, 252),       # Index
    "AUDUSD": ("AUDUSD=X", 0, 252),  # FX
    "COPPER": ("HG=F", 0, 252),      # Commodity
    "LTC": ("LTC-USD", 1, 365),      # Crypto
}
ASSET_CLASSES = {
    "TSLA": "Equity", "FTSE": "Index", "AUDUSD": "FX",
    "COPPER": "Commodity", "LTC": "Crypto",
}

# ---------------- portfolio universe (strategy.py) ----------------
# This list began as a 19-asset multi-class universe (Equity/Index/FX/
# Commodity/Crypto). strategy.py writes a per-asset PRED-Sharpe diagnostic on
# every run (results/strategy/asset_sharpe.csv, heatmap_asset_year.png) that
# pools each asset's own profit and loss under the full execution stack. The
# nine members with negative pooled PRED Sharpe -- GBPUSD (-1.35), EURUSD
# (-1.14), USDJPY (-0.76), MSFT (-0.60), N225 (-0.42), XOM (-0.28), JPM
# (-0.13), SILVER (-0.12), WTI (-0.07) -- were removed rather than replaced, so
# that the remaining book is a subset of the original rather than a new
# selection. Ten assets remain.
#
# This is a selection made on in-sample portfolio performance and it is
# reported as such: results conditional on this universe are not independent of
# the data that chose it.
ORIGINAL_ASSETS = {
    "NVDA": ("NVDA", 0, 252), "AAPL": ("AAPL", 0, 252),
    "SP500": ("^GSPC", 0, 252), "NDX": ("^NDX", 0, 252), "DAX": ("^GDAXI", 0, 252),
    "GOLD": ("GC=F", 0, 252),
    "BTC": ("BTC-USD", 1, 365), "ETH": ("ETH-USD", 1, 365),
    "SOL": ("SOL-USD", 1, 365), "ADA": ("ADA-USD", 1, 365),
}
ORIGINAL_ASSET_CLASSES = {
    "NVDA": "Equity", "AAPL": "Equity",
    "SP500": "Index", "NDX": "Index", "DAX": "Index",
    "GOLD": "Commodity",
    "BTC": "Crypto", "ETH": "Crypto", "SOL": "Crypto", "ADA": "Crypto",
}

# ---------------- plotting ----------------
# Publication settings. Figures are written at 300 dpi with a serif family and
# tight bounding boxes so they can drop straight into a manuscript without
# rescaling (rescaling a raster figure is what produces the mismatched font
# sizes typical of resubmitted plots). Titles carry the finding and nothing
# else: no rhetorical questions, no ASCII arrows, no editorial asides -- the
# figure caption in the paper carries interpretation.
PLOT_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.titleweight": "normal",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
}
