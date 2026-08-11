# Setup and reproduction guide

How to install the environment and reproduce every number in the study, from
raw prices through to the tables and figures used in the manuscript.

The repository is split into two layers, and they have very different costs:

| Layer | What it does | Needs a GPU? | Wall clock |
|---|---|---|---|
| **Pipeline** (`run_all.py`) | Downloads prices, fits all models, runs the backtest | Strongly recommended | 6–8 h on CPU, under 1 h on a T4 |
| **Analysis** (see below) | Replays cached forecasts to produce the statistics and tables | No | ~30 min total |

If you only want to check the reported statistics, run the analysis layer. It
reads the cached forecasts committed under `artifacts/` and refits nothing.

---

## 1. Environment

Python 3.11 or 3.12 (3.10 also works).

```bash
git clone <repository-url>
cd kalman-trend-pred

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Vendored dependency

The transformer-class models are the upstream Time-Series-Library
implementations, used unmodified. They are vendored rather than pip-installed so
that the commit is pinned at `4e938a1`. If `third_party/Time-Series-Library` is
empty:

```bash
git clone https://github.com/thuml/Time-Series-Library.git \
    third_party/Time-Series-Library
cd third_party/Time-Series-Library && git checkout 4e938a1 && cd ../..
```

Only `layers/` and `models/` are imported, which need `torch` and `einops`
alone; the upstream `requirements.txt` is not required.

> **Before the first `git add` in this repository.** If
> `third_party/Time-Series-Library` still contains its own `.git` directory
> (because it was cloned in place), git records it as an empty gitlink and the
> vendored source is *not* published. Either remove the nested history so the
> files are tracked normally:
>
> ```bash
> rm -rf third_party/Time-Series-Library/.git
> ```
>
> or register it as a real submodule pinned to the same commit. The upstream
> `LICENSE` is at `third_party/Time-Series-Library/LICENSE` and must be kept
> either way.

### GPU

`models.DEVICE` selects CUDA automatically when it is available. To check what
the machine will use:

```bash
python gpu_probe.py
```

---

## 2. Data

Fifteen daily price series are committed under `data/` so that the pipeline runs
offline against exactly the series used in the study. To refresh them from the
provider instead, delete the CSVs and run:

```bash
python data.py
```

Prices are daily closes pulled with explicit epoch bounds. Requesting a maximal
date range silently returns monthly bars, so the loader asserts daily
granularity and will fail loudly rather than proceed on downsampled data.

---

## 3. Full pipeline

```bash
python run_all.py                  # every step, in order
python run_all.py <step>           # a single step
python run_all.py check            # re-render the acceptance report only
```

Steps, in the order `run_all.py` executes them:

| Step | Module | Produces | Approx. time |
|---|---|---|---|
| `data` | `data.py` | `data/*.csv` | seconds (cached) |
| `design` | `design_figure.py` | split diagram | seconds |
| `leakage` | `leakage.py` | causality audit of the Kalman target | seconds |
| `filters` | `leakage_suite.py` | causality audit of all nine transforms | ~1 min |
| `forecast` | `forecast.py` | per-model forecasts, `best_model.json` | **6–8 h CPU / <1 h T4** |
| `horizons` | `horizons.py` | filter × horizon sweep | ~20 min |
| `strategy` | `strategy.py` | portfolio, ablation arms, paired proofs | ~10 min |
| `symmetry` | `reversal_symmetry.py` | matched-reversal identities | ~1 min |
| `folds` | `fold_analysis.py` | per-fold diagnostics | ~2 min |
| `years` | `year_analysis.py` | return attribution | ~2 min |
| `voltarget` | `vol_target_sweep.py` | overlay surface and controls | ~5 min |
| `export` | `export_results.py` | consolidated `results/paper_export/` | seconds |

`sweep` (`sweep.py`) is excluded from `all` because it is hours of fitting on
its own and no headline result depends on it. Run it by name if wanted.

Every step appends tolerance-based checks to `results/ACCEPTANCE.md`. A step
that fails a check exits non-zero.

### Running the pipeline on Colab

`run_experiments_colab.ipynb` runs the whole pipeline end to end on a hosted
GPU and packages `results/` for download. Set *Runtime → Change runtime type →
GPU* (a T4 is sufficient) before starting. This is the practical route if no
local GPU is available.

### Resuming an interrupted forecast run

```bash
FORECAST_RESUME=1 python run_all.py forecast     # skip models already on disk
FORECAST_WORKERS=4 python run_all.py forecast    # fit (asset, fold) cells in parallel
```

`FORECAST_WORKERS` changes wall clock only; it reproduces the serial predictions
exactly. `FORECAST_WORKERS=0` auto-sizes from available GPU memory.

### Regenerating the cached artefacts

`artifacts/` is produced by the pipeline, not by hand. After a full run:

```bash
python strategy_models.py          # writes the per-(fold, asset) component cache
```

then copy `results/strategy_models/_components.pkl` to
`artifacts/strategy_components.pkl` and the per-model prediction CSVs from
`results/paper_export/` to `artifacts/forecast_predictions/`.

---

## 4. Analysis layer

These modules refit nothing. They replay the cached forecasts in `artifacts/`
and write to `results/analysis/`. Run them in this order — later modules read
files written by earlier ones.

```bash
python transform_sweep.py          # ~2 min   model-free transform x horizon sweep
python predictive_ability.py       # ~4 min   SPA, reality check, Romano-Wolf
python exposure_matching.py        # ~13 min  matched reversal, costs, deflated Sharpe
python excess_accuracy.py          # ~6 min   ExDA, accuracy intervals, universe test
python benchmark_residual.py       # ~10 s    where the sign-agreement benchmark fails
python turn_exit_ablation.py       # ~3 min   pre-specified turn-classifier removal test
python return_mechanism.py         # ~2 min   magnitude vs frequency decomposition
python reversal_symmetry.py        # ~2 min   friction algebra of the matched reversal
python strategy_proofs.py          # ~2 min   supporting per-fold and exposure tables
python block_sensitivity.py        # ~10 min  bootstrap block lengths 10 / 20 / 40
```

Dependencies worth knowing:

- `predictive_ability.py` needs `slope_daily_accuracy.csv` from
  `transform_sweep.py`.
- `benchmark_residual.py` needs `slope_generality_sweep.csv` from
  `transform_sweep.py`.
- `block_sensitivity.py` imports `excess_accuracy.py`, `return_mechanism.py`
  and `predictive_ability.py`.

### Verification

```bash
python verify_headline.py          # re-derives 20 headline numbers from cache
python run_all.py symmetry         # matched-reversal identities, into ACCEPTANCE.md
python verify_manuscript.py        # cross-checks the numbers against the LaTeX source
```

`verify_headline.py` exits non-zero if any headline figure fails to reproduce
within tolerance. `verify_manuscript.py` additionally requires the manuscript
sources at `../paper/manuscript.tex` and is skipped without them.

---

## 5. Layout

```
config.py              every experiment constant, in one file
data.py                price download and cache
targets.py             causal Kalman filter and supervised-window construction
filters.py             nine trend transforms, four causal and five not
models.py              all model fits, plus the turn classifier
tslib_adapter.py       adapter for the vendored Time-Series-Library models
evaluation.py          splits, metrics, significance tests, bootstrap
strategy.py            portfolio construction and the signal ablation
run_all.py             pipeline orchestrator and acceptance checks

portfolio_replay.py    replays the portfolio from cached components (library)
transform_sweep.py     ... analysis modules, see section 4
predictive_ability.py
exposure_matching.py
excess_accuracy.py
benchmark_residual.py
turn_exit_ablation.py
return_mechanism.py
reversal_symmetry.py
strategy_proofs.py
block_sensitivity.py
verify_headline.py     verification entry points
verify_manuscript.py

artifacts/             cached forecasts the analysis layer replays (tracked)
data/                  daily price series (tracked)
results/               all generated output (not tracked)
third_party/           vendored Time-Series-Library at commit 4e938a1
```

---

## 6. Reproducibility notes

Deterministic components reproduce exactly. Neural components are pinned to a
fixed initialisation, but CUDA and CPU kernels differ in floating-point
summation order, so neural results reproduce on a given device rather than bit
for bit across devices.

The 90-column linear design is strongly collinear, so the `LR` model identifier
uses Ridge regression with a fixed `alpha=1.0` on training-standardised
features. The coefficient is specified in `config.py` and is never selected on
validation or test data. This removes the implementation-dependent numerical
rank choice of unregularised least squares. The parameter-free slope rule is
still only two subtractions.

The cached component pickle in `artifacts/` was written under numpy >= 2, which
renamed `numpy.core` to `numpy._core`. `portfolio_replay.install_numpy2_shim()`
aliases the old module tree onto the new names so the file also loads under
numpy 1.x. It is called automatically.
