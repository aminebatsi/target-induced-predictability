# kalman-trend-pred

Code and cached results for *Predictable trends, unpredictable prices: how the
choice of target inflates direction accuracy in cryptocurrency and traditional
asset markets*.

The study asks why machine-learning studies of financial direction routinely
report 70–90% accuracy when short-horizon price direction is close to
unpredictable. The answer tested here is that such accuracies measure the
*target* rather than the model: scoring against a smoothed price is not the same
as scoring against the price change a position earns.

**Start with [SETUP-README.md](SETUP-README.md)** for installation and the full
reproduction order.

---

## What the code does

**One protocol, three truths.** Thirteen models from five families are trained
under a single purged, embargoed, walk-forward protocol over five assets, giving
6,760 out-of-sample forecasts each. The *same* prediction vector is then scored
against three different truths: the seven-day change of a strictly causal Kalman
trend (the training target), the raw seven-day price change, and the next-day
return a position actually earns.

**A baseline that fits nothing.** A parameter-free rule that extrapolates the
trend's last one-day increment is scored on the same anchors. It matches or beats
every fitted model on the smoothed target while scoring at chance on price.

**A no-model benchmark.** For a centred elliptical pair, the probability that two
variables share a sign is fixed by their correlation through Sheppard's arcsine
formula. Evaluated at the target's own correlation with the increment being
extrapolated, this gives a reference that contains no model at all. A model-free
sweep over fifteen causal transforms, five horizons, fifteen assets and five
folds measures how far real filtered series depart from it.

**A causality audit.** The target transform is audited rather than asserted: an
analytic influence kernel and a future-randomisation test, both required to
return exact zeros.

**An economic diagnostic.** Forecasts are carried through a costed execution
stack, and the signal is reversed inside it with gross exposure and turnover
held identical by construction. This asks whether the forecast's sign survives
an economic transformation; it is not a proposal to trade.

## Layout

Two layers, described in full in [SETUP-README.md](SETUP-README.md):

- **Pipeline** — `run_all.py` orchestrates data, the causality audits, the model
  sweep, the portfolio and the export. Hours of fitting; a GPU helps.
- **Analysis** — modules that replay the cached forecasts in `artifacts/` and
  refit nothing. Around thirty minutes on a laptop.

`config.py` holds every constant that defines the experiment, so the whole
specification can be read in one file.

## Data

Fifteen daily price series (`data/`) are committed so the pipeline runs offline
against exactly the series used in the study. They are daily closes from a
public market-data interface, unadjusted, retrieved with explicit date bounds.
Index legs are index levels and commodity legs are continuous rather than rolled
contracts, which is why the portfolio is described as an economic diagnostic
rather than an executable strategy.

## Third-party code

`third_party/Time-Series-Library` is the upstream implementation of the
transformer-class models, vendored unmodified at commit `4e938a1` so the
comparison uses the authors' own code rather than a reimplementation. It carries
its own licence; see that directory.

## Verification

Three entry points re-derive the reported numbers from the cached artefacts and
fail loudly on drift:

```bash
python verify_headline.py      # 20 headline figures, from cache
python run_all.py symmetry     # matched-reversal algebraic identities
python verify_manuscript.py    # cross-check against the LaTeX source
```
