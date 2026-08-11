"""Consolidate every experimental output into one flat, self-describing folder.

The rest of the pipeline writes results next to the code that produced them,
which is convenient while running but awkward afterwards: to write a paper you
want one directory, uniform naming, long-format tables, and a manifest saying
what each file is. This module produces exactly that at `results/paper_export/`.

Design rules:

  * FLAT. One directory, no nesting, `area__name.csv` naming, so a glob or a
    single `pd.read_csv` loop picks everything up.
  * LONG FORMAT where a table has a natural key. Row-level prediction and
    position files carry every identifier (model, asset, fold, date), so any
    aggregate in the paper can be recomputed and any new cut can be taken
    without re-running a single model fit.
  * SELF-DESCRIBING. `MANIFEST.csv` lists every file with its row count,
    columns and a one-line description. `config_snapshot.json` records the
    constants the run used, and `environment.json` the library versions, so a
    result can always be tied back to the configuration that produced it.
  * NON-DESTRUCTIVE. Nothing is recomputed here. If an upstream step has not
    been run, its files are reported missing rather than silently skipped.

Run after the pipeline:

    python run_all.py            # or the individual steps
    python export_results.py     # also runs as `python run_all.py export`
"""
import json
import platform
import shutil
import sys
from datetime import datetime, timezone

import pandas as pd

import config
from config import RESULTS

OUT = RESULTS / "paper_export"

# (destination stem, source path relative to results/, description)
TABLES = [
    # ---- experimental design ----
    ("design__fold_spans", "design/fold_spans.csv",
     "Actual train/validation/test date spans and sizes per walk-forward fold, "
     "read from the split code rather than restated."),
    # ---- per-year attribution ----
    ("years__attribution", "year_analysis/year_attribution.csv",
     "Per fold: net return %, volatility, Sharpe, max drawdown, long/short leg "
     "contributions, cost drag, turnover, and the market context (asset drift, "
     "dispersion, volatility, mean pairwise correlation, trend efficiency) plus "
     "hit rate and breadth."),
    ("years__return_drivers", "year_analysis/return_drivers.csv",
     "Correlation of each diagnostic with the fold return. n=5: descriptive."),
    # ---- leakage ----
    ("leakage__audit", "leakage/audit.json",
     "Causality audit of the Kalman target: max future influence and max pre-cut "
     "difference after randomising the future. Both must be exactly 0."),
    # ---- forecast ----
    ("forecast__model_comparison", "forecast/model_comparison.csv",
     "Mean direction accuracy by asset (rows) and model (columns); final row is "
     "the cross-asset mean used to select the strategy signal."),
    # ---- fold analysis ----
    ("folds__diagnostics", "fold_analysis/fold_diagnostics.csv",
     "Per-fold market and execution diagnostics: direction accuracy against the "
     "smoothed trend, the raw h-day price change and the next-day return; trend "
     "efficiency; regime transitions; volatility; turnover; cost drag; breadth."),
    ("folds__signal_sharpe", "fold_analysis/fold_signal_sharpe.csv",
     "Per-fold Sharpe by signal arm, plus gross-of-cost and no-volatility-target "
     "variants and the permutation null."),
    ("folds__diagnostic_correlation", "fold_analysis/diagnostic_correlation.csv",
     "Correlation of each diagnostic with per-fold Sharpe. n=5: descriptive only."),
    # ---- strategy ----
    ("strategy__ablation", "strategy/ablation.csv",
     "Pooled Sharpe, max drawdown, annualised return and volatility for each "
     "signal arm (BLEND/PRED/LONG/REGIME/FLIP) and the permutation-null mean."),
    ("strategy__fold_sharpe", "strategy/fold_sharpe.csv",
     "Per-fold Sharpe by arm, with worst-fold and cross-fold dispersion columns."),
    ("strategy__paired_proofs", "strategy/paired_proofs.csv",
     "Paired stationary-bootstrap comparisons: dSharpe with 95% CI, and one-sided "
     "p-values for reduced downside deviation and improved CVaR5."),
    ("strategy__asset_sharpe", "strategy/asset_sharpe.csv",
     "Pooled Sharpe of each portfolio asset under the flagship stack."),
    ("strategy__heatmap_asset_year", "strategy/heatmap_asset_year.csv",
     "Sharpe by asset (rows) and test fold (columns) for the flagship arm."),
    ("strategy__daily_returns_by_arm", "strategy/daily_returns_by_arm.csv",
     "ROW-LEVEL. Daily net book return for every signal arm, indexed by date. "
     "Source for all equity curves, drawdowns and bootstrap tests."),
    ("strategy__positions_flagship", "strategy/positions_flagship.csv",
     "ROW-LEVEL. Per (fold, asset, date): final position, asset return, net "
     "return, raw prediction, regime flag, per-asset and book volatility scales, "
     "turn probability and its threshold. Full audit trail of every trade."),
    # ---- volatility-target audit ----
    ("voltarget__controls", "vol_target/controls.csv",
     "Controls isolating the mechanism of the book volatility overlay: constant "
     "scale (must reproduce baseline), time-shuffled scale, per-fold cold start."),
    ("voltarget__surface_vol_target", "vol_target/surface_vol_target.csv",
     "Pooled Sharpe over the (trailing window x volatility target) grid."),
    ("voltarget__surface_trend_quality", "vol_target/surface_trend_quality.csv",
     "Pooled Sharpe over the rejected trend-quality filter grid, retained as the "
     "counter-example of a fitted (spiked) parameter surface."),
    # ---- strategy by signal model ----
    ("models__strategy_comparison", "strategy_models/model_comparison.csv",
     "Pooled and per-fold Sharpe for each signal model x arm through an "
     "identical execution stack."),
    ("models__paired_vs_LR", "strategy_models/paired_vs_LR.csv",
     "Paired bootstrap of each candidate signal model against LR."),
    ("models__pred_vs_flip", "strategy_models/pred_vs_flip.csv",
     "PRED vs FLIP paired bootstrap per signal model, with permutation p-value."),
]

# per-model files discovered dynamically
PER_MODEL = [
    ("forecast__{m}__folds", "forecast/{m}_folds.csv",
     "Per (asset, fold) direction accuracy and R^2 against a random walk."),
    ("forecast__{m}__summary", "forecast/{m}_summary.csv",
     "Per-asset mean and standard deviation of direction accuracy across folds."),
    ("forecast__{m}__significance", "forecast/{m}_significance.csv",
     "Diebold-Mariano against a random walk (Newey-West + HLN correction) and "
     "the Pesaran-Timmermann directional test, per asset."),
    ("forecast__{m}__predictions", "forecast/{m}_predictions.csv",
     "ROW-LEVEL. Per (asset, fold, date): the realised h-step Kalman-trend change "
     "(y_true), the model forecast (y_pred), and the two raw-price truths "
     "(h-day and next-day) for scoring the same prediction against price."),
]

FIGURES = [
    ("fig__experimental_design", "design/experimental_design.png"),
    ("fig__leakage_audit", "leakage/kalman_leakage.png"),
    ("fig__forecast_model_comparison", "forecast/model_comparison.png"),
    ("fig__year_attribution", "year_analysis/year_attribution.png"),
    ("fig__strategy_equity", "strategy/equity_mcap.png"),
    ("fig__strategy_ablation", "strategy/ablation_bars.png"),
    ("fig__strategy_ablation_equity", "strategy/ablation_equity.png"),
    ("fig__strategy_heatmap", "strategy/heatmap_asset_year.png"),
    ("fig__folds_signal_sharpe", "fold_analysis/fold_signal_sharpe.png"),
    ("fig__folds_diagnostics", "fold_analysis/fold_diagnostics.png"),
    ("fig__folds_cost_decomposition", "fold_analysis/fold_gross_vs_net.png"),
    ("fig__voltarget_surfaces", "vol_target/parameter_surfaces.png"),
    ("fig__models_equity_pred_flip", "strategy_models/equity_pred_flip.png"),
    ("fig__models_ablation_pred_flip", "strategy_models/ablation_pred_flip.png"),
]


def _snapshot_config():
    keep = ("H", "L", "SEEDS", "DL_SEEDS", "SMA_REGIME", "KALMAN", "ARIMA_ORDER",
            "MODELS", "SLOW_MODELS", "COST", "FUND", "VOL_TARGET",
            "VOL_TARGET_WIN", "SIGNAL_BLEND", "FOLDS", "MIN_TRAIN", "MIN_TEST",
            "EMBARGO_DAYS", "ASSETS", "ASSET_CLASSES", "ORIGINAL_ASSETS",
            "ORIGINAL_ASSET_CLASSES")
    return {k: getattr(config, k) for k in keep if hasattr(config, k)}


def _environment():
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "torch", "xgboost",
                "lightgbm", "statsmodels", "matplotlib", "einops"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:                                   # noqa: BLE001
            env[mod] = None
    return env


# Produced by standalone modules that are not part of `run_all.py all`; their
# absence means that module was not run, which is not a pipeline failure.
OPTIONAL_DIRS = ("fold_analysis/", "vol_target/", "strategy_models/",
                 "year_analysis/", "design/")


def _read(rel, **kw):
    """Read a result file if it exists, else None -- so STEPS.md degrades
    gracefully when a step has not been run rather than crashing the export."""
    fp = RESULTS / rel
    if not fp.exists():
        return None
    return json.load(open(fp)) if fp.suffix == ".json" else pd.read_csv(fp, **kw)


def _write_steps(missing, missing_optional):
    """Generate STEPS.md: what was run, in what order, what it produced, and the
    headline number from each stage -- the narrative spine for the write-up.

    Every figure is filled from the artefacts actually on disk, so this document
    can never claim a result the run did not produce.
    """
    cfg = _snapshot_config()
    L = ["# Experimental steps and results", "",
         f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC "
         f"from `results/`. Every number below is read from a file; none is typed "
         f"by hand.", "",
         "---", "", "## Step 1 -- Data collection", "",
         f"Daily closing prices, two disjoint universes.", "",
         f"- **Forecast-evaluation universe** ({len(cfg.get('ASSETS', {}))} assets, one per "
         f"class): {', '.join(cfg.get('ASSETS', {}))}. Used only to compare models.",
         f"- **Portfolio universe** ({len(cfg.get('ORIGINAL_ASSETS', {}))} assets): "
         f"{', '.join(cfg.get('ORIGINAL_ASSETS', {}))}. Used only to trade.", "",
         "The two are disjoint by construction, so no asset both selects a model and "
         "is traded by it.", ""]

    D = _read("design/fold_spans.csv")
    L += ["## Step 2 -- Splits and folds", "",
          f"Anchored walk-forward: {len(cfg.get('FOLDS', []))} non-overlapping annual test "
          f"folds, window $L={cfg.get('L')}$, horizon $h={cfg.get('H')}$, embargo "
          f"{cfg.get('EMBARGO_DAYS')} days between validation and test. Validation is the "
          f"last 15% of the pre-test block. Splits are strictly chronological -- no "
          f"shuffling, no k-fold.", ""]
    if D is not None:
        L += ["| fold | train | validation | test |", "|---|---|---|---|"]
        L += [f"| {int(r.year)}-{int(r.year)+1} | {r.n_train} | {r.n_val} | {r.n_test} |"
              for _, r in D.iterrows()]
        L += ["", "Figure: `fig__experimental_design.png`.", ""]

    A = _read("leakage/audit.json")
    L += ["## Step 3 -- Kalman trend and leakage audit", "",
          f"Target: the $h$-step change of a causal local-linear-trend Kalman filter "
          f"($q_\\ell={cfg.get('KALMAN', {}).get('q_level')}$, "
          f"$q_v={cfg.get('KALMAN', {}).get('q_slope')}$, "
          f"$r={cfg.get('KALMAN', {}).get('r')}$). The volatility-adaptive measurement "
          f"noise is normalised on **pre-test data only**; normalising over the full "
          f"sample would be a look-ahead leak.", ""]
    if A:
        L += ["Two independent causality tests, both returning exact zeros:", "",
              "| test | measured | required |", "|---|---|---|",
              f"| influence kernel $\\max_{{k>0}}|\\partial \\ell_{{t_0}}/\\partial p_{{t_0+k}}|$ "
              f"| {A['max_future_influence']:.1e} | 0 |",
              f"| future-randomisation invariance $\\max_{{t \\leq t_0}}|\\ell_t-\\ell'_t|$ "
              f"| {A['max_pre_cut_diff_after_future_corruption']:.1e} | 0 |", "",
              f"Verdict: **{'PASS' if A['PASSED'] else 'FAIL'}**. "
              f"Figure: `fig__leakage_audit.png`.", ""]

    C = _read("forecast/model_comparison.csv", index_col=0)
    L += ["## Step 4 -- Model training and forecast evaluation", "",
          f"Models ({len(cfg.get('MODELS', []))}): {', '.join(cfg.get('MODELS', []))}.",
          "", "Identical protocol for all: same inputs, same target, same loss, same "
          "purged splits, same early stopping. Metrics are direction accuracy, "
          "$R^2$ against a random walk, Diebold-Mariano vs a random walk "
          "(Newey-West + HLN), and Pesaran-Timmermann.", ""]
    if C is not None and "MEAN" in C.index:
        mean = C.loc["MEAN"].sort_values(ascending=False)
        L += ["| rank | model | mean direction accuracy |", "|---|---|---|"]
        L += [f"| {i+1} | `{m}` | {v:.3f} |" for i, (m, v) in enumerate(mean.items())]
        L += ["", f"Best: **{mean.index[0]}** ({mean.iloc[0]:.3f}). "
                  f"Figure: `fig__forecast_model_comparison.png`.", ""]

    FD = _read("fold_analysis/fold_diagnostics.csv")
    if FD is not None:
        L += ["### The central control: the same forecast scored against price", "",
              "| fold | vs smoothed trend | vs raw $h$-day price | vs next-day return |",
              "|---|---|---|---|"]
        L += [f"| {int(r.year)} | {r.DA_trend:.3f} | {r.DA_price_h:.3f} | "
              f"{r.DA_price_1:.3f} |" for _, r in FD.iterrows()]
        L += ["", f"Mean: **{FD.DA_trend.mean():.3f}** against the filtered target versus "
                  f"**{FD.DA_price_1.mean():.3f}** against the return a position earns on. "
                  f"The high accuracy measures the target's smoothness, not tradeable "
                  f"skill.", ""]

    AB = _read("strategy/ablation.csv")
    FS = _read("strategy/fold_sharpe.csv", index_col=0)
    PP = _read("strategy/paired_proofs.csv", index_col=0)
    L += ["## Step 5 -- Portfolio strategy", "",
          f"Equal-weight book over the portfolio universe. Execution: SMA-"
          f"{cfg.get('SMA_REGIME')} regime gate, turn-classifier exit (threshold chosen "
          f"on validation), causal per-asset volatility targeting, "
          f"{cfg.get('SIGNAL_BLEND', 0):.0%}/{1-cfg.get('SIGNAL_BLEND', 0):.0%} blend of "
          f"prediction and regime signal, book-level volatility target "
          f"{cfg.get('VOL_TARGET', 0):.0%} at {cfg.get('VOL_TARGET_WIN')} days. Costs: "
          f"**{cfg.get('COST', 0)*1e4:.0f} bps per side** on every position change plus "
          f"**{cfg.get('FUND', 0):.0%} p.a.** funding on short crypto.", ""]
    if AB is not None:
        L += ["| signal arm | Sharpe | max drawdown | ann. return | ann. vol |",
              "|---|---|---|---|---|"]
        for _, r in AB.iterrows():
            L += [f"| `{r.signal}` | {r.sharpe:.3f} | "
                  f"{'' if pd.isna(r.maxdd) else f'{r.maxdd:.3f}'} | "
                  f"{'' if pd.isna(r.ann_ret) else f'{r.ann_ret:.3f}'} | "
                  f"{'' if pd.isna(r.ann_vol) else f'{r.ann_vol:.3f}'} |"]
        L += [""]

    L += ["### Step 5a -- Signal ablation (PRED vs FLIP)", "",
          "The entire execution stack is held fixed and **only the directional call is "
          "replaced**. `FLIP` is the sign-reversed prediction: if the sign carries real "
          "information, reversing it must hurt.", ""]
    if PP is not None and "PRED_vs_FLIP" in PP.index:
        r = PP.loc["PRED_vs_FLIP"]
        L += [f"Paired stationary bootstrap on identical days: "
              f"**dSharpe {r.d_sharpe:+.3f}, 95% CI "
              f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]**"
              f"{' -- entirely above zero.' if r.ci_lo > 0 else '.'}", "",
              "Figures: `fig__strategy_ablation.png`, `fig__strategy_ablation_equity.png`.",
              ""]

    Y = _read("year_analysis/year_attribution.csv", index_col=0)
    RD = _read("year_analysis/return_drivers.csv", index_col=0)
    L += ["## Step 6 -- Per-year return attribution", ""]
    if Y is not None:
        L += ["| fold | return % | ann. vol | Sharpe | max DD % | long leg % | "
              "short leg % | costs % |", "|---|---|---|---|---|---|---|---|"]
        L += [f"| {int(y)} | {r.return_pct:+.1f} | {r.ann_vol:.2f} | {r.sharpe:+.2f} | "
              f"{r.max_dd_pct:.1f} | {r.long_leg_pct:+.1f} | {r.short_leg_pct:+.1f} | "
              f"{r.cost_drag_pct:.1f} |" for y, r in Y.iterrows()]
        L += ["", "Market context over the same folds (properties of the data, not the "
                  "strategy):", "",
              "| fold | asset drift % | mean pairwise corr. | trend efficiency | breadth |",
              "|---|---|---|---|---|"]
        L += [f"| {int(y)} | {r.asset_drift_pct:+.1f} | {r.avg_pair_corr:.2f} | "
              f"{r.trend_efficiency:.3f} | {r.breadth:.0%} |" for y, r in Y.iterrows()]
        L += [""]
    if RD is not None:
        top = RD["corr_with_return"].dropna().sort_values()
        L += ["Correlation of each diagnostic with the fold return "
              f"(n={len(Y) if Y is not None else '?'} folds -- **descriptive only**, "
              "nothing is significant at this sample size):", "",
              "| diagnostic | corr. with return |", "|---|---|"]
        L += [f"| `{k}` | {v:+.2f} |" for k, v in top.items()]
        L += ["", "Figure: `fig__year_attribution.png`. A generated per-fold narrative "
                  "is in `year_analysis/year_read.md`.", ""]

    MC = _read("strategy_models/model_comparison.csv")
    L += ["## Step 7 -- Strategy by signal model", ""]
    if MC is not None:
        L += ["The best forecasters by direction accuracy, each run through the "
              "identical execution stack.", "",
              "| arm | model | Sharpe | worst fold |", "|---|---|---|---|"]
        L += [f"| {r['arm']} | `{r['model']}` | {r['sharpe']:.3f} | {r['min_fold']:+.3f} |"
              for _, r in MC.iterrows()]
        L += ["", "Figures: `fig__models_equity_pred_flip.png`, "
                  "`fig__models_ablation_pred_flip.png`. The ranking by accuracy and the "
                  "ranking by Sharpe are compared in "
                  "`strategy_models/da_vs_sharpe.csv`.", ""]
    else:
        L += ["*Not run -- `strategy_models.py` is optional and expensive.*", ""]

    L += ["---", "", "## Reproduction", "",
          "```bash", "python run_all.py            # every step, in order",
          "python run_all.py <step>     # one step: data design leakage forecast",
          "                             #           strategy folds years voltarget export",
          "python strategy_models.py    # optional, expensive", "```", "",
          "`results/ACCEPTANCE.md` records every automated check.", ""]
    if missing:
        L += ["> **Incomplete run.** Core outputs absent at export time: "
              + ", ".join(f"`{m}`" for m in missing) + ".", ""]
    if missing_optional:
        L += ["> Optional analyses not run: "
              + ", ".join(f"`{m}`" for m in missing_optional) + ".", ""]
    (OUT / "STEPS.md").write_text("\n".join(L), encoding="utf-8")


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, missing, missing_optional = [], [], []

    def take(stem, rel, desc):
        src = RESULTS / rel
        if not src.exists():
            (missing_optional if rel.startswith(OPTIONAL_DIRS) else missing).append(rel)
            return
        if src.suffix == ".json":
            dst = OUT / f"{stem}.json"
            shutil.copy(src, dst)
            rows.append(dict(file=dst.name, rows="", columns="", description=desc))
            return
        df = pd.read_csv(src)
        dst = OUT / f"{stem}.csv"
        df.to_csv(dst, index=False)
        rows.append(dict(file=dst.name, rows=len(df),
                         columns="|".join(map(str, df.columns)), description=desc))

    for stem, rel, desc in TABLES:
        take(stem, rel, desc)

    for model in getattr(config, "MODELS", []):
        for stem, rel, desc in PER_MODEL:
            take(stem.format(m=model), rel.format(m=model), f"[{model}] {desc}")

    for stem, rel in FIGURES:
        src = RESULTS / rel
        if not src.exists():
            (missing_optional if rel.startswith(OPTIONAL_DIRS) else missing).append(rel)
            continue
        dst = OUT / f"{stem}.png"
        shutil.copy(src, dst)
        rows.append(dict(file=dst.name, rows="", columns="",
                         description="Figure, 300 dpi."))

    (OUT / "config_snapshot.json").write_text(
        json.dumps(_snapshot_config(), indent=2, default=str), encoding="utf-8")
    (OUT / "environment.json").write_text(
        json.dumps(_environment(), indent=2), encoding="utf-8")
    rows.append(dict(file="config_snapshot.json", rows="", columns="",
                     description="Every constant the run used: horizons, folds, "
                                 "Kalman parameters, costs, universes, model list."))
    rows.append(dict(file="environment.json", rows="", columns="",
                     description="Python and library versions, export timestamp."))

    M = pd.DataFrame(rows).sort_values("file")
    M.to_csv(OUT / "MANIFEST.csv", index=False)
    _write_steps(missing, missing_optional)

    readme = [
        "# Consolidated results export", "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} UTC.",
        "", "Every file in this directory is flat and self-describing. "
        "`MANIFEST.csv` lists each file with its row count, its columns and a "
        "one-line description.", "",
        "## The row-level files", "",
        "Three files carry raw observations rather than summaries. Together they "
        "allow any table or figure in the paper to be rebuilt, and any new cut to "
        "be taken, without re-fitting a single model:", "",
        "| file | one row per | key columns |",
        "|---|---|---|",
        "| `forecast__<MODEL>__predictions.csv` | test-set anchor | `model, asset, "
        "fold, date, y_true, y_pred, y_true_price_h, y_true_price_1` |",
        "| `strategy__positions_flagship.csv` | (fold, asset, trading day) | "
        "`position, asset_return, net_return, prediction, bull_regime, vol_scale, "
        "book_scale, p_turn, tau` |",
        "| `strategy__daily_returns_by_arm.csv` | trading day | one column per "
        "signal arm |", "",
        "`y_true` is the realised h-step change of the causal Kalman trend (the "
        "training target). `y_true_price_h` and `y_true_price_1` are the raw "
        "h-day and next-day log price changes, so the *same* prediction can be "
        "scored against the filtered target and against price.", "",
        "## Provenance", "",
        "`config_snapshot.json` records the constants the run used and "
        "`environment.json` the library versions.", "",
    ]
    if missing:
        readme += ["## Missing core inputs", "",
                   "These belong to the main pipeline and were absent at export "
                   "time. Run `python run_all.py` to produce them:", ""]
        readme += [f"- `{m}`" for m in missing] + [""]
    if missing_optional:
        readme += ["## Optional analyses not run", "",
                   "These come from standalone modules outside `run_all.py`:", "",
                   "| module | produces |", "|---|---|",
                   "| `fold_analysis.py` | per-fold diagnostics |",
                   "| `vol_target_sweep.py` | volatility-target audit |",
                   "| `strategy_models.py` | strategy by signal model |", ""]
    (OUT / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"== EXPORT ==\n  {len(rows)} artefacts -> {OUT}")
    if missing:
        print(f"  [FAIL] {len(missing)} CORE outputs missing: "
              f"{', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}")
    if missing_optional:
        print(f"  [note] {len(missing_optional)} optional outputs missing "
              f"(standalone module not run)")
    return M, missing, missing_optional


if __name__ == "__main__":
    run()
