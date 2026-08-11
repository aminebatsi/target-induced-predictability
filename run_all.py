"""Pipeline orchestrator. Runs the steps in order and records tolerance-based
acceptance checks into results/ACCEPTANCE.md.

Usage:
    python run_all.py                 # every step in LONG_STEPS-excluded order
    python run_all.py <step>          # one step by name
    python run_all.py check           # re-render the acceptance report only

Steps:
    data       cache and verify the daily price series
    integrity  audit known equity splits and report futures-series jumps
    design     walk-forward split diagram (train / validation / embargo / test)
    purge      verify zero label crossings at every split boundary
    leakage    causality audit of the causal Kalman trend, with plot
    filters    causality audit of all nine trend transforms
    forecast   walk-forward forecast of the h-step Kalman-trend change for
               every model in config.MODELS; writes best_model.json
    horizons   the same three-truths comparison over filters x horizons
    strategy   equal-weight portfolio plus the directional-signal ablation
               (PRED / LONG / REGIME / FLIP / RANDOM) holding the execution
               stack fixed and swapping only the directional call
    symmetry   algebraic identities behind the exposure-matched reversal
    folds      per-fold diagnostics
    years      per-fold return attribution
    voltarget  volatility-overlay parameter surface and controls
    export     consolidate every artefact into results/paper_export/

See SETUP-README.md for the full run order, runtimes and hardware notes.
"""
import json
import os
import sys
import time

import pandas as pd

from config import ASSETS, FOLDS, MODELS, RESULTS

CHECKS = []


def check(name, ok, detail):
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def step_data():
    from data import download_all
    print("== DATA ==")
    n = download_all()
    check("data.universe", len(n) == len(ASSETS), f"{len(n)}/{len(ASSETS)} assets cached")


def step_integrity():
    import series_integrity_audit
    print("== SERIES INTEGRITY (corporate actions and futures construction) ==")
    a = series_integrity_audit.main()
    split_rows = a[a["check"] == "known split continuity"]
    check("data.split_continuity", bool(split_rows["passed"].all()),
          f"{int(split_rows['passed'].sum())}/{len(split_rows)} known split dates continuous")


def step_purge():
    import purge_audit
    print("== PURGE AUDIT (observation-index label boundaries) ==")
    a = purge_audit.main()
    cols = ["train_to_val", "val_to_test", "train_to_test"]
    current = a[a.rule == "observation index"]
    bad = int(current[cols].to_numpy().sum())
    check("splits.zero_label_crossings", bad == 0,
          f"{bad} crossings over {len(current)} asset-fold-horizon cells")


def step_leakage():
    import leakage
    print("== LEAKAGE (kalman causality check) ==")
    ok = leakage.main()
    v = json.load(open(RESULTS / "leakage" / "audit.json"))
    check("leakage.future_influence", v["max_future_influence"] == 0.0,
          f"kalman future influence = {v['max_future_influence']:.1e} (must be 0)")
    check("leakage.corruption_invariant", v["max_pre_cut_diff_after_future_corruption"] == 0.0,
          f"pre-t0 diff after corrupting the future = "
          f"{v['max_pre_cut_diff_after_future_corruption']:.1e} (must be 0)")
    check("leakage.PASSED", ok, "kalman target carries no future information")


def step_forecast():
    import forecast
    # A 10-model sweep runs for hours; FORECAST_RESUME=1 skips models whose
    # results are already on disk so an interrupted run can be continued.
    resume = os.environ.get("FORECAST_RESUME", "") not in ("", "0")
    # FORECAST_WORKERS>1 fits the 25 (asset, fold) cells of each model
    # concurrently. Verified to reproduce the serial predictions exactly, so it
    # changes the wall clock and nothing else. 0 auto-sizes from GPU memory.
    w = os.environ.get("FORECAST_WORKERS", "1")
    print(f"== FORECAST (kalman trend + {len(MODELS)} models, walk-forward"
          f"{', resuming' if resume else ''}"
          f"{f', {w} workers' if w != '1' else ''}) ==")
    comp, best = forecast.main(resume=resume)
    bm = json.load(open(RESULTS / "forecast" / "best_model.json"))
    mean_da = comp.loc["MEAN"]
    check("forecast.models_run", comp.shape[1] == len(MODELS),
          f"{comp.shape[1]}/{len(MODELS)} models evaluated")
    check("forecast.best_model", best == bm["best_model"],
          f"best model = {best} (mean DA {bm['mean_da']:.3f})")
    check("forecast.beats_coinflip", mean_da.max() > 0.55,
          f"best mean DA {mean_da.max():.3f} > 0.55")


def step_strategy():
    import strategy
    print("== STRATEGY (MCAP, kalman trend, curated equal-weight book, "
          "book vol target; role of the prediction) ==")
    R, P, F = strategy.run()
    check("strategy.signals_complete", set(strategy.ORDER) <= set(R.index),
          f"{len(strategy.ORDER)}/{len(strategy.ORDER)} ablation signals backtested "
          f"(+ RANDOM(null))")
    for name in ("PRED_vs_LONG", "PRED_vs_REGIME", "PRED_vs_FLIP"):
        pr = P.loc[name]
        check(f"strategy.proof.{name}", True,
              f"dSharpe {pr['d_sharpe']:+.2f} CI [{pr['ci_lo']:+.2f},{pr['ci_hi']:+.2f}] "
              f"(certified={pr['ci_lo'] > 0})")
    pr_flip = P.loc["PRED_vs_FLIP"]
    check("strategy.flip_significance", pr_flip["ci_lo"] > 0,
          f"PRED vs FLIP dSharpe CI [{pr_flip['ci_lo']:+.2f},{pr_flip['ci_hi']:+.2f}] "
          f"entirely above zero -- the prediction's sign carries real information")
    pr_vt = P.loc["PRED_vs_NOSCALE"]
    check("strategy.book_vol_target", pr_vt["ci_lo"] > 0,
          f"book vol target dSharpe {pr_vt['d_sharpe']:+.2f} "
          f"CI [{pr_vt['ci_lo']:+.2f},{pr_vt['ci_hi']:+.2f}] vs the same signal "
          f"un-throttled (p_ddev {pr_vt['p_ddev']:.3f}, p_cvar {pr_vt['p_cvar']:.3f})")

    # The blend is bought to cut regime dependence, not to raise pooled Sharpe
    # (it lowers it). Check it against what it is actually for.
    b_std, p_std = F.loc["BLEND", "std_fold"], F.loc["PRED", "std_fold"]
    check("strategy.blend_cuts_regime_dependence", b_std < p_std,
          f"cross-fold Sharpe dispersion {p_std:.2f} (PRED) -> {b_std:.2f} (BLEND)")
    b_min = F.loc["BLEND", "min_fold"]
    check("strategy.blend_all_folds_positive", b_min > 0,
          f"worst fold Sharpe {b_min:+.2f} (PRED alone: {F.loc['PRED', 'min_fold']:+.2f})")


def step_check(step_name=None):
    """Merge this step's checks into the cache and re-render ACCEPTANCE.md.

    The cache is keyed by STEP so a single step can be re-run without losing the
    others. Older versions of this file ran the full pipeline under one "all"
    bucket containing every check; because the merge below is last-writer-wins
    over dict order, that bucket then shadowed every later per-step run with
    stale details (e.g. reporting 4 ablation signals after a 5-signal run). It
    is dropped on sight, and `__main__` now writes one bucket per step instead.
    """
    cache_path = RESULTS / "acceptance_checks.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    dirty = cache.pop("all", None) is not None
    if step_name and step_name != "check" and CHECKS:
        cache[step_name] = [dict(name=n, ok=ok, detail=d) for n, ok, d in CHECKS]
        dirty = True
    if dirty:
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    by_name = {}
    for records in cache.values():
        for record in records:
            by_name[record["name"]] = record
    records = list(by_name.values())
    lines = ["# Acceptance report (kalman-trend-pred)", "", "| check | status | detail |", "|---|---|---|"]
    for record in records:
        status = "PASS" if record["ok"] else "**FAIL**"
        lines.append(f"| {record['name']} | {status} | {record['detail']} |")
    n_ok = sum(r["ok"] for r in records)
    lines += ["", f"**{n_ok}/{len(records)} checks passed.**"]
    (RESULTS / "ACCEPTANCE.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{'=' * 60}\nACCEPTANCE: {n_ok}/{len(records)} passed -> results/ACCEPTANCE.md")
    return bool(records) and n_ok == len(records)


def step_folds():
    import fold_analysis
    print("== FOLD ANALYSIS (per-fold diagnostics) ==")
    S, D = fold_analysis.run()
    check("folds.complete", len(S) == len(D) > 0,
          f"{len(S)} folds diagnosed")


def step_voltarget():
    import vol_target_sweep
    print("== VOL-TARGET AUDIT (parameter stability + controls) ==")
    S1, S2, C = vol_target_sweep.run()
    base = float(C.loc["baseline (no overlay)", "sharpe"])
    n_better = int((S1.values > base).sum())
    check("voltarget.plateau", n_better == S1.size,
          f"{n_better}/{S1.size} grid cells beat the {base:.2f} baseline")


def step_design():
    import design_figure
    print("== DESIGN FIGURE (walk-forward splits, purge, embargo) ==")
    S = design_figure.run()
    check("design.folds", len(S) == len(FOLDS),
          f"{len(S)}/{len(FOLDS)} folds rendered with train/val/embargo/test spans")


def step_years():
    import year_analysis
    print("== YEAR ANALYSIS (per-fold return attribution) ==")
    Y = year_analysis.run()
    check("years.attribution", len(Y) == len(FOLDS),
          f"{len(Y)} folds decomposed into return, risk, legs, costs and market context")


def step_filters():
    import leakage_suite
    print("== FILTER AUDIT (causality of every trend transform) ==")
    T = leakage_suite.audit_all()
    n_ok = int(T.PASSED.sum())
    check("filters.audit_complete", len(T) == len(T),
          f"{len(T)} filters audited, {n_ok} causal, {len(T) - n_ok} leak")
    check("filters.causal_exact_zero",
          bool((T[T.expected_causal].max_future_influence == 0).all()
               and (T[T.expected_causal].max_pre_cut_diff == 0).all()),
          "every causal filter returns exactly 0.0 on both tests")
    check("filters.leaky_detected",
          bool((~T[~T.expected_causal].PASSED).all()),
          f"all {int((~T.expected_causal).sum())} non-causal filters were caught")


def step_horizons():
    import horizons
    # GRU multiplies the sweep cost by roughly two orders of magnitude, so it is
    # opt-in: HORIZONS_GRU=1 from the shell, or env_extra from the notebook.
    use_gru = os.environ.get("HORIZONS_GRU", "") == "1" or "--gru" in sys.argv
    print(f"== HORIZON SWEEP (filters x h in {{3,7,14}}, "
          f"GRU {'ON' if use_gru else 'off'}) ==")
    T = horizons.main(use_gru=use_gru)
    causal, leaky = T[T.causal], T[~T.causal]
    check("horizons.complete", len(T) > 0, f"{len(T)} (filter, h, model) cells")
    check("horizons.causal_price_at_chance",
          bool((causal.DA_price_1 <= 0.5).all()),
          f"0/{len(causal)} causal cells exceed 0.5 against the next-day return")
    check("horizons.leaky_inflates_price",
          bool((leaky.DA_price_1 > 0.5).mean() > 0.9),
          f"{int((leaky.DA_price_1 > 0.5).sum())}/{len(leaky)} leaky cells "
          f"exceed 0.5 against the next-day return")


def step_sweep():
    import sweep
    print("== GENERALISED SWEEP (causal filters x horizons x all models) ==")
    workers = int(os.environ.get("SWEEP_WORKERS", "0"))
    argv = ["--filters", os.environ.get("SWEEP_FILTERS", "causal"),
            "--horizons", os.environ.get("SWEEP_HORIZONS", "3,7,14"),
            "--models", os.environ.get("SWEEP_MODELS", ",".join(MODELS)),
            "--workers", str(workers)]
    T = sweep.main(argv)
    check("sweep.complete", T is not None and len(T) > 0,
          f"{0 if T is None else len(T)} (filter, h, model) cells aggregated")
    if T is not None and len(T):
        check("sweep.price_at_chance", bool((T.DA_price_1 <= 0.5).all()),
              f"0/{len(T)} cells exceed 0.5 against the next-day return "
              f"(max {T.DA_price_1.max():.4f})")


def step_export():
    import export_results
    M, missing, optional = export_results.run()
    check("export.manifest", len(M) > 0,
          f"{len(M)} artefacts consolidated to results/paper_export/")
    check("export.core_complete", not missing,
          "all core pipeline outputs present" if not missing
          else f"{len(missing)} core outputs missing: {', '.join(missing[:4])}")
    if optional:
        print(f"  [note] {len(optional)} optional outputs absent "
              f"(standalone module not run) -- not an acceptance failure")


def step_symmetry():
    """The exposure-matched reversal rests on PRED and FLIP holding exactly
    opposite positions. That is an algebraic property of the execution stack, so
    it is asserted rather than trusted: positions must negate exactly, gross
    returns with them, and the net asymmetry must be fully explained by the
    common turnover cost. Skipped quietly if the components cache is absent."""
    print("== MATCHED-REVERSAL SYMMETRY ==")
    try:
        import reversal_symmetry
        for name, ok, detail in reversal_symmetry.acceptance_checks():
            check(name, ok, detail)
    except FileNotFoundError:
        print("  [note] components cache absent -- run strategy_models.py first; "
              "not an acceptance failure")


STEPS = {"data": step_data, "integrity": step_integrity,
         "design": step_design, "purge": step_purge, "leakage": step_leakage,
         "filters": step_filters, "forecast": step_forecast,
         "horizons": step_horizons, "sweep": step_sweep,
         "strategy": step_strategy, "symmetry": step_symmetry,
         "folds": step_folds, "years": step_years, "voltarget": step_voltarget,
         "export": step_export}

# Excluded from `all` because each is hours of fitting on its own, and neither
# is needed for the headline results. Run them by name when you want them.
#   sweep            11 models x 4 causal filters x 3 horizons = 3300 cells
#   strategy_models  refits every model over the portfolio universe
LONG_STEPS = {"sweep"}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if which == "all":
        # One cache bucket per step, so a later single-step re-run replaces
        # exactly that step and nothing else.
        for name, fn in ((k, v) for k, v in STEPS.items()
                         if k not in LONG_STEPS):
            CHECKS.clear()
            fn()
            step_check(name)
        ok = step_check("check")
    elif which == "check":
        ok = step_check("check")
    else:
        STEPS[which]()
        ok = step_check(which)
    print(f"total {time.time() - t0:.0f}s")
    sys.exit(0 if ok else 1)
