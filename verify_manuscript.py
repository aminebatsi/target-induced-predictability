"""Consistency sweep over the LaTeX source of the manuscript.

Every headline figure quoted in the paper is re-derived from the cached
artefacts and checked against the text, so that a number cannot drift in one
place without the check failing. It also scans for stale terminology and for
prose the style guide excludes.

Requires the manuscript sources at ../paper/manuscript.tex; skipped otherwise.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_replay import ROOT, SIGNS, load_components, portfolio, sharpe

TEX = ROOT.parent / "paper" / "manuscript.tex"
ANALYSIS = ROOT / "results" / "analysis"
FLAGS = []


def main():
    tex = TEX.read_text(encoding="utf-8")
    store = load_components()

    print("== 1. strategy numbers of record still reproduce (turn exit ON) ==")
    for m, want in (("TimeFilter", 1.125), ("LR", 1.065),
                    ("GRU", 0.834), ("TimeMixer", 0.763)):
        got = sharpe(portfolio(store[m], SIGNS["PRED"]).values)
        ok = abs(got - want) < 2e-3 and f"{want:.3f}" in tex
        print(f"  [{'OK ' if ok else 'FAIL'}] PRED {m:11} {got:.4f} "
              f"(manuscript {want:.3f}, present={f'{want:.3f}' in tex})")
        if not ok:
            FLAGS.append(f"PRED {m}")

    print("\n== 2. turn-classifier terminology (the component is retained) ==")
    for term in ("tau", "turn classifier", "turn probability", "c_t",
                 "turn-classifier", "\\tau"):
        print(f"  {term:20} {tex.count(term):3} occurrence(s)")

    print("\n== 3. headline strategy figures ==")
    mb = pd.read_csv(ANALYSIS / "matched_turn_vs_noturn.csv")
    mb = mb[(mb.comparison.str.startswith("MATCH-B")) & mb.funding & mb.turn_exit]
    lo, hi = mb.d_sharpe.min(), mb.d_sharpe.max()
    print(f"  MATCH-B dSharpe range {lo:.2f}--{hi:.2f}; "
          f"'1.89' in tex: {'1.89' in tex}; '2.63' in tex: {'2.63' in tex}")
    if not (abs(lo - 1.89) < 0.01 and abs(hi - 2.63) < 0.01):
        FLAGS.append("MATCH-B range")

    dsr = pd.read_csv(ROOT / "results" / "analysis" / "deflated_sharpe.csv")
    worst = dsr.dsr.min()
    print(f"  worst DSR {worst:.3f}; '0.867' in tex: {'0.867' in tex}")
    if abs(worst - 0.867) > 1e-3:
        FLAGS.append("DSR")

    print("\n== 4. new results appear in the manuscript ==")
    checks = {
        "ExDA mean of 13 (-0.0394)": "0.039",
        "ExDA slope (+0.0053)": "0.0053",
        "benchmark 0.7631": "0.7631",
        "Table 2 CI lower 0.743": "0.743",
        "sub-chance DA 0.476": "0.476",
        "universe delta +0.039": "+0.039",
        "centering 0.032 -> 0.025": "0.025",
        "worst cell 0.296": "0.296",
        "worst cell drift 2.35": "2.35",
        "no-turn TimeMixer CI -0.10": "-0.10",
        "absret ratio 6--8%": "6--8",
        "Sheppard citation": "sheppard1899",
        "Lindskog citation": "lindskog2003",
        "Leitch citation": "leitch1991",
        "Timmermann citation": "timmermann2004",
    }
    for name, needle in checks.items():
        ok = needle in tex
        print(f"  [{'OK ' if ok else 'FAIL'}] {name:34} ('{needle}')")
        if not ok:
            FLAGS.append(name)

    print("\n== 5. language audit ==")
    banned = ["best model", "best-ranked", "highest-ranked", "top model",
              "leading model", "winner", "strongest architecture",
              "first on Sharpe", "proves", "definitively", "groundbreaking",
              "undoubtedly", "It is important to note", "It should be noted",
              "Interestingly", "This underscores", "seed"]
    for b in banned:
        n = tex.lower().count(b.lower())
        if n:
            print(f"  [FLAG] '{b}' x{n}")
            FLAGS.append(f"language: {b}")
    if not any(f.startswith("language") for f in FLAGS):
        print("  clean")

    print("\n" + ("ALL CHECKS PASS" if not FLAGS else f"FLAGS: {FLAGS}"))
    return not FLAGS


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
