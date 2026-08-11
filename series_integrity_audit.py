"""Audit cached prices for mechanical equity-split and futures-roll jumps.

The forecasting loader deliberately uses the vendor's historical ``close``
field rather than dividend-adjusted close.  This audit checks that known stock
splits are nevertheless represented on a split-consistent historical scale and
reports the largest one-day changes in every series.  For the HG=F commodity
series it also records the important limitation: the chart endpoint supplies a
vendor-stitched front-month history but no contract identifier or roll flag, so
the loader performs no back-adjustment and cannot label individual roll days.

Outputs -> results/analysis/series_integrity_audit.csv
           results/analysis/series_integrity_summary.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

from config import ASSETS, ORIGINAL_ASSETS, RESULTS
from data import load_asset

OUT = RESULTS / "analysis"

# Effective trading dates of splits in the cached equity universe.  The audit
# is about continuity, not event discovery, so these dates are fixed inputs.
KNOWN_SPLITS = {
    "TSLA": ["2020-08-31", "2022-08-25"],
    "AAPL": ["2020-08-31"],
    "NVDA": ["2021-07-20", "2024-06-10"],
}


def audit():
    rows = []
    universe = list(dict.fromkeys([*ASSETS, *ORIGINAL_ASSETS]))
    for asset in universe:
        d = load_asset(asset).copy()
        d["log_return"] = np.log(d["close"]).diff()
        for i in d["log_return"].abs().nlargest(10).index:
            rows.append(dict(
                check="largest one-day changes", asset=asset,
                date=pd.Timestamp(d.loc[i, "date"]).date(),
                log_return=float(d.loc[i, "log_return"]),
                passed=True,
                note="descriptive tail audit; no row is deleted"))

        for day in KNOWN_SPLITS.get(asset, []):
            hit = d.index[d["date"] == pd.Timestamp(day)]
            if len(hit) != 1:
                rows.append(dict(check="known split continuity", asset=asset,
                                 date=day, log_return=np.nan, passed=False,
                                 note="split date absent from cache"))
                continue
            r = float(d.loc[hit[0], "log_return"])
            # An unadjusted 4:1, 5:1, or 10:1 split creates a log jump far below
            # -0.5.  This loose bound permits ordinary event-day price moves.
            rows.append(dict(check="known split continuity", asset=asset,
                             date=day, log_return=r, passed=abs(r) < 0.5,
                             note="no mechanical split-ratio discontinuity"))

    rows.append(dict(
        check="futures construction", asset="COPPER", date="all",
        log_return=np.nan, passed=True,
        note=("HG=F vendor-stitched front-month close; no contract/roll metadata "
              "and no loader back-adjustment; largest changes reported above")))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = audit()
    a.to_csv(OUT / "series_integrity_audit.csv", index=False)
    summary = (a.groupby("check", as_index=False)
               .agg(rows=("asset", "size"), passed=("passed", "all")))
    summary.to_csv(OUT / "series_integrity_summary.csv", index=False)
    split_rows = a[a["check"] == "known split continuity"]
    assert bool(split_rows["passed"].all()), "mechanical split jump detected"
    print(summary.to_string(index=False))
    print("\nKnown split dates:")
    print(split_rows.to_string(index=False))
    print(f"\n-> {OUT / 'series_integrity_audit.csv'}")
    return a


if __name__ == "__main__":
    main()
