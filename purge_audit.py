"""Audit of the purged walk-forward split.

The split in `evaluation.fold_indices` purges in the asset's own observation
index. This module checks that claim exhaustively rather than trusting it: for
every asset, every fold and every horizon used anywhere in the study it counts
the anchors whose label endpoint falls on the wrong side of a partition
boundary. Every count must be zero.

The same counts are produced under the superseded calendar-day rule
(`legacy=True`) so that the size of the correction is measured rather than
asserted.

Outputs -> results/analysis/purge_audit.csv
"""
import numpy as np
import pandas as pd

from config import ASSETS, FOLDS, HORIZONS, L, RESULTS
from data import load_asset
from evaluation import fold_indices

OUT = RESULTS / "analysis"


def _overlaps(idx, h):
    """Anchors whose label endpoint crosses the next partition boundary."""
    tr, va, te = idx
    return dict(
        n_train=len(tr), n_val=len(va), n_test=len(te),
        train_to_val=int(np.sum(tr + h >= va[0])) if len(va) else 0,
        val_to_test=int(np.sum(va + h >= te[0])) if len(va) else 0,
        train_to_test=int(np.sum(tr + h >= te[0])))


def audit(horizons=None):
    horizons = sorted(set(HORIZONS if horizons is None else horizons))
    rows = []
    for asset in ASSETS:
        df = load_asset(asset)
        dts = df["date"].values
        for k, (ts, tz) in enumerate(FOLDS):
            cut = int(np.searchsorted(dts, np.datetime64(tz) + np.timedelta64(2, "D")))
            if cut < 700:
                continue
            dates = dts[:cut]
            for h in horizons:
                anchors = np.arange(L - 1, cut - h)
                ad = dates[anchors]
                for rule in ("observation index", "calendar days"):
                    idx = fold_indices(ad, ts, tz, h=h,
                                       legacy=(rule == "calendar days"))
                    if idx is None:
                        continue
                    # median calendar span of an h-observation label
                    end = np.minimum(idx[0] + h, len(ad) - 1)
                    span = np.median((ad[end] - ad[idx[0]])
                                     .astype("timedelta64[D]").astype(float))
                    rows.append(dict(asset=asset, fold=k, horizon=h, rule=rule,
                                     label_span_days=float(span),
                                     **_overlaps(idx, h)))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    A = audit()
    A.to_csv(OUT / "purge_audit.csv", index=False)

    cols = ["train_to_val", "val_to_test", "train_to_test"]
    tot = A.groupby("rule")[cols].sum()
    print(A[A.rule == "observation index"].to_string(index=False))
    print("\ntotal boundary-crossing anchors:")
    print(tot.to_string())

    bad = A[(A.rule == "observation index")][cols].to_numpy().sum()
    print(f"\nobservation-index rule: {bad} crossing anchors "
          f"over {len(A[A.rule == 'observation index'])} (asset, fold, horizon) cells")
    assert bad == 0, "purge audit failed"
    print(f"-> {OUT / 'purge_audit.csv'}")
    return A


if __name__ == "__main__":
    main()
