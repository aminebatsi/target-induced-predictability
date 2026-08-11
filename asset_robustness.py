"""Does the headline gap depend on any one asset or any one fold?

Litecoin trades every calendar day and therefore contributes 1801 of the 6760
test anchors, about 27%, more than any other series. Pooled figures could in
principle be carried by it. The same question applies to the five folds, which
cover a single five-year window.

This module recomputes the paper's central quantities per asset, per fold, and
under leave-one-asset-out, using the cached forecasts. Nothing is refitted.

Outputs -> results/analysis/robust_by_asset.csv
           results/analysis/robust_by_fold.csv
           results/analysis/robust_loao.csv
"""
import numpy as np
import pandas as pd

from portfolio_replay import OUT
from predictive_ability import MODELS, load_predictions


def _acc(d):
    s = np.sign(d.y_pred.values)
    s[s == 0] = 1.0
    return (float(np.mean(s == np.sign(d.y_true.values))),
            float(np.mean(s == np.sign(d.y_true_price_h.values))),
            float(np.mean(s == np.sign(d.y_true_price_1.values))))


def _summary(P, label, value):
    """Mean over the thirteen models of each of the three accuracies."""
    rows = []
    for m in sorted(P.model.unique()):
        t, ph, p1 = _acc(P[P.model == m])
        rows.append(dict(model=m, target=t, price_h=ph, price_1=p1))
    D = pd.DataFrame(rows)
    return dict(**{label: value}, n_obs=len(P) // P.model.nunique(),
                mean_target=D.target.mean(), mean_price_h=D.price_h.mean(),
                mean_price_1=D.price_1.mean(),
                best_target=D.target.max(),
                gap=D.target.mean() - D.price_1.mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = load_predictions()

    A = pd.DataFrame([_summary(P[P.asset == a], "asset", a)
                      for a in sorted(P.asset.unique())])
    F = pd.DataFrame([_summary(P[P.fold == k], "fold", int(k))
                      for k in sorted(P.fold.unique())])
    Lo = pd.DataFrame([_summary(P[P.asset != a], "excluded", a)
                       for a in sorted(P.asset.unique())])
    Lo = pd.concat([pd.DataFrame([_summary(P, "excluded", "none (all five)")]),
                    Lo], ignore_index=True)

    for df, fp in ((A, "robust_by_asset.csv"), (F, "robust_by_fold.csv"),
                   (Lo, "robust_loao.csv")):
        df.to_csv(OUT / fp, index=False)

    pd.set_option("display.width", 200)
    print("== by asset (mean over the thirteen models) ==")
    print(A.round(4).to_string(index=False))
    print("\n== by fold ==")
    print(F.round(4).to_string(index=False))
    print("\n== leave one asset out ==")
    print(Lo.round(4).to_string(index=False))

    print(f"\ngap range across assets : {A.gap.min():.3f} to {A.gap.max():.3f}")
    print(f"gap range across folds  : {F.gap.min():.3f} to {F.gap.max():.3f}")
    print(f"gap range under LOAO    : {Lo.gap.min():.3f} to {Lo.gap.max():.3f}")
    print(f"target accuracy exceeds price-1 in every asset: "
          f"{bool((A.mean_target > A.mean_price_1).all())}")
    print(f"target accuracy exceeds price-1 in every fold : "
          f"{bool((F.mean_target > F.mean_price_1).all())}")
    print(f"price-1 stays below 0.50 in every asset: "
          f"{bool((A.mean_price_1 < 0.5).all())}  "
          f"(range {A.mean_price_1.min():.4f}-{A.mean_price_1.max():.4f})")
    print(f"price-1 stays below 0.50 in every fold : "
          f"{bool((F.mean_price_1 < 0.5).all())}  "
          f"(range {F.mean_price_1.min():.4f}-{F.mean_price_1.max():.4f})")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
