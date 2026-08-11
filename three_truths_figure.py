"""The paper's central figure: one forecast vector, three scoring objects.

Each model contributes a single out-of-sample prediction vector, which is then
scored against the transformed target it was trained on, the raw h-day price
change over the same horizon, and the next one-day price move. Nothing is
refitted here; the predictions are read from the cached forecast artifacts.

Outputs -> results/analysis/three_truths.csv
           results/figures/fig__three_truths.png
"""
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PLOT_RC, RESULTS
from predictive_ability import MODELS, PRED_DIR

OUT = RESULTS / "analysis"
FIGS = RESULTS / "figures"
plt.rcParams.update(PLOT_RC)


def table():
    """Pooled directional accuracy per model against each of the three objects,
    plus the parameter-free slope rule scored on the same anchors."""
    rows = []
    for m in MODELS:
        fp = PRED_DIR / f"forecast__{m}__predictions.csv"
        if not fp.exists():
            continue
        d = pd.read_csv(fp)
        s = np.sign(d.y_pred.values)
        rows.append(dict(
            model=m, n=len(d),
            target=float(np.mean(s == np.sign(d.y_true.values))),
            price_h=float(np.mean(s == np.sign(d.y_true_price_h.values))),
            price_1=float(np.mean(s == np.sign(d.y_true_price_1.values)))))
    T = pd.DataFrame(rows).sort_values("target", ascending=False)
    T["gap"] = T.target - T.price_1
    return T


def figure(T):
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    x = np.arange(len(T))
    w = 0.27
    ax.bar(x - w, T.target, w, label=r"transformed target",
           color="royalblue", alpha=0.9)
    ax.bar(x, T.price_h, w, label=r"raw $h$-day price change",
           color="0.55", alpha=0.9)
    ax.bar(x + w, T.price_1, w, label="next-day price move",
           color="0.78", alpha=0.95)
    ax.axhline(0.5, color="k", ls="--", lw=1.1)
    ax.text(len(T) - 0.4, 0.508, "0.50", fontsize=8, color="0.3", ha="right")
    ax.set_xticks(x, T.model, rotation=32, ha="right")
    ax.set_ylim(0.40, 0.85)
    ax.set_ylabel("directional accuracy")
    ax.set_xlabel("model, ordered by accuracy on the object it was trained on")
    ax.legend(loc="upper right", ncol=1)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "fig__three_truths.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T = table()
    T.to_csv(OUT / "three_truths.csv", index=False)
    figure(T)
    print(T.round(4).to_string(index=False))
    print(f"\nmean over {len(T)} models: target {T.target.mean():.4f}  "
          f"price-h {T.price_h.mean():.4f}  price-1 {T.price_1.mean():.4f}")
    print(f"best-worst spread: target {T.target.max()-T.target.min():.4f}  "
          f"price-1 {T.price_1.max()-T.price_1.min():.4f}")
    print(f"-> {FIGS / 'fig__three_truths.png'}")


if __name__ == "__main__":
    main()
