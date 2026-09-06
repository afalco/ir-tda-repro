#!/usr/bin/env python3
"""Step 14 -- how much of the reported Q2 is the partition?

The property regression is validated leave-one-out with the number of PLS
components, or the ridge penalty, chosen by an inner five-fold loop on each
training fold. That is a nested design and no information from the held-out
compound reaches the model. It still reports a single number, and with 39
compounds a single number hides two things a referee is entitled to see: how
much it moves with the arbitrary seed of the inner split, and how much it moves
when the outer loop is not leave-one-out.

Three questions.

    A  Does the inner split matter? The reported figures use one seed. Twenty
       are run here.
    B  What does the estimate look like under a repeated outer k-fold, which
       has less variance than leave-one-out and is the more usual choice at
       this sample size?
    C  Does the ordering between representations survive? A mean difference
       matters less than how often the ordering holds across repetitions, which
       is what a reader would rely on.

Writes:
    results/repeated_cv_inner.csv
    results/repeated_cv_outer.csv
    results/repeated_cv_ordering.csv
    figures/14_repeated_cv.png
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cross_decomposition import PLSRegression  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from irtda import dataset, features  # noqa: E402

_reg = __import__("07_property_regression")
_window = __import__("11_moulding_window")

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

ALPHAS = np.logspace(-2, 6, 40)
MAX_COMPONENTS = 6
INNER_FOLDS = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parts", default="ABC")
    p.add_argument("--target", default="T5")
    p.add_argument("--inner-seeds", type=int, default=20)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--models", default="ridge,PLS")
    return p.parse_args()


# ---------------------------------------------------------------------------
def _ridge(X_tr, y_tr, X_te, seed):
    model = make_pipeline(_reg._prune(), _reg._scaler(), RidgeCV(alphas=ALPHAS))
    model.fit(X_tr, y_tr)
    return model.predict(X_te).ravel()


def _pls(X_tr, y_tr, X_te, seed):
    """PLS with the component count chosen by an inner k-fold on the training set."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        splitter = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
        folds = list(splitter.split(X_tr))
        best_k, best_err = 1, np.inf
        for k in range(1, min(MAX_COMPONENTS, len(X_tr) - 2) + 1):
            errors = []
            for inner, held in folds:
                model = make_pipeline(_reg._prune(), _reg._scaler(),
                                      PLSRegression(k, scale=False))
                model.fit(X_tr[inner], y_tr[inner])
                errors.append(((model.predict(X_tr[held]).ravel()
                                - y_tr[held]) ** 2).mean())
            err = float(np.mean(errors))
            if err < best_err:
                best_k, best_err = k, err
        model = make_pipeline(_reg._prune(), _reg._scaler(),
                              PLSRegression(best_k, scale=False))
        model.fit(X_tr, y_tr)
        return model.predict(X_te).ravel()


MODELS = {"ridge": _ridge, "PLS": _pls}


def family_mean(y, train, test, families):
    out = np.empty(len(test))
    for k, index in enumerate(test):
        same = train[families[train] == families[index]]
        out[k] = y[same].mean() if same.size else y[train].mean()
    return out


def cross_validate(X, y, families, splits, model, seed) -> float:
    """Q2 over a given list of (train, test) splits, pooled over the folds."""
    predicted = np.empty(len(y))
    for train, test in splits:
        if X is None:
            predicted[test] = family_mean(y, train, test, families)
        else:
            predicted[test] = MODELS[model](X[train], y[train], X[test], seed)
    return _reg.scores(y, predicted)["Q2"]


def loo_splits(n):
    return [(np.setdiff1d(np.arange(n), [i]), np.array([i])) for i in range(n)]


def kfold_splits(n, k, seed):
    return list(KFold(n_splits=k, shuffle=True, random_state=seed).split(np.arange(n)))


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(PROCESSED / "labels.csv")
    targets = pd.read_csv(PROCESSED / "targets.csv")
    data = np.load(PROCESSED / "spectra.npz", allow_pickle=True)
    wavenumber, raw = data["wavenumber"], data["intensity"]
    families = labels["family"].to_numpy()
    y = targets[args.target].to_numpy(dtype=float)
    n = len(y)

    representations = _window.build_representations(raw, wavenumber)
    config = features.DescriptorConfig(adaptive_threshold=False)
    descriptors = features.compute_diagrams(
        dataset.normalise(raw, "minmax"), wavenumber, config)
    representations["Persistence image"] = features.build_imagers(
        descriptors, config)[0].transform(descriptors.lifetimes)
    representations["Family mean"] = None
    models = args.models.split(",")

    # -- A. does the inner split matter? ------------------------------------
    if "A" in args.parts:
        print(f"\nA. leave-one-out over {args.inner_seeds} seeds of the inner split")
        splits = loo_splits(n)
        rows = []
        for name, X in representations.items():
            for model in models:
                if X is None and model != models[0]:
                    continue
                scores = [cross_validate(X, y, families, splits, model, seed)
                          for seed in range(args.inner_seeds)]
                rows.append(dict(representation=name,
                                 model="-" if X is None else model,
                                 mean=float(np.mean(scores)),
                                 sd=float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                                 lo=float(np.min(scores)), hi=float(np.max(scores))))
                print(f"  {name:30s} {rows[-1]['model']:6s} "
                      f"Q2 = {rows[-1]['mean']:.3f} +- {rows[-1]['sd']:.3f}  "
                      f"[{rows[-1]['lo']:.3f}, {rows[-1]['hi']:.3f}]")
        pd.DataFrame(rows).to_csv(RESULTS / "repeated_cv_inner.csv", index=False)
        print("\n  Ridge has no inner randomness by construction; the spread shown"
              "\n  for PLS is the cost of choosing its components on one split.")

    # -- B. repeated outer k-fold -------------------------------------------
    if "B" in args.parts:
        print(f"\nB. repeated outer k-fold, {args.repeats} repetitions")
        rows = []
        per_repeat: dict[tuple[str, str, int], list[float]] = {}
        for k in (5, 10):
            for name, X in representations.items():
                for model in models:
                    if X is None and model != models[0]:
                        continue
                    scores = [cross_validate(X, y, families,
                                             kfold_splits(n, k, seed), model, 0)
                              for seed in range(args.repeats)]
                    per_repeat[(name, model, k)] = scores
                    loo = cross_validate(X, y, families, loo_splits(n), model, 0)
                    rows.append(dict(representation=name,
                                     model="-" if X is None else model, k=k,
                                     loo=loo, mean=float(np.mean(scores)),
                                     sd=float(np.std(scores, ddof=1)),
                                     lo=float(np.percentile(scores, 2.5)),
                                     hi=float(np.percentile(scores, 97.5))))
            print(f"  k = {k}")
            for row in [r for r in rows if r["k"] == k]:
                print(f"    {row['representation']:30s} {row['model']:6s} "
                      f"LOO {row['loo']:6.3f}   {k}-fold {row['mean']:6.3f} "
                      f"+- {row['sd']:.3f}  [{row['lo']:6.3f}, {row['hi']:6.3f}]")
        pd.DataFrame(rows).to_csv(RESULTS / "repeated_cv_outer.csv", index=False)

        # -- C. does the ordering survive? ----------------------------------
        if "C" in args.parts:
            print("\nC. how often the fingerprint image leads, per repetition")
            reference_model = "ridge" if "ridge" in models else models[0]
            ours = per_repeat.get(("Fingerprint image", reference_model, 5))
            rows = []
            for (name, model, k), scores in per_repeat.items():
                if k != 5 or model != reference_model or name == "Fingerprint image":
                    continue
                wins = float(np.mean([a > b for a, b in zip(ours, scores)]))
                delta = float(np.mean(np.array(ours) - np.array(scores)))
                rows.append(dict(against=name, model=model, mean_delta=delta,
                                 fraction_ahead=wins))
                print(f"    vs {name:30s} dQ2 = {delta:+.3f}   ahead in "
                      f"{100 * wins:5.1f}% of repetitions")
            pd.DataFrame(rows).to_csv(RESULTS / "repeated_cv_ordering.csv",
                                      index=False)

    # -- figure --------------------------------------------------------------
    outer_path = RESULTS / "repeated_cv_outer.csv"
    order_path = RESULTS / "repeated_cv_ordering.csv"
    if "B" in args.parts and outer_path.exists():
        outer = pd.read_csv(outer_path)
        order = [r for r in ["Persistence image", "Raw spectra (SNV)",
                             "SG 2nd derivative", "Family mean",
                             "Peaks, position + prominence", "Fingerprint image"]
                 if r in set(outer.representation)]
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4),
                                 gridspec_kw={"width_ratios": [1.4, 1.0]})

        ax = axes[0]
        colour = {"Fingerprint image": "#0046a0", "Family mean": "#888888"}
        for i, name in enumerate(order):
            for k, offset, alpha in ((5, -0.16, 0.45), (10, 0.16, 1.0)):
                row = outer[(outer.representation == name) & (outer.k == k)]
                if row.empty:
                    continue
                row = row.iloc[0]
                c = colour.get(name, "#b05800")
                ax.plot([row.lo, row.hi], [i + offset] * 2, color=c, lw=2.5,
                        alpha=alpha, solid_capstyle="butt")
                ax.plot([row["mean"]], [i + offset], "o", ms=5, color=c, alpha=alpha)
            row = outer[(outer.representation == name) & (outer.k == 5)].iloc[0]
            ax.plot([row.loo], [i], "|", ms=16, mew=2, color="#111111",
                    label="leave-one-out" if i == 0 else None)
        ax.axvline(0, color="#888888", lw=0.8)
        ax.set_yticks(range(len(order)), order, fontsize=8.5)
        ax.set_xlabel(r"$Q^2$, onset $T_5$")
        ax.set_title("(a) repeated 5- and 10-fold: mean and central 95% "
                     "of repetitions", fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc="lower left")

        ax = axes[1]
        if order_path.exists():
            ordering = pd.read_csv(order_path).sort_values("fraction_ahead")
            y_pos = np.arange(len(ordering))
            ax.barh(y_pos, 100 * ordering.fraction_ahead, color="#0046a0")
            for i, (value, delta) in enumerate(zip(ordering.fraction_ahead,
                                                   ordering.mean_delta)):
                ax.text(100 * value + 1.5, i, f"{100 * value:.0f}%  "
                        rf"($\Delta Q^2$ {delta:+.2f})", va="center", fontsize=8)
            ax.axvline(50, color="#888888", lw=0.8, ls=":")
            ax.set_xlim(0, 135)
            ax.set_yticks(y_pos, ordering["against"], fontsize=8.5)
            ax.set_xlabel("repetitions in which the fingerprint image leads [%]")
            ax.set_title("(b) the ordering, repetition by repetition", fontsize=10)

        fig.tight_layout()
        fig.savefig(FIGURES / "14_repeated_cv.png", dpi=180)
        plt.close(fig)
        print(f"figure written to {FIGURES / '14_repeated_cv.png'}")

    print(f"\nwritten to {RESULTS}")


if __name__ == "__main__":
    main()
