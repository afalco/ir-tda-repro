#!/usr/bin/env python3
"""Step 12 -- a safety margin with a finite-sample guarantee.

The moulding rule of step 11 sets the admissible process temperature at the
predicted onset less a margin taken as an empirical quantile of the
leave-one-out residual. With 39 compounds that quantile is estimated from two or
three points in the tail, and it is optimistic: asked for a 5% risk of
over-heating it delivers 7.7%, and asked for 1% it delivers 5.1%. A rule that
does not keep its promise is worse than no rule.

The jackknife+ of Barber, Candes, Ramdas and Tibshirani (2021) replaces the
quantile with a construction that has a coverage guarantee in finite samples,
with no assumption on the distribution of the residuals and none on the
regression method. For a one-sided lower bound on a new compound's onset,

    L(x) = the floor(alpha (n+1))-th smallest of { mu_{-i}(x) + R_i },

with R_i the signed leave-one-out residual, and the guarantee is

    P(T5 > L(x)) >= 1 - 2 alpha.

The factor of two is the price of the construction and is stated rather than
hidden: a rule that must over-heat at most 5% of compounds is obtained by
setting alpha = 0.025.

Evaluating the guarantee honestly needs a second layer of resampling. The bound
for compound i must be built without i, so the residuals entering it are
themselves computed on the remaining 38 compounds, leaving out one more at a
time: 39 x 38 fits, which for ridge costs seconds.

Also reported: a paired bootstrap on the difference in margin between each
representation and the family mean, since the eight degrees claimed in step 11
carried no uncertainty; and a permutation test of the onset prediction itself.

One thing the data cannot do is worth recording here rather than in a footnote.
The bound needs the floor(alpha (m+1))-th order statistic of m = n - 1 = 38
candidates; at alpha = 0.025 that index is zero, so no such order statistic
exists and the bound degenerates to the smallest candidate. A distribution-free
5% guarantee needs alpha >= 1/39, that is a nominal risk of 5.1%: this data set
is one compound short of certifying the number step 11 quotes.

Writes:
    results/conformal_margin.csv
    results/conformal_bootstrap.csv
    results/conformal_permutation.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

_reg = __import__("07_property_regression")
_window = __import__("11_moulding_window")

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

ALPHAS = (0.025, 0.005)          # jackknife+ guarantees 1 - 2 alpha
ALPHA_LABEL = {0.025: "5", 0.005: "1"}
ALPHAS_RIDGE = np.logspace(-2, 6, 40)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--boot", type=int, default=5000)
    p.add_argument("--permutations", type=int, default=1000)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Predictors, written so that the conformal loops can call them cheaply
# ---------------------------------------------------------------------------
def ridge_predictor(X):
    """A predictor closing over the features only; the target is an argument.

    Keeping the target out of the closure is what lets the permutation test of
    part C refit the model on a shuffled target rather than silently reusing the
    original one.
    """
    def predict(y, train, at):
        model = make_pipeline(_reg._prune(), _reg._scaler(),
                              RidgeCV(alphas=ALPHAS_RIDGE))
        model.fit(X[train], y[train])
        return model.predict(X[at]).ravel()
    return predict


def family_predictor(families):
    def predict(y, train, at):
        out = np.empty(len(at))
        for k, index in enumerate(at):
            same = train[families[train] == families[index]]
            out[k] = y[same].mean() if same.size else y[train].mean()
        return out
    return predict


def loo_predictions(predict, y) -> np.ndarray:
    n = len(y)
    return np.array([predict(y, np.setdiff1d(np.arange(n), [i]), np.array([i]))[0]
                     for i in range(n)])


def jackknife_plus_bound(predict, y, held_out: int, alpha: float) -> float:
    """One-sided jackknife+ lower bound for the compound ``held_out``.

    Built entirely from the other ``n - 1`` compounds: for each of them a model
    is fitted on the remaining ``n - 2``, giving both a residual at that
    compound and a prediction at the held-out one.
    """
    n = len(y)
    others = np.setdiff1d(np.arange(n), [held_out])
    candidates = np.empty(len(others))
    for k, j in enumerate(others):
        train = np.setdiff1d(others, [j])
        prediction = predict(y, train, np.array([j, held_out]))
        residual = y[j] - prediction[0]
        candidates[k] = prediction[1] + residual
    m = len(candidates)
    rank = int(np.floor(alpha * (m + 1)))
    if rank < 1:                       # too few compounds for this alpha
        return float(np.min(candidates))
    return float(np.sort(candidates)[rank - 1])


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(PROCESSED / "labels.csv")
    targets = pd.read_csv(PROCESSED / "targets.csv")
    data = np.load(PROCESSED / "spectra.npz", allow_pickle=True)
    wavenumber, raw = data["wavenumber"], data["intensity"]
    families = labels["family"].to_numpy()
    y = targets["T5"].to_numpy(dtype=float)
    n = len(y)

    predictors = {"Family mean": family_predictor(families)}
    for name, X in _window.build_representations(raw, wavenumber).items():
        predictors[name] = ridge_predictor(X)

    # -- A. the two margins, and whether they keep their promise -------------
    rows = []
    bounds: dict[tuple[str, float], np.ndarray] = {}
    for name, predict in predictors.items():
        loo = loo_predictions(predict, y)
        residual = y - loo
        for alpha in ALPHAS:
            nominal = 2 * alpha
            naive = loo + _window.nested_margin(residual, nominal)
            conformal = np.array([jackknife_plus_bound(predict, y, i, alpha)
                                  for i in range(n)])
            bounds[(name, alpha)] = conformal
            rows.append(dict(
                representation=name, nominal_risk=nominal,
                naive_margin=float(np.mean(loo - naive)),
                naive_overheated=float((y < naive).mean()),
                conformal_margin=float(np.mean(loo - conformal)),
                conformal_overheated=float((y < conformal).mean())))
        print(f"  {name:30s} done")
    margin = pd.DataFrame(rows)
    margin.to_csv(RESULTS / "conformal_margin.csv", index=False)

    print("\nA. margin and realised risk, empirical quantile against jackknife+")
    show = margin.copy()
    show["nominal"] = (100 * show.nominal_risk).round(0).astype(int).astype(str) + "%"
    print(show[["representation", "nominal", "naive_margin", "naive_overheated",
                "conformal_margin", "conformal_overheated"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))

    # -- B. is the gain over the family mean real? --------------------------
    rng = np.random.default_rng(args.seed)
    reference = bounds[("Family mean", 0.025)]
    reference_loo = loo_predictions(predictors["Family mean"], y)
    rows = []
    for name, predict in predictors.items():
        if name == "Family mean":
            continue
        loo = loo_predictions(predict, y)
        gain = (reference_loo - reference) - (loo - bounds[(name, 0.025)])
        deltas = np.array([gain[rng.integers(0, n, n)].mean()
                           for _ in range(args.boot)])
        rows.append(dict(representation=name, gain=float(gain.mean()),
                         lo=float(np.percentile(deltas, 2.5)),
                         hi=float(np.percentile(deltas, 97.5)),
                         p_above_zero=float((deltas > 0).mean())))
    bootstrap = pd.DataFrame(rows)
    bootstrap.to_csv(RESULTS / "conformal_bootstrap.csv", index=False)
    print("\nB. margin returned against the family mean, at 5% risk "
          "(paired bootstrap)")
    print(bootstrap.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\nC. permutation test of the onset prediction, "
          f"{args.permutations} permutations")
    # -- C. is the prediction better than chance? ---------------------------
    rows = []
    for name, predict in predictors.items():
        if name == "Family mean":
            continue
        observed = _reg.scores(y, loo_predictions(predict, y))["Q2"]
        null = np.empty(args.permutations)
        for b in range(args.permutations):
            shuffled = rng.permutation(y)
            null[b] = _reg.scores(shuffled, loo_predictions(predict, shuffled))["Q2"]
        # The p-value counts the permutation itself, which keeps it valid at
        # any number of permutations rather than only asymptotically.
        p_value = float((1 + (null >= observed).sum()) / (1 + args.permutations))
        rows.append(dict(representation=name, Q2=observed,
                         null_mean=float(null.mean()),
                         null_q95=float(np.percentile(null, 95)),
                         p_value=p_value))
        print(f"  {name:30s} Q2={observed:.3f}  null 95th pct={np.percentile(null, 95):.3f}"
              f"  p={p_value:.4f}")
    permutation = pd.DataFrame(rows)
    permutation.to_csv(RESULTS / "conformal_permutation.csv", index=False)

    print(f"\nwritten to {RESULTS}")


if __name__ == "__main__":
    main()
