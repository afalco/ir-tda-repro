#!/usr/bin/env python3
"""Step 7 -- predict processing-relevant properties from the spectral topology.

Infrared and thermogravimetric measurements are independent characterisations
of the same specimen, so predicting one from the other is a genuine inference
rather than a re-description of a single measurement. The targets are the
quantities that bound the thermal processing window of a compound; the
predictors are the topological descriptors of its IR spectrum.

Two reference points are reported alongside every model, and both matter for
reading the result:

    mean         predict the global mean of the target. The cross-validated
                 R^2 is measured against this, so it sits at zero by
                 construction.
    family mean  predict the mean of the target over the material family. A
                 model that does not beat this has learned nothing beyond the
                 family assignment, which Sect. 4 already showed is
                 recoverable.

Validation is leave-one-out throughout, with the number of PLS components
chosen by an inner five-fold loop on the training fold, so no information from
the held-out sample reaches the model.

Writes:
    results/regression_scores.csv
    results/regression_predictions.csv
    figures/07_regression.png
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
from sklearn.feature_selection import VarianceThreshold  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def _scaler() -> StandardScaler:
    """Centre the features without rescaling them individually.

    Every representation used here has homogeneous units across its columns --
    absorbance for the raw spectra, image intensity for the two persistence
    images -- so dividing each column by its own standard deviation is not
    meaningful. It is also numerically destructive for the images, whose
    pixels are mostly empty: a pixel with a variance of 1e-30 across the data
    set would be amplified to the scale of the informative ones.
    """
    return StandardScaler(with_mean=True, with_std=False)


def _prune():
    """Drop columns that are constant across the training fold.

    The persistence images are sparse: most pixels are empty for every sample.
    Such columns carry no information and make the PLS deflation
    ill-conditioned, so they are removed inside the pipeline, i.e. refitted on
    each training fold rather than chosen once on the whole data set.
    """
    return VarianceThreshold(threshold=1e-12)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irtda import dataset, plotting  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

TARGETS = ["T5", "T10", "T50", "T_dtg_peak", "residue"]
MAX_COMPONENTS = 6
INNER_FOLDS = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Cross-validated R^2 (Q^2), RMSE and MAPE, referenced to the global mean."""
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "Q2": 1.0 - ss_res / ss_tot,
        "RMSE": float(np.sqrt(ss_res / len(y_true))),
        "MAPE": float(100.0 * np.mean(np.abs((y_true - y_pred) / y_true))),
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def pls_predict(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out predictions of a PLS model with a nested component choice."""
    n = len(y)
    predictions = np.empty(n)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)

        for i in range(n):
            train = np.setdiff1d(np.arange(n), [i])
            X_tr, y_tr = X[train], y[train]

            best_k, best_err = 1, np.inf
            splitter = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=0)
            folds = list(splitter.split(X_tr))
            for k in range(1, min(MAX_COMPONENTS, len(train) - 2) + 1):
                errors = []
                for inner, held in folds:
                    model = make_pipeline(_prune(), _scaler(), PLSRegression(k, scale=False))
                    model.fit(X_tr[inner], y_tr[inner])
                    errors.append(
                        ((model.predict(X_tr[held]).ravel() - y_tr[held]) ** 2).mean()
                    )
                err = float(np.mean(errors))
                if err < best_err:
                    best_k, best_err = k, err

            model = make_pipeline(_prune(), _scaler(), PLSRegression(best_k, scale=False))
            model.fit(X_tr, y_tr)
            predictions[i] = model.predict(X[i: i + 1]).ravel()[0]

    return predictions


def ridge_predict(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out predictions of a ridge model with a nested alpha choice."""
    n = len(y)
    predictions = np.empty(n)
    alphas = np.logspace(-2, 6, 40)
    for i in range(n):
        train = np.setdiff1d(np.arange(n), [i])
        model = make_pipeline(_prune(), _scaler(), RidgeCV(alphas=alphas))
        model.fit(X[train], y[train])
        predictions[i] = model.predict(X[i: i + 1]).ravel()[0]
    return predictions


def family_mean_predict(families: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out prediction from the mean of the sample's own family."""
    n = len(y)
    predictions = np.empty(n)
    for i in range(n):
        train = np.setdiff1d(np.arange(n), [i])
        same = train[families[train] == families[i]]
        predictions[i] = y[same].mean() if same.size else y[train].mean()
    return predictions


def bootstrap_delta_q2(y, y_a, y_b, n_boot=5000, seed=0):
    """Paired bootstrap confidence interval for Q2(a) - Q2(b).

    Samples are resampled with replacement and both models are rescored on the
    same resample, so the comparison stays paired. With 39 samples this is the
    minimum needed to say whether a difference in Q2 means anything.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        ss_tot = ((yy - yy.mean()) ** 2).sum()
        if ss_tot <= 0:
            deltas[b] = np.nan
            continue
        q2_a = 1.0 - ((yy - y_a[idx]) ** 2).sum() / ss_tot
        q2_b = 1.0 - ((yy - y_b[idx]) ** 2).sum() / ss_tot
        deltas[b] = q2_a - q2_b
    deltas = deltas[np.isfinite(deltas)]
    return (float(np.percentile(deltas, 2.5)),
            float(np.percentile(deltas, 97.5)),
            float((deltas > 0).mean()))


def _expand(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Reinsert NaNs where the target was missing."""
    out = np.full(len(keep), np.nan)
    out[keep] = values
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(PROCESSED / "labels.csv")
    targets = pd.read_csv(PROCESSED / "targets.csv")
    data = np.load(PROCESSED / "spectra.npz", allow_pickle=True)

    representations = {
        "PI": np.load(RESULTS / "persistence_images.npy"),
        "TFI": np.load(RESULTS / "fingerprint_images.npy"),
        "raw spectra": dataset.normalise(data["intensity"], "snv"),
    }
    families = labels["family"].to_numpy()

    rows: list[dict] = []
    predictions: dict[str, object] = {
        "sheet": labels["sheet"].astype(str), "family": families
    }

    for target in TARGETS:
        y_all = targets[target].to_numpy(dtype=float)
        keep = np.isfinite(y_all)
        y = y_all[keep]
        predictions[f"{target}_true"] = y_all

        y_hat = family_mean_predict(families[keep], y)
        rows.append({"target": target, "representation": "family mean",
                     "model": "-", "n": int(keep.sum()), **scores(y, y_hat)})
        predictions[f"{target}_family mean"] = _expand(y_hat, keep)

        for name, X in representations.items():
            for model_name, fn in (("PLS", pls_predict), ("ridge", ridge_predict)):
                y_hat = fn(X[keep], y)
                rows.append({"target": target, "representation": name,
                             "model": model_name, "n": int(keep.sum()),
                             **scores(y, y_hat)})
                predictions[f"{target}_{name}_{model_name}"] = _expand(y_hat, keep)

        best = max((r for r in rows
                    if r["target"] == target and r["model"] != "-"),
                   key=lambda r: r["Q2"])
        reference = next(r for r in rows if r["target"] == target
                         and r["representation"] == "family mean")
        print(f"{target:12s} best {best['representation']:12s} {best['model']:5s} "
              f"Q2={best['Q2']:6.3f}  MAPE={best['MAPE']:5.2f}%   "
              f"(family mean Q2={reference['Q2']:6.3f})")

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "regression_scores.csv", index=False)

    # -- paired bootstrap comparisons ---------------------------------------
    # For every target the two contenders of interest are compared against the
    # two reference points, so that a difference in Q2 can be read against its
    # own uncertainty rather than at face value.
    contenders = [("TFI", "ridge"), ("raw spectra", "ridge")]
    references = [("family mean", None), ("raw spectra", "ridge")]

    comparisons = []
    for target in TARGETS:
        y_all = targets[target].to_numpy(dtype=float)
        keep = np.isfinite(y_all)
        y = y_all[keep]

        def prediction_of(representation: str, model: str | None) -> np.ndarray:
            key = (f"{target}_{representation}" if model is None
                   else f"{target}_{representation}_{model}")
            return np.asarray(predictions[key], dtype=float)[keep]

        def q2_of(representation: str, model: str | None) -> float:
            mask = ((table["target"] == target)
                    & (table["representation"] == representation)
                    & (table["model"] == (model if model else "-")))
            return float(table.loc[mask, "Q2"].iloc[0])

        for rep, model in contenders:
            for ref_rep, ref_model in references:
                if (rep, model) == (ref_rep, ref_model):
                    continue
                lo, hi, p_better = bootstrap_delta_q2(
                    y, prediction_of(rep, model), prediction_of(ref_rep, ref_model)
                )
                comparisons.append({
                    "target": target,
                    "model": f"{rep} + {model}",
                    "Q2": q2_of(rep, model),
                    "reference": ref_rep if ref_model is None
                                 else f"{ref_rep} + {ref_model}",
                    "reference_Q2": q2_of(ref_rep, ref_model),
                    "delta_Q2": q2_of(rep, model) - q2_of(ref_rep, ref_model),
                    "delta_Q2_lo": lo,
                    "delta_Q2_hi": hi,
                    "P_better": p_better,
                })

    comparison_table = pd.DataFrame(comparisons)
    comparison_table.to_csv(RESULTS / "regression_comparisons.csv", index=False)
    print()
    print(comparison_table.round(3).to_string(index=False))
    pd.DataFrame(predictions).to_csv(RESULTS / "regression_predictions.csv",
                                     index=False)

    print()
    print(table.pivot_table(index="target", columns=["representation", "model"],
                            values="Q2").round(3).to_string())

    _figure(table, predictions, targets, families)
    print(f"\nwritten to {RESULTS} and {FIGURES / '07_regression.png'}")


def _figure(table, predictions, targets, families) -> None:
    """Observed against predicted, for the best model of each target."""
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(3.4 * len(TARGETS), 3.6))

    for ax, target in zip(np.atleast_1d(axes), TARGETS):
        sub = table[(table["target"] == target) & (table["model"] != "-")]
        best = sub.loc[sub["Q2"].idxmax()]
        key = f"{target}_{best['representation']}_{best['model']}"

        y = targets[target].to_numpy(dtype=float)
        y_hat = np.asarray(predictions[key], dtype=float)
        for family in sorted(set(families)):
            idx = [i for i, f in enumerate(families) if f == family]
            ax.scatter(y[idx], y_hat[idx], s=32, alpha=0.85,
                       color=plotting.FAMILY_COLOURS.get(family, "#7f7f7f"),
                       label=family, edgecolor="white", linewidth=0.6)

        finite = np.isfinite(y) & np.isfinite(y_hat)
        lo = float(min(y[finite].min(), y_hat[finite].min()))
        hi = float(max(y[finite].max(), y_hat[finite].max()))
        ax.plot([lo, hi], [lo, hi], color="#999999", lw=1, zorder=0)
        ax.set_title(f"{target}\n{best['representation']} + {best['model']}, "
                     f"$Q^2$ = {best['Q2']:.2f}", fontsize=10)
        ax.set_xlabel("measured")
        ax.grid(alpha=0.25)

    np.atleast_1d(axes)[0].set_ylabel("predicted (leave-one-out)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("Prediction of thermogravimetric properties from the topology "
                 "of the infrared spectrum")
    fig.tight_layout()
    fig.savefig(FIGURES / "07_regression.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
