#!/usr/bin/env python3
"""Step 11 -- from a cross-validated Q^2 to a moulding decision.

A coefficient of determination is not a processing statement. What a converter
needs from the onset of degradation is the highest temperature at which a
compound can be formed without beginning to decompose, and what a predictor of
that onset is worth is measured in degrees of usable processing window, not in
Q^2. This script makes that translation.

The decision rule is the obvious one. A compound is formed at a process
temperature T_p, and it must not have begun to degrade there, so the admissible
temperature is the predicted onset less a margin that absorbs the prediction
error:

    T_admissible = T5_predicted + q_alpha(residual),

where q_alpha is the alpha-quantile of the leave-one-out residual
T5_true - T5_predicted, estimated on the other compounds. By construction a
compound is over-heated with probability alpha. The margin is what the
prediction costs, in degrees, and it is directly comparable across
representations: a better predictor buys back processing window.

Three things are reported.

    A  the margin each representation requires at 5% and 1% risk;
    B  how often the resulting decision is right, over a grid of process
       temperatures, and how often it is wrong in the direction that matters --
       a compound accepted for a temperature at which it degrades;
    C  which families the spectral prediction helps, by comparing the spread of
       the onset within a family with the error of the family mean.

Writes:
    results/moulding_margin.csv
    results/moulding_decision.csv
    results/moulding_by_family.csv
    figures/11_moulding_window.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from irtda import dataset, features, peaks, preprocess, images  # noqa: E402

_reg = __import__("07_property_regression")

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

RISKS = (0.05, 0.01)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_representations(raw, wavenumber):
    """The descriptors compared, each as a feature matrix over the 39 compounds."""
    config = features.DescriptorConfig(adaptive_threshold=False)
    normalised = dataset.normalise(raw, "minmax")
    descriptors = features.compute_diagrams(normalised, wavenumber, config)
    _, fingerprint_imager = features.build_imagers(descriptors, config)

    tables = [peaks.peak_table(row, wavenumber,
                               peaks.PeakConfig(prominence=config.min_persistence))
              for row in normalised]
    diagrams = [peaks.attribute_diagram(t, "prominence") for t in tables]
    probe = images.PersistenceImager(resolution=config.resolution)
    probe.fit(diagrams)
    pixel = probe.lifetime_range[1] / config.resolution
    peak_imager = images.PersistenceImager(
        resolution=config.resolution,
        sigma=(config.sigma_wavenumber, max(pixel, 1e-6)))
    peak_imager.fit(diagrams)

    return {
        "Fingerprint image": fingerprint_imager.transform(descriptors.fingerprints),
        "Peaks, position + prominence": peak_imager.transform(diagrams),
        "Raw spectra (SNV)": preprocess.CHAINS["SNV"](raw, wavenumber),
        "SG 2nd derivative": preprocess.CHAINS["SG 2nd derivative"](raw, wavenumber),
    }


def nested_margin(residuals: np.ndarray, risk: float) -> np.ndarray:
    """Margin for each compound, estimated from the residuals of the others.

    Taking the quantile over all residuals including the compound's own would
    let the held-out sample set its own safety margin. The margin is therefore
    itself computed leave-one-out.
    """
    n = len(residuals)
    out = np.empty(n)
    for i in range(n):
        others = np.delete(residuals, i)
        out[i] = float(np.quantile(others, risk))
    return out


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(PROCESSED / "labels.csv")
    targets = pd.read_csv(PROCESSED / "targets.csv")
    data = np.load(PROCESSED / "spectra.npz", allow_pickle=True)
    wavenumber, raw = data["wavenumber"], data["intensity"]
    families = labels["family"].to_numpy()

    y = targets["T5"].to_numpy(dtype=float)
    assert np.isfinite(y).all(), "every compound has a usable onset"

    predictions = {"Family mean": _reg.family_mean_predict(families, y)}
    for name, X in build_representations(raw, wavenumber).items():
        predictions[name] = _reg.ridge_predict(X, y)

    print(f"onset of degradation: {y.min():.0f}-{y.max():.0f} C, "
          f"median {np.median(y):.0f} C, standard deviation {y.std(ddof=1):.1f} C\n")

    # -- A. the margin each representation costs ----------------------------
    rows = []
    for name, predicted in predictions.items():
        residual = y - predicted
        row = dict(representation=name,
                   Q2=_reg.scores(y, predicted)["Q2"],
                   MAE=float(np.abs(residual).mean()))
        for risk in RISKS:
            margin = nested_margin(residual, risk)
            admissible = predicted + margin
            row[f"margin_{int(risk * 100)}"] = float(-np.mean(margin))
            row[f"overheated_{int(risk * 100)}"] = float((y < admissible).mean())
        rows.append(row)
    margins = pd.DataFrame(rows)
    margins.to_csv(RESULTS / "moulding_margin.csv", index=False)
    print("A. margin the prediction costs, in degrees of processing window")
    print(margins.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # -- B. the decision over a grid of process temperatures ----------------
    grid = np.arange(210.0, 320.0, 5.0)
    rows = []
    for name, predicted in predictions.items():
        residual = y - predicted
        margin = nested_margin(residual, 0.05)
        admissible = predicted + margin
        for temperature in grid:
            accepted = admissible >= temperature       # the rule says: safe to mould
            safe = y >= temperature                    # and it actually is
            rows.append(dict(
                representation=name, process_temperature=float(temperature),
                accepted=int(accepted.sum()),
                truly_safe=int(safe.sum()),
                unsafe_accepted=int((accepted & ~safe).sum()),
                safe_rejected=int((~accepted & safe).sum()),
                correct=float((accepted == safe).mean())))
    decision = pd.DataFrame(rows)
    decision.to_csv(RESULTS / "moulding_decision.csv", index=False)
    print("\nB. decision over process temperatures of 210-315 C, in 5 C steps")
    summary = decision.groupby("representation").agg(
        mean_correct=("correct", "mean"),
        unsafe_accepted=("unsafe_accepted", "sum"),
        safe_rejected=("safe_rejected", "sum"))
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))

    # -- C. where the spectrum adds to knowing the family -------------------
    frame = pd.DataFrame({"family": families, "T5": y,
                          "fingerprint": predictions["Fingerprint image"],
                          "family_mean": predictions["Family mean"]})
    rows = []
    for name, group in frame.groupby("family"):
        if len(group) < 2:
            continue
        rows.append(dict(
            family=name, n=len(group),
            spread=float(group.T5.max() - group.T5.min()),
            sd=float(group.T5.std(ddof=1)),
            MAE_family_mean=float(np.abs(group.T5 - group.family_mean).mean()),
            MAE_fingerprint=float(np.abs(group.T5 - group.fingerprint).mean())))
    by_family = pd.DataFrame(rows).sort_values("spread", ascending=False)
    by_family["gain"] = by_family.MAE_family_mean - by_family.MAE_fingerprint
    by_family.to_csv(RESULTS / "moulding_by_family.csv", index=False)
    print("\nC. within-family spread of the onset, and what the spectrum adds")
    print(by_family.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    # -- figure --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    ax = axes[0]
    predicted = predictions["Fingerprint image"]
    margin = nested_margin(y - predicted, 0.05)
    order = np.argsort(predicted)
    ax.plot([y.min() - 5, y.max() + 5], [y.min() - 5, y.max() + 5],
            color="#bbbbbb", lw=1)
    ax.fill_between(predicted[order], (predicted + margin)[order], predicted[order],
                    color="#0046a0", alpha=0.15,
                    label="margin at 5% risk")
    for family in np.unique(families):
        mask = families == family
        ax.scatter(predicted[mask], y[mask], s=26, label=family)
    ax.set_xlabel(r"predicted onset $\hat{T}_5$ [$^\circ$C]")
    ax.set_ylabel(r"measured onset $T_5$ [$^\circ$C]")
    ax.set_title("(a) the prediction and its safety margin", fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=2)

    ax = axes[1]
    # Both constructions, since the empirical quantile does not deliver its
    # nominal risk and the comparison between them is the point.
    conformal_path = RESULTS / "conformal_margin.csv"
    names = list(margins.representation)
    y_pos = np.arange(len(names))
    height = 0.38
    ax.barh(y_pos - height / 2, margins.margin_5, height=height,
            color="#c9a227", label="empirical quantile")
    if conformal_path.exists():
        conformal = (pd.read_csv(conformal_path)
                     .query("nominal_risk == 0.05")
                     .set_index("representation")
                     .reindex(names))
        ax.barh(y_pos + height / 2, conformal.conformal_margin, height=height,
                color=["#0046a0" if n == "Fingerprint image" else "#b05800"
                       for n in names],
                label=r"jackknife$+$, guaranteed")
        for i, value in enumerate(conformal.conformal_margin):
            ax.text(value + 0.8, i + height / 2, f"{value:.0f}", va="center",
                    fontsize=7.5)
    for i, value in enumerate(margins.margin_5):
        ax.text(value + 0.8, i - height / 2, f"{value:.0f}", va="center",
                fontsize=7.5, color="#7a6410")
    ax.set_yticks(y_pos, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r"margin at $5\,\%$ nominal risk [$^\circ$C]")
    ax.set_title("(b) processing window surrendered to uncertainty", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=2)

    ax = axes[2]
    width = 0.38
    y_pos = np.arange(len(by_family))
    ax.barh(y_pos - width / 2, by_family.MAE_family_mean, height=width,
            color="#888888", label="family mean")
    ax.barh(y_pos + width / 2, by_family.MAE_fingerprint, height=width,
            color="#0046a0", label="fingerprint image")
    ax.set_yticks(y_pos, [f"{f} (n={n}, spread {s:.0f}$^\\circ$C)"
                          for f, n, s in zip(by_family.family, by_family.n,
                                             by_family.spread)], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r"mean absolute error [$^\circ$C]")
    ax.set_title("(c) where the spectrum adds to knowing the family", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(FIGURES / "11_moulding_window.png", dpi=180)
    plt.close(fig)
    print(f"\nwritten to {RESULTS} and {FIGURES / '11_moulding_window.png'}")


if __name__ == "__main__":
    main()
