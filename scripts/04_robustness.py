#!/usr/bin/env python3
"""Step 4 -- robustness of the descriptors to realistic measurement artefacts.

On clean, perfectly aligned spectra measured on one instrument, comparing the
raw absorbance vectors is hard to beat. The argument for topological
descriptors made in the reference papers is a different one: they are designed
to stay stable when the signals no longer match point by point. This script
tests that claim directly on the present data set.

Four artefacts routinely seen in ATR practice are simulated at increasing
amplitude:

    baseline    a smooth quadratic drift, from scattering or crystal fouling
    shift       a rigid translation of the wavenumber axis, from calibration
    noise       additive white noise, from short acquisition times
    intensity   a smooth multiplicative envelope, from variable contact pressure

Each perturbed spectrum is then matched back against the *unperturbed* library
by nearest neighbour. The reported score is the fraction of spectra that
retrieve themselves, which needs no class labels and directly measures whether
a representation still identifies a material after the artefact.

Writes:
    results/robustness.csv
    figures/04_robustness.png
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

from irtda import dataset, features  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

LEVELS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

ARTEFACTS = {
    "baseline drift": dict(unit="fraction of the absorbance range", scale=0.40),
    "wavenumber shift": dict(unit="cm$^{-1}$", scale=16.0),
    "additive noise": dict(unit="fraction of the absorbance range", scale=0.05),
    "intensity envelope": dict(unit="relative amplitude", scale=0.50),
}


def perturb(intensity, wavenumber, artefact, amplitude, rng):
    """Apply one artefact of the given amplitude to every spectrum."""
    x = np.array(intensity, dtype=float, copy=True)
    span = x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True)
    t = (wavenumber - wavenumber[0]) / (wavenumber[-1] - wavenumber[0])

    if artefact == "baseline drift":
        # Random quadratic, drawn once per spectrum.
        coefficients = rng.normal(size=(len(x), 3))
        basis = np.vstack([np.ones_like(t), t, t**2])
        drift = coefficients @ basis
        drift /= np.abs(drift).max(axis=1, keepdims=True)
        return x + amplitude * span * drift

    if artefact == "wavenumber shift":
        shifts = rng.uniform(-amplitude, amplitude, size=len(x))
        return np.vstack([
            np.interp(wavenumber, wavenumber + s, row) for row, s in zip(x, shifts)
        ])

    if artefact == "additive noise":
        return x + amplitude * span * rng.normal(size=x.shape)

    if artefact == "intensity envelope":
        phase = rng.uniform(0, 2 * np.pi, size=(len(x), 1))
        envelope = 1.0 + amplitude * np.sin(2 * np.pi * t[None, :] + phase)
        return x * envelope

    raise ValueError(f"unknown artefact: {artefact!r}")


def self_retrieval(reference: np.ndarray, query: np.ndarray) -> float:
    """Fraction of perturbed spectra whose nearest clean neighbour is themselves."""
    d = np.linalg.norm(query[:, None, :] - reference[None, :, :], axis=2)
    return float((d.argmin(axis=1) == np.arange(len(query))).mean())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=5,
                   help="independent random draws averaged per level")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    data = np.load(PROCESSED / "spectra.npz", allow_pickle=True)
    wavenumber = data["wavenumber"]
    raw = data["intensity"]

    adaptive = features.DescriptorConfig(adaptive_threshold=True)
    fixed = features.DescriptorConfig(adaptive_threshold=False)

    # Reference representations, computed once on the clean spectra. The image
    # grids fitted here are reused for every perturbed copy.
    clean = dataset.normalise(raw, "minmax")
    clean_adaptive, clean_fixed = features.compute_diagrams_multi(
        clean, wavenumber, [adaptive, fixed]
    )
    imager, fingerprint_imager = features.build_imagers(clean_adaptive, adaptive)

    def represent(intensity_raw, descriptors_adaptive, descriptors_fixed):
        return {
            "baseline: raw spectra": dataset.normalise(intensity_raw, "snv"),
            "PI (fixed threshold)": imager.transform(descriptors_fixed.lifetimes),
            "TFI (fixed threshold)": fingerprint_imager.transform(
                descriptors_fixed.fingerprints
            ),
            "TFI (adaptive threshold)": fingerprint_imager.transform(
                descriptors_adaptive.fingerprints
            ),
        }

    reference = represent(raw, clean_adaptive, clean_fixed)

    rows = []
    for artefact, spec in ARTEFACTS.items():
        for level in LEVELS:
            amplitude = level * spec["scale"]
            scores = {name: [] for name in reference}

            for repeat in range(args.repeats):
                rng = np.random.default_rng(args.seed + 1000 * repeat)
                perturbed_raw = perturb(raw, wavenumber, artefact, amplitude, rng)
                perturbed = dataset.normalise(perturbed_raw, "minmax")
                query = represent(
                    perturbed_raw,
                    *features.compute_diagrams_multi(
                        perturbed, wavenumber, [adaptive, fixed]
                    ),
                )
                for name in reference:
                    scores[name].append(self_retrieval(reference[name], query[name]))

            for name, values in scores.items():
                rows.append({
                    "artefact": artefact,
                    "level": float(level),
                    "amplitude": float(amplitude),
                    "unit": spec["unit"],
                    "representation": name,
                    "self_retrieval": float(np.mean(values)),
                    "self_retrieval_sd": float(np.std(values)),
                })
            print(f"{artefact:20s} level {level:.2f}  " + "  ".join(
                f"{n}={np.mean(v):.2f}" for n, v in scores.items()))

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "robustness.csv", index=False)

    # -- figure --------------------------------------------------------------
    colours = {
        "baseline: raw spectra": "#7f7f7f",
        "PI (fixed threshold)": "#d62728",
        "TFI (fixed threshold)": "#9467bd",
        "TFI (adaptive threshold)": "#1f77b4",
    }
    fig, axes = plt.subplots(1, len(ARTEFACTS), figsize=(4.0 * len(ARTEFACTS), 3.4),
                             sharey=True)
    for ax, (artefact, spec) in zip(axes, ARTEFACTS.items()):
        sub = table[table["artefact"] == artefact]
        for name, colour in colours.items():
            s = sub[sub["representation"] == name]
            ax.errorbar(s["amplitude"], s["self_retrieval"], yerr=s["self_retrieval_sd"],
                        marker="o", ms=4, lw=1.6, capsize=2, color=colour, label=name)
        ax.set_title(artefact, fontsize=11)
        ax.set_xlabel(spec["unit"])
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("self-retrieval rate")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Stability of each representation under measurement artefacts "
                 f"({args.repeats} random draws per level)")
    fig.tight_layout()
    fig.savefig(FIGURES / "04_robustness.png", dpi=180)
    plt.close(fig)

    print(f"\nwritten to {RESULTS / 'robustness.csv'} and "
          f"{FIGURES / '04_robustness.png'}")


if __name__ == "__main__":
    main()
