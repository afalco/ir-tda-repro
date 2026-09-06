#!/usr/bin/env python3
"""Step 2 -- persistence diagrams, persistence images and diagram distances.

Each normalised spectrum is treated as a one-dimensional filtration function.
The degree-0 diagram of its superlevel-set filtration pairs every absorption
band with the level at which it merges into a stronger neighbour, so that the
persistence of a feature is the topological prominence of the band.

Two vectorisations are produced:

    PI   the classical persistence image over (birth, lifetime), as in
         Frahi et al. (2020). It is invariant under reparametrisation of the
         wavenumber axis and therefore ignores band positions.
    TFI  the "topological fingerprint image" over (wavenumber, lifetime),
         which keeps the prominence-based robustness of persistence while
         restoring the chemical information carried by band positions.

Writes:
    results/diagrams.npz
    results/persistence_images.npy
    results/fingerprint_images.npy
    results/distance_sliced_wasserstein.npy
    results/diagram_summary.csv
    figures/02_*.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irtda import clustering, dataset, features, persistence, plotting  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--normalisation", default="minmax",
                   choices=["minmax", "max", "snv", "none"])
    p.add_argument("--kind", default="peaks", choices=["peaks", "valleys"],
                   help="filtration direction: absorption bands or troughs")
    p.add_argument("--min-persistence", type=float, default=2e-3,
                   help="features below this prominence are treated as noise")
    p.add_argument("--max-features", type=int, default=300,
                   help="keep at most this many features per diagram")
    p.add_argument("--resolution", type=int, default=20,
                   help="persistence image side length in pixels")
    p.add_argument("--sigma", type=float, default=None,
                   help="Gaussian kernel width for the classical image")
    p.add_argument("--sigma-wavenumber", type=float, default=25.0,
                   help="kernel width along the wavenumber axis of the TFI [cm-1]")
    p.add_argument("--fixed-threshold", action="store_true",
                   help="disable the noise-adaptive pruning threshold")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    data = np.load(PROCESSED / "spectra.npz", allow_pickle=True)
    labels = pd.read_csv(PROCESSED / "labels.csv")
    wavenumber = data["wavenumber"]
    intensity = dataset.normalise(data["intensity"], args.normalisation)

    # -- persistence diagrams ------------------------------------------------
    config = features.DescriptorConfig(
        kind=args.kind,
        min_persistence=args.min_persistence,
        max_features=args.max_features,
        resolution=args.resolution,
        sigma=args.sigma,
        sigma_wavenumber=args.sigma_wavenumber,
        adaptive_threshold=not args.fixed_threshold,
    )
    descriptors = features.compute_diagrams(intensity, wavenumber, config)
    diagrams = descriptors.diagrams
    raw_sizes = descriptors.n_features_raw

    kept = [len(d) for d in diagrams]
    print(f"pruning threshold: {np.mean(descriptors.thresholds):.2e} "
          f"(min {min(descriptors.thresholds):.2e}, "
          f"max {max(descriptors.thresholds):.2e})")
    print(f"features per diagram: raw {np.mean(raw_sizes):.0f} "
          f"(min {min(raw_sizes)}, max {max(raw_sizes)}) -> "
          f"pruned {np.mean(kept):.0f} (min {min(kept)}, max {max(kept)})")

    np.savez_compressed(
        RESULTS / "diagrams.npz",
        **{f"d{i}": d for i, d in enumerate(diagrams)},
        **{f"f{i}": f for i, f in enumerate(descriptors.fingerprints)},
        sheet=labels["sheet"].to_numpy(),
    )

    # -- vectorisations ------------------------------------------------------
    imager, fingerprint_imager = features.build_imagers(descriptors, config)

    persistence_images = imager.transform(descriptors.lifetimes)
    np.save(RESULTS / "persistence_images.npy", persistence_images)
    print(f"persistence images:  {persistence_images.shape}  "
          f"birth {imager.birth_range[0]:.2f}..{imager.birth_range[1]:.2f}, "
          f"lifetime {imager.lifetime_range[0]:.2f}..{imager.lifetime_range[1]:.2f}")

    fingerprint_images = fingerprint_imager.transform(descriptors.fingerprints)
    np.save(RESULTS / "fingerprint_images.npy", fingerprint_images)
    print(f"fingerprint images:  {fingerprint_images.shape}  "
          f"wavenumber {fingerprint_imager.birth_range[0]:.0f}.."
          f"{fingerprint_imager.birth_range[1]:.0f} cm-1")

    # -- pairwise distances between diagrams ---------------------------------
    np.save(RESULTS / "distance_sliced_wasserstein.npy",
            clustering.distance_matrix(diagrams, metric="sliced"))

    # -- figures -------------------------------------------------------------
    for row in (0, len(diagrams) // 2):
        meta = labels.iloc[row]
        # The first column of the fingerprint diagram is the wavenumber of the
        # maximum generating each feature, row-aligned with the pruned diagram.
        birth_wavenumber = descriptors.fingerprints[row][:, 0]
        plotting.plot_diagram(
            wavenumber, intensity[row], diagrams[row],
            f"sample {meta['sheet']} -- {meta['family']} -- {meta['description_en']}",
            FIGURES / f"02_diagram_{meta['sheet']}.png",
            birth_wavenumber=birth_wavenumber,
        )

    order = labels.sort_values(["family", "sheet"]).index.to_numpy()[:12]
    panel_labels = [f"{labels.sheet[i]} ({labels.family[i]})" for i in order]
    plotting.plot_persistence_image(
        imager, persistence_images[order], panel_labels,
        FIGURES / "02_persistence_images.png",
        title="persistence images  (birth $\\rightarrow$, lifetime $\\uparrow$)",
    )
    plotting.plot_persistence_image(
        fingerprint_imager, fingerprint_images[order], panel_labels,
        FIGURES / "02_fingerprint_images.png",
        title="topological fingerprint images  "
              "(wavenumber $\\rightarrow$, lifetime $\\uparrow$)",
    )

    pd.DataFrame({
        "sheet": labels["sheet"],
        "family": labels["family"],
        "n_features_raw": raw_sizes,
        "n_features_kept": kept,
        "pruning_threshold": descriptors.thresholds,
        "total_persistence": [float(persistence.persistence_of(d).sum()) for d in diagrams],
        "max_persistence": [float(persistence.persistence_of(d).max()) for d in diagrams],
    }).to_csv(RESULTS / "diagram_summary.csv", index=False)

    print(f"\nwritten to {RESULTS}")


if __name__ == "__main__":
    main()
