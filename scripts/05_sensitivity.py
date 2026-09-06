#!/usr/bin/env python3
"""Step 5 -- sensitivity of the results to the descriptor hyper-parameters.

Sweeps the pruning threshold, the image resolution and the kernel width, and
scores the resulting clustering of the TPU / PUR / TR subset each time. A
descriptor whose score depends strongly on these choices would be reporting the
choices rather than the data, so this is a check on the previous step rather
than a search for the best setting.

Writes:
    results/sensitivity.csv
    figures/05_sensitivity.png
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

from irtda import clustering, dataset, features  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

MAIN_FAMILIES = ("TPU", "PUR", "TR")
THRESHOLDS = (1e-3, 2e-3, 5e-3, 1e-2)
RESOLUTIONS = (16, 24, 32, 48, 64)
KERNEL_WIDTHS = (10.0, 20.0, 40.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    data = np.load(PROCESSED / "spectra.npz", allow_pickle=True)
    labels = pd.read_csv(PROCESSED / "labels.csv")
    wavenumber = data["wavenumber"]
    intensity = dataset.normalise(data["intensity"], "minmax")

    families = labels["family"].to_numpy()
    mask = np.isin(families, MAIN_FAMILIES)

    rows = []
    for threshold in THRESHOLDS:
        configs = [
            features.DescriptorConfig(min_persistence=threshold,
                                      adaptive_threshold=False)
        ]
        descriptors = features.compute_diagrams_multi(
            intensity, wavenumber, configs
        )[0]
        n_features = float(np.mean([len(d) for d in descriptors.diagrams]))

        for resolution in RESOLUTIONS:
            for width in KERNEL_WIDTHS:
                config = features.DescriptorConfig(
                    min_persistence=threshold, resolution=resolution,
                    sigma_wavenumber=width, adaptive_threshold=False,
                )
                _, fingerprint_imager = features.build_imagers(descriptors, config)
                vectors = fingerprint_imager.transform(descriptors.fingerprints)

                predicted = clustering.kmeans_labels(
                    vectors[mask], len(MAIN_FAMILIES), seed=args.seed
                )
                scores = clustering.evaluate(families[mask], predicted)
                rows.append({
                    "min_persistence": threshold,
                    "mean_n_features": n_features,
                    "resolution": resolution,
                    "sigma_wavenumber": width,
                    **scores,
                })

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "sensitivity.csv", index=False)

    modal = table["ARI"].round(3).mode().iloc[0]
    n_modal = int((table["ARI"].round(3) == modal).sum())
    print(f"{len(table)} configurations; ARI ranges "
          f"{table['ARI'].min():.3f}-{table['ARI'].max():.3f}, "
          f"equal to {modal:.3f} in {n_modal} of them")
    print(table.groupby("min_persistence")[["ARI", "AMI"]]
          .agg(["mean", "min", "max"]).to_string())

    # -- figure --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for width, marker in zip(KERNEL_WIDTHS, "os^"):
        for ax, (key, xlabel) in zip(
            axes, [("resolution", "image resolution [pixels per axis]"),
                   ("min_persistence", "pruning threshold")]
        ):
            sub = table[table["sigma_wavenumber"] == width].groupby(key)["ARI"]
            ax.plot(sub.mean().index, sub.mean().values, marker=marker, lw=1.6,
                    label=f"kernel width {width:.0f} cm$^{{-1}}$")
            ax.set_xlabel(xlabel)
            ax.grid(alpha=0.25)
    axes[1].set_xscale("log")
    axes[0].set_ylabel("ARI, TPU/PUR/TR subset")
    axes[0].set_ylim(0, 0.6)
    axes[0].legend(fontsize=8)
    fig.suptitle("Sensitivity of the topological fingerprint image "
                 "to its hyper-parameters")
    fig.tight_layout()
    fig.savefig(FIGURES / "05_sensitivity.png", dpi=180)
    plt.close(fig)

    print(f"\nwritten to {RESULTS / 'sensitivity.csv'} and "
          f"{FIGURES / '05_sensitivity.png'}")


if __name__ == "__main__":
    main()
