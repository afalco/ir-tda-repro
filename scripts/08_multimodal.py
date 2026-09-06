#!/usr/bin/env python3
"""Step 8 -- multimodal descriptors combining the IR and the DTG topology.

Infrared alone does not separate thermoplastic from cast polyurethane: both are
polyurethanes and their spectra are dominated by the same bands. Their thermal
decomposition, on the other hand, need not be identical, so a descriptor that
sees both measurements may resolve what neither resolves alone.

Two remarks govern the construction.

First, the TG curve itself is topologically almost empty. Residual mass is a
monotonically decreasing function of temperature, and a monotone function has a
single connected component at every level of its superlevel-set filtration, so
its degree-0 diagram carries one point and no information. The topology lives
in the derivative: each maximum of the DTG curve is one decomposition step and
its prominence says how cleanly that step separates from its neighbours. This
is verified rather than assumed below.

Second, and more importantly, the thermal targets of step 7 (T5, T50, residue
and the peak DTG temperature) are all *derived from the TG curve*. Predicting
them from DTG topology would be circular, and this script therefore evaluates
multimodal descriptors on the identification task only.

Writes:
    results/multimodal_scores.csv
    results/contingency_multimodal.csv
    results/dtg_diagram_summary.csv
    figures/08_multimodal.png
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
from scipy.signal import savgol_filter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irtda import clustering, features, images, persistence, plotting, thermal  # noqa: E402

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

MAIN_FAMILIES = ("TPU", "PUR", "TR")
POLYURETHANES = ("TPU", "PUR")

# Temperature grid on which the DTG curves are compared.
T_MIN, T_MAX, N_T = 40.0, 800.0, 600


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resolution", type=int, default=20)
    p.add_argument("--sigma-temperature", type=float, default=20.0,
                   help="kernel width along the temperature axis [C]")
    p.add_argument("--min-persistence", type=float, default=2e-3)
    return p.parse_args()


def dtg_curves(sheets) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the temperature grid and the TG and DTG curves of each sample.

    Read from `data/processed/tg_curves.npz`, written by step 6, rather than
    from the raw workbook: the workbook belongs to the laboratory that produced
    it and is not redistributed with this repository.
    """
    stored = np.load(PROCESSED / "tg_curves.npz", allow_pickle=True)
    grid = stored["temperature"]
    order = {str(s): i for i, s in enumerate(stored["sheet"])}

    tg, dtg = [], []
    for sheet in sheets:
        w_grid = stored["mass"][order[str(sheet)]]
        smooth = savgol_filter(w_grid, window_length=31, polyorder=3)
        rate = savgol_filter(-np.gradient(smooth, grid), window_length=31, polyorder=3)
        tg.append(smooth)
        dtg.append(rate)

    return grid, np.vstack(tg), np.vstack(dtg)


def block_normalise(*blocks: np.ndarray) -> np.ndarray:
    """Concatenate feature blocks, each scaled to unit Frobenius norm.

    Without this the block with the larger numerical scale would dominate the
    Euclidean distance, and the comparison between one modality and two would
    confound the effect of adding information with the effect of changing the
    scale.
    """
    scaled = []
    for block in blocks:
        norm = np.linalg.norm(block)
        scaled.append(block / norm if norm > 0 else block)
    return np.hstack(scaled)


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(PROCESSED / "labels.csv")
    sheets = labels["sheet"].astype(str).tolist()
    families = labels["family"].to_numpy()

    ir_features = np.load(RESULTS / "fingerprint_images.npy")
    grid, tg, dtg = dtg_curves(sheets)

    # -- the TG curve is topologically trivial; check rather than assume -----
    tg_sizes, dtg_sizes = [], []
    for curve in tg:
        normalised = (curve - curve.min()) / (curve.max() - curve.min())
        tg_sizes.append(len(persistence.spectrum_diagram(normalised, "peaks")))
    for curve in dtg:
        normalised = (curve - curve.min()) / (curve.max() - curve.min())
        dtg_sizes.append(len(persistence.spectrum_diagram(normalised, "peaks")))

    print(f"features in the TG diagram:  mean {np.mean(tg_sizes):5.1f} "
          f"(min {min(tg_sizes)}, max {max(tg_sizes)})")
    print(f"features in the DTG diagram: mean {np.mean(dtg_sizes):5.1f} "
          f"(min {min(dtg_sizes)}, max {max(dtg_sizes)})")

    # -- DTG fingerprint images ---------------------------------------------
    config = features.DescriptorConfig(
        min_persistence=args.min_persistence,
        resolution=args.resolution,
        sigma_wavenumber=args.sigma_temperature,
        adaptive_threshold=False,
    )
    dtg_normalised = np.vstack([
        (c - c.min()) / (c.max() - c.min()) for c in dtg
    ])
    descriptors = features.compute_diagrams(dtg_normalised, grid, config)

    lifetime_pixel = 1.0 / args.resolution
    imager = images.PersistenceImager(
        resolution=args.resolution,
        sigma=(args.sigma_temperature, lifetime_pixel),
    )
    dtg_features = imager.fit_transform(descriptors.fingerprints)

    pd.DataFrame({
        "sheet": sheets,
        "family": families,
        "n_features_tg": tg_sizes,
        "n_features_dtg": dtg_sizes,
        "n_features_dtg_pruned": [len(d) for d in descriptors.diagrams],
    }).to_csv(RESULTS / "dtg_diagram_summary.csv", index=False)

    # -- representations to compare -----------------------------------------
    # Two non-topological controls are included. If clustering the DTG curve
    # directly, or the handful of scalars already extracted from it in step 6,
    # separates the polyurethanes as well as the DTG persistence image does,
    # then the topology is contributing nothing and the result belongs to the
    # thermal measurement rather than to the method.
    scalar_targets = pd.read_csv(PROCESSED / "targets.csv")
    scalars = scalar_targets[list(thermal.TG_TARGETS)].to_numpy(dtype=float)
    scalars = (scalars - scalars.mean(axis=0)) / scalars.std(axis=0)

    representations = {
        "IR only (TFI)": block_normalise(ir_features),
        "DTG only (TFI)": block_normalise(dtg_features),
        "IR + DTG (TFI)": block_normalise(ir_features, dtg_features),
        "control: raw DTG curve": block_normalise(dtg_normalised),
        "control: TG scalars": block_normalise(scalars),
    }

    rows = []
    contingencies = {}
    for name, X in representations.items():
        for subset, mask, k in (
            (f"all (n={len(families)})", np.ones(len(families), bool),
             len(set(families))),
            (f"TPU/PUR/TR (n={int(np.isin(families, MAIN_FAMILIES).sum())})",
             np.isin(families, MAIN_FAMILIES), len(MAIN_FAMILIES)),
            (f"TPU/PUR (n={int(np.isin(families, POLYURETHANES).sum())})",
             np.isin(families, POLYURETHANES), len(POLYURETHANES)),
        ):
            predicted = clustering.kmeans_labels(X[mask], k, seed=args.seed)
            rows.append({"subset": subset, "representation": name, "k": k,
                         **clustering.evaluate(families[mask], predicted, X[mask])})
            if subset.startswith("TPU/PUR ("):
                contingencies[name] = clustering.contingency_table(
                    families[mask], predicted
                )

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "multimodal_scores.csv", index=False)
    print()
    print(table.to_string(index=False, float_format=lambda v: f"{v:6.3f}"))

    combined = pd.concat(contingencies, names=["representation"])
    combined.to_csv(RESULTS / "contingency_multimodal.csv")
    print("\nTPU vs PUR, two clusters:")
    print(combined.to_string())

    # -- figure --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))

    for curve, family in zip(dtg, families):
        axes[0].plot(grid, curve, lw=0.8, alpha=0.75,
                     color=plotting.FAMILY_COLOURS.get(family, "#7f7f7f"))
    axes[0].set_xlabel("temperature [$^\\circ$C]")
    axes[0].set_ylabel("mass-loss rate [% $^\\circ$C$^{-1}$]")
    axes[0].set_title("DTG curves")
    axes[0].grid(alpha=0.25)

    order = np.argsort(families, kind="stable")
    axes[1].imshow(dtg_features[order], aspect="auto", cmap="viridis")
    axes[1].set_title("DTG fingerprint images")
    axes[1].set_xlabel("flattened pixel")
    axes[1].set_ylabel("sample, grouped by family")

    sub = table[table["subset"].str.startswith("TPU/PUR (")]
    colours = {"IR only (TFI)": "#7f7f7f", "DTG only (TFI)": "#2ca02c",
               "IR + DTG (TFI)": "#1f77b4", "control: raw DTG curve": "#d62728",
               "control: TG scalars": "#ff7f0e"}
    axes[2].bar(range(len(sub)), sub["ARI"],
                color=[colours[r] for r in sub["representation"]])
    axes[2].set_xticks(range(len(sub)), sub["representation"], rotation=20,
                       ha="right", fontsize=9)
    axes[2].set_ylabel("ARI, TPU vs PUR")
    axes[2].set_title("Separating the two polyurethanes")
    axes[2].grid(alpha=0.25, axis="y")

    fig.suptitle("Multimodal topological descriptors: infrared and DTG")
    fig.tight_layout()
    fig.savefig(FIGURES / "08_multimodal.png", dpi=180)
    plt.close(fig)

    print(f"\nwritten to {RESULTS} and {FIGURES / '08_multimodal.png'}")


if __name__ == "__main__":
    main()
