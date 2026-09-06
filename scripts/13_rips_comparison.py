#!/usr/bin/env python3
"""Step 13 -- the Vietoris-Rips pipeline, on the same data.

The argument for the lower-star filtration made in this work is a construction
argument: for a function sampled on a line the degree-0 diagram of the
superlevel-set filtration is exact, parameter-free and chemically readable,
whereas the H1 classes of a Vietoris-Rips complex built on the plane curve are
largely an artefact of the sampling density and of the arbitrary ratio between
the units of the two axes. Conti et al. (2023) take the opposite view for Raman
spectra, on the ground that a lower-star filtration produces no H1 at all.

An argument by construction is not evidence. This script runs their pipeline on
these spectra and compares it with ours on the same three tasks, and it puts a
number on the objection: the ratio between the axes, which nothing in the
measurement fixes, is swept over two orders of magnitude.

    A  how the H1 content depends on the axis ratio;
    B  the three tasks, head to head;
    C  what each construction costs.

Writes:
    results/rips_aspect.csv
    results/rips_comparison.csv
    results/rips_cost.csv
    figures/13_rips.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from irtda import clustering, dataset, features, images, peaks, persistence, rips  # noqa: E402

_reg = __import__("07_property_regression")

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

ASPECTS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--parts", default="ABC")
    p.add_argument("--n-points", type=int, default=600)
    return p.parse_args()


def rips_for_all(X, wavenumber, config):
    return [rips.rips_diagrams(row, wavenumber, config) for row in X]


def evaluate(name, F, y, keep, families, main, rows, seed):
    q2 = _reg.scores(y[keep], _reg.ridge_predict(F[keep], y[keep]))["Q2"]
    ari_all = clustering.evaluate(
        families, clustering.kmeans_labels(F, 6, seed=seed), F)["ARI"]
    scores = clustering.evaluate(
        families[main], clustering.kmeans_labels(F[main], 3, seed=seed), F[main])
    rows.append(dict(representation=name, Q2_T5=q2, ARI_all=ari_all,
                     ARI_main=scores["ARI"], AMI_main=scores["AMI"]))
    return rows[-1]


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(PROCESSED / "labels.csv")
    targets = pd.read_csv(PROCESSED / "targets.csv")
    data = np.load(PROCESSED / "spectra.npz", allow_pickle=True)
    wavenumber, raw = data["wavenumber"], data["intensity"]
    families = labels["family"].to_numpy()
    main_subset = np.isin(families, ["TPU", "PUR", "TR"])
    X = dataset.normalise(raw, "minmax")
    y = targets["T5"].to_numpy(dtype=float)
    keep = np.isfinite(y)

    # -- A. how much of H1 is the spectrum, and how much the axis ratio? ----
    if "A" in args.parts:
        print("\nA. dependence of the Vietoris-Rips diagram on the axis ratio")
        rows = []
        for aspect in ASPECTS:
            config = rips.RipsConfig(aspect=aspect, n_points=args.n_points)
            diagrams = rips_for_all(X, wavenumber, config)
            n_h1 = np.array([len(d[1]) for d in diagrams], dtype=float)
            total_h1 = np.array([float((d[1][:, 1] - d[1][:, 0]).sum())
                                 if d[1].size else 0.0 for d in diagrams])
            curves = rips.betti_features(diagrams, config)
            entry = evaluate(f"VR H1, aspect {aspect:g}", curves["H1"], y, keep,
                             families, main_subset, [], args.seed)
            rows.append(dict(aspect=aspect, n_H1_mean=float(n_h1.mean()),
                             n_H1_min=float(n_h1.min()), n_H1_max=float(n_h1.max()),
                             total_H1_persistence=float(total_h1.mean()),
                             Q2_T5=entry["Q2_T5"], ARI_main=entry["ARI_main"]))
            print(f"  aspect {aspect:5g}  H1 features {n_h1.mean():6.1f} "
                  f"[{n_h1.min():.0f}, {n_h1.max():.0f}]   "
                  f"Q2(T5)={entry['Q2_T5']:6.3f}  ARI={entry['ARI_main']:.3f}")
        frame = pd.DataFrame(rows)
        frame.to_csv(RESULTS / "rips_aspect.csv", index=False)
        print(f"\n  H1 count varies by a factor of "
              f"{frame.n_H1_mean.max() / max(frame.n_H1_mean.min(), 1e-9):.0f} "
              f"across the sweep; Q2 spans "
              f"[{frame.Q2_T5.min():.3f}, {frame.Q2_T5.max():.3f}] and ARI "
              f"[{frame.ARI_main.min():.3f}, {frame.ARI_main.max():.3f}].")

    # -- B. head to head -----------------------------------------------------
    if "B" in args.parts:
        print("\nB. the two constructions on the same three tasks")
        rows = []
        config = rips.RipsConfig(aspect=1.0, n_points=args.n_points)
        diagrams = rips_for_all(X, wavenumber, config)
        curves = rips.betti_features(diagrams, config)
        for degree in ("H0", "H1", "H0+H1"):
            evaluate(f"Vietoris-Rips, Betti {degree}", curves[degree], y, keep,
                     families, main_subset, rows, args.seed)

        descriptor_config = features.DescriptorConfig(adaptive_threshold=False)
        descriptors = features.compute_diagrams(X, wavenumber, descriptor_config)
        imager, fingerprint_imager = features.build_imagers(
            descriptors, descriptor_config)
        evaluate("Lower-star, persistence image",
                 imager.transform(descriptors.lifetimes), y, keep, families,
                 main_subset, rows, args.seed)
        evaluate("Lower-star, fingerprint image",
                 fingerprint_imager.transform(descriptors.fingerprints), y, keep,
                 families, main_subset, rows, args.seed)

        tables = [peaks.peak_table(row, wavenumber, peaks.PeakConfig(
            prominence=descriptor_config.min_persistence)) for row in X]
        peak_diagrams = [peaks.attribute_diagram(t, "prominence") for t in tables]
        probe = images.PersistenceImager(resolution=descriptor_config.resolution)
        probe.fit(peak_diagrams)
        pixel = probe.lifetime_range[1] / descriptor_config.resolution
        peak_imager = images.PersistenceImager(
            resolution=descriptor_config.resolution,
            sigma=(descriptor_config.sigma_wavenumber, max(pixel, 1e-6)))
        peak_imager.fit(peak_diagrams)
        evaluate("Peaks, position + prominence",
                 peak_imager.transform(peak_diagrams), y, keep, families,
                 main_subset, rows, args.seed)
        evaluate("Raw spectra (SNV)", dataset.normalise(raw, "snv"), y, keep,
                 families, main_subset, rows, args.seed)

        frame = pd.DataFrame(rows)
        frame.to_csv(RESULTS / "rips_comparison.csv", index=False)
        print(frame.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # -- C. what each construction costs ------------------------------------
    if "C" in args.parts:
        print("\nC. cost per spectrum")
        rows = []
        for n_points in (300, 600, 1000):
            config = rips.RipsConfig(aspect=1.0, n_points=n_points)
            start = time.perf_counter()
            for row in X[:5]:
                rips.rips_diagrams(row, wavenumber, config)
            rows.append(dict(construction="Vietoris-Rips", points=n_points,
                             seconds=(time.perf_counter() - start) / 5))
        start = time.perf_counter()
        for row in X:
            persistence.superlevel_persistence(row, return_locations=True)
        rows.append(dict(construction="lower-star (union-find)",
                         points=len(wavenumber),
                         seconds=(time.perf_counter() - start) / len(X)))
        frame = pd.DataFrame(rows)
        frame.to_csv(RESULTS / "rips_cost.csv", index=False)
        print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        vr = frame.query("construction == 'Vietoris-Rips' and points == 600").seconds.iloc[0]
        ls = frame.query("construction != 'Vietoris-Rips'").seconds.iloc[0]
        print(f"\n  at 600 subsampled points Vietoris-Rips costs {vr / ls:.0f} times "
              f"the lower-star sweep over all {len(wavenumber)} channels")

    # -- figure --------------------------------------------------------------
    aspect_path, comparison_path = RESULTS / "rips_aspect.csv", RESULTS / "rips_comparison.csv"
    if aspect_path.exists() and comparison_path.exists():
        sweep = pd.read_csv(aspect_path)
        head = pd.read_csv(comparison_path)
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

        ax = axes[0]
        ax.plot(sweep.aspect, sweep.n_H1_mean, marker="o", color="#b05800")
        ax.fill_between(sweep.aspect, sweep.n_H1_min, sweep.n_H1_max,
                        color="#b05800", alpha=0.15)
        ax.set_xscale("log")
        ax.set_xlabel("ratio between the axes (arbitrary)")
        ax.set_ylabel(r"$H_1$ features per spectrum")
        ax.set_title(r"(a) the $H_1$ content is a choice, not a measurement",
                     fontsize=10)

        ax = axes[1]
        ax.axhline(0, color="#888888", lw=0.8)
        ax.plot(sweep.aspect, sweep.Q2_T5, marker="o", color="#b05800",
                label=r"$Q^2$, onset $T_5$")
        ax.plot(sweep.aspect, sweep.ARI_main, marker="s", color="#c9a227",
                label="ARI, families")
        tfi = head.set_index("representation").loc["Lower-star, fingerprint image"]
        ax.axhline(tfi.Q2_T5, color="#0046a0", lw=2,
                   label=r"fingerprint image, $Q^2$")
        ax.set_xscale("log")
        ax.set_xlabel("ratio between the axes (arbitrary)")
        ax.set_ylabel("score")
        ax.set_title(r"(b) and so is everything $H_1$ predicts", fontsize=10)
        ax.legend(fontsize=7.5, frameon=False, loc="lower left")

        ax = axes[2]
        order = ["Vietoris-Rips, Betti H1", "Vietoris-Rips, Betti H0+H1",
                 "Vietoris-Rips, Betti H0", "Lower-star, persistence image",
                 "Lower-star, fingerprint image", "Peaks, position + prominence",
                 "Raw spectra (SNV)"]
        table = head.set_index("representation").reindex(order)
        colour = ["#0046a0" if "Lower-star" in n else
                  "#444444" if "Raw" in n else "#b05800" for n in order]
        y_pos = np.arange(len(order))
        ax.barh(y_pos - 0.2, table.Q2_T5, height=0.38, color=colour,
                label=r"$Q^2$, onset $T_5$")
        ax.barh(y_pos + 0.2, table.ARI_main, height=0.38, color=colour, alpha=0.45,
                label="ARI, families")
        ax.axvline(0, color="#888888", lw=0.8)
        ax.set_yticks(y_pos, [n.replace("Vietoris-Rips, ", "VR ")
                              .replace("Lower-star, ", "") for n in order],
                      fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("score")
        ax.set_title("(c) the two constructions, same data", fontsize=10)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")

        fig.tight_layout()
        fig.savefig(FIGURES / "13_rips.png", dpi=180)
        plt.close(fig)
        print(f"figure written to {FIGURES / '13_rips.png'}")

    print(f"\nwritten to {RESULTS}")


if __name__ == "__main__":
    main()
