#!/usr/bin/env python3
"""Step 3 -- unsupervised grouping of the materials and comparison with the labels.

Four representations are compared under the same clustering protocol:

    PI        k-means on the classical persistence images
    TFI       k-means on the position-aware topological fingerprint images
    W-sliced  average-linkage clustering of the sliced Wasserstein distances
              between persistence diagrams
    baseline  k-means on the raw spectra after standard normal variate
              correction, i.e. no topology at all

The material family read from the reference list is never used to build the
clusters; it only scores them, through the adjusted Rand index (ARI) and the
adjusted mutual information (AMI). Both are corrected for chance, so 0 means
"no better than a random partition" and 1 means "exact agreement".

Writes:
    results/clustering_scores.csv
    results/cluster_assignments*.csv
    results/contingency_*.csv
    figures/03_*.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irtda import clustering, dataset, plotting  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Families represented by a single sample cannot be recovered by clustering and
# distort the agreement indices; a second evaluation is therefore restricted to
# the families with enough replicates.
MAIN_FAMILIES = ("TPU", "PUR", "TR")

SLUGS = {
    "PI (k-means)": "pi",
    "TFI (k-means)": "tfi",
    "W-sliced (average linkage)": "wsliced",
    "baseline: raw spectra (k-means)": "baseline",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--linkage", default="average",
                   choices=["average", "complete", "single"])
    return p.parse_args()


def run_all(pi, tfi, distances, spectra_snv, families, n_clusters, seed, linkage):
    """Cluster every representation and score it against the reference families."""
    runs: dict[str, tuple[np.ndarray, dict[str, float]]] = {}

    for name, features in (("PI (k-means)", pi), ("TFI (k-means)", tfi),
                           ("baseline: raw spectra (k-means)", spectra_snv)):
        predicted = clustering.kmeans_labels(features, n_clusters, seed=seed)
        runs[name] = (predicted, clustering.evaluate(families, predicted, features))

    predicted = clustering.hierarchical_labels(
        distances, n_clusters, precomputed=True, linkage=linkage
    )
    runs["W-sliced (average linkage)"] = (
        predicted,
        clustering.evaluate(families, predicted, distances, precomputed=True),
    )
    return runs


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(PROCESSED / "labels.csv")
    data = np.load(PROCESSED / "spectra.npz", allow_pickle=True)
    pi = np.load(RESULTS / "persistence_images.npy")
    tfi = np.load(RESULTS / "fingerprint_images.npy")
    distances = np.load(RESULTS / "distance_sliced_wasserstein.npy")
    spectra_snv = dataset.normalise(data["intensity"], "snv")

    families = labels["family"].to_numpy()
    sheets = labels["sheet"].astype(str).to_numpy()

    rows: list[dict] = []
    assignments = {"sheet": sheets, "family": families}

    # -- evaluation A: all samples, one cluster per family -------------------
    n_all = len(set(families))
    runs_all = run_all(pi, tfi, distances, spectra_snv, families,
                       n_all, args.seed, args.linkage)
    for name, (predicted, scores) in runs_all.items():
        rows.append({"subset": f"all (n={len(families)})", "k": n_all,
                     "method": name, **scores})
        assignments[name] = predicted

    # -- evaluation B: the three well-represented families -------------------
    mask = np.isin(families, MAIN_FAMILIES)
    runs_main = run_all(
        pi[mask], tfi[mask], distances[np.ix_(mask, mask)], spectra_snv[mask],
        families[mask], len(MAIN_FAMILIES), args.seed, args.linkage,
    )
    sub = {"sheet": sheets[mask], "family": families[mask]}
    for name, (predicted, scores) in runs_main.items():
        rows.append({"subset": f"{'/'.join(MAIN_FAMILIES)} (n={int(mask.sum())})",
                     "k": len(MAIN_FAMILIES), "method": name, **scores})
        sub[name] = predicted

    scores_df = pd.DataFrame(rows)
    scores_df.to_csv(RESULTS / "clustering_scores.csv", index=False)
    pd.DataFrame(assignments).to_csv(RESULTS / "cluster_assignments.csv", index=False)
    pd.DataFrame(sub).to_csv(RESULTS / "cluster_assignments_main.csv", index=False)

    print(scores_df.to_string(index=False, float_format=lambda v: f"{v:6.3f}"))

    # -- contingency tables --------------------------------------------------
    for tag, runs, fams in (("all", runs_all, families),
                            ("main", runs_main, families[mask])):
        for name, (predicted, _) in runs.items():
            table = clustering.contingency_table(fams, predicted)
            table.to_csv(RESULTS / f"contingency_{tag}_{SLUGS[name]}.csv")
            plotting.plot_contingency(
                table, FIGURES / f"03_contingency_{tag}_{SLUGS[name]}.png",
                f"{name} -- {tag}",
            )

    # -- figures -------------------------------------------------------------
    plotting.plot_dendrogram(distances, sheets, families,
                             FIGURES / "03_dendrogram.png", method=args.linkage)

    coords = MDS(n_components=2, dissimilarity="precomputed", n_init=4,
                 random_state=args.seed,
                 normalized_stress="auto").fit_transform(distances)
    plotting.plot_embedding(coords, sheets, families, FIGURES / "03_mds_diagrams.png",
                            "MDS of the sliced Wasserstein distance between diagrams")

    for tag, features, name in (("pi", pi, "persistence images"),
                                ("tfi", tfi, "topological fingerprint images")):
        pca = PCA(n_components=2, random_state=args.seed)
        coords = pca.fit_transform(features)
        plotting.plot_embedding(
            coords, sheets, families, FIGURES / f"03_pca_{tag}.png",
            f"PCA of the {name} "
            f"({100 * pca.explained_variance_ratio_.sum():.0f} % of the variance)",
        )

    print(f"\nwritten to {RESULTS} and {FIGURES}")


if __name__ == "__main__":
    main()
