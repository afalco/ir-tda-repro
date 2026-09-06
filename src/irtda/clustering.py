"""Distances between persistence diagrams and unsupervised grouping."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)

__all__ = [
    "sliced_wasserstein",
    "wasserstein",
    "distance_matrix",
    "kmeans_labels",
    "hierarchical_labels",
    "evaluate",
    "contingency_table",
]


# ---------------------------------------------------------------------------
# Distances between diagrams
# ---------------------------------------------------------------------------
def _project_to_diagonal(diagram: np.ndarray) -> np.ndarray:
    """Orthogonal projection of each point onto the diagonal ``birth = death``."""
    if diagram.size == 0:
        return diagram.reshape(-1, 2)
    mid = diagram.mean(axis=1)
    return np.column_stack([mid, mid])


def sliced_wasserstein(
    diagram_a: np.ndarray, diagram_b: np.ndarray, n_directions: int = 64
) -> float:
    """Sliced Wasserstein distance between two persistence diagrams.

    Following Carriere, Cuturi and Oudot (2017): each diagram is augmented with
    the diagonal projections of the other one, both are projected onto a set of
    directions, and the one-dimensional Wasserstein-1 distances are averaged.
    Unlike the exact optimal matching it is cheap enough to be evaluated for
    every pair in the data set, and it is a proper metric.
    """
    a = np.atleast_2d(diagram_a)
    b = np.atleast_2d(diagram_b)
    if a.size == 0 and b.size == 0:
        return 0.0

    a_aug = np.vstack([a, _project_to_diagonal(b)])
    b_aug = np.vstack([b, _project_to_diagonal(a)])

    thetas = np.linspace(-np.pi / 2, np.pi / 2, n_directions, endpoint=False)
    directions = np.column_stack([np.cos(thetas), np.sin(thetas)])

    proj_a = np.sort(a_aug @ directions.T, axis=0)
    proj_b = np.sort(b_aug @ directions.T, axis=0)
    return float(np.abs(proj_a - proj_b).sum(axis=0).mean())


def wasserstein(
    diagram_a: np.ndarray, diagram_b: np.ndarray, order: float = 2.0
) -> float:
    """Exact ``order``-Wasserstein distance via optimal partial matching.

    Points may be matched to each other or to the diagonal; the cost matrix is
    therefore of size ``(n_a + n_b) x (n_a + n_b)``. Prune the diagrams before
    calling this on large inputs.
    """
    a = np.atleast_2d(diagram_a)
    b = np.atleast_2d(diagram_b)
    n_a, n_b = len(a), len(b)
    if n_a == 0 and n_b == 0:
        return 0.0

    size = n_a + n_b
    cost = np.zeros((size, size))

    if n_a and n_b:
        cost[:n_a, :n_b] = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    if n_a:
        cost[:n_a, n_b:] = np.inf
        d_a = np.abs(a[:, 1] - a[:, 0]) / np.sqrt(2.0)
        cost[np.arange(n_a), n_b + np.arange(n_a)] = d_a
    if n_b:
        cost[n_a:, :n_b] = np.inf
        d_b = np.abs(b[:, 1] - b[:, 0]) / np.sqrt(2.0)
        cost[n_a + np.arange(n_b), np.arange(n_b)] = d_b
    cost[n_a:, n_b:] = 0.0

    finite = np.isfinite(cost)
    cost = np.where(finite, cost, cost[finite].max() * 1e6 + 1.0)

    rows, cols = linear_sum_assignment(cost**order)
    return float(cost[rows, cols].__pow__(order).sum() ** (1.0 / order))


def distance_matrix(diagrams: list[np.ndarray], metric: str = "sliced", **kwargs):
    """Symmetric pairwise distance matrix between diagrams."""
    fn = {"sliced": sliced_wasserstein, "wasserstein": wasserstein}[metric]
    n = len(diagrams)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = fn(diagrams[i], diagrams[j], **kwargs)
    return d


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def kmeans_labels(features: np.ndarray, n_clusters: int, seed: int = 0) -> np.ndarray:
    return KMeans(n_clusters=n_clusters, n_init=50, random_state=seed).fit_predict(
        features
    )


def hierarchical_labels(
    data: np.ndarray,
    n_clusters: int,
    precomputed: bool = False,
    linkage: str = "ward",
) -> np.ndarray:
    if precomputed:
        model = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage=linkage
        )
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    return model.fit_predict(data)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    true_labels, predicted_labels, data: np.ndarray | None = None, precomputed=False
) -> dict[str, float]:
    """Agreement between a clustering and the reference material families."""
    scores = {
        "ARI": adjusted_rand_score(true_labels, predicted_labels),
        "AMI": adjusted_mutual_info_score(true_labels, predicted_labels),
    }
    if data is not None and len(set(predicted_labels)) > 1:
        scores["silhouette"] = silhouette_score(
            data, predicted_labels, metric="precomputed" if precomputed else "euclidean"
        )
    return scores


def contingency_table(true_labels, predicted_labels) -> pd.DataFrame:
    """Cross-tabulation of reference families against cluster assignments."""
    return pd.crosstab(
        pd.Series(list(true_labels), name="family"),
        pd.Series([f"C{c}" for c in predicted_labels], name="cluster"),
    )
