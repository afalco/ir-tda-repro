"""Vietoris--Rips descriptors of a spectrum, as used in the literature.

Conti et al. (2023) analyse Raman spectra by treating each sample of the
spectrum as a point of the plane and building a Vietoris--Rips filtration on the
resulting point cloud, vectorising the diagrams as Betti curves. They state
their reason for preferring it over a lower-star filtration plainly: the
lower-star construction "does not generate points in H1 for 1D signals".

This module implements that pipeline so that the two can be compared on the same
data rather than argued about. It also exposes the parameter the comparison
turns on. A point cloud in the plane needs a metric, and the two axes of a
spectrum carry different units --- wavenumber and absorbance --- so the distance
between two samples depends on a ratio between those units that nothing in the
measurement fixes. ``aspect`` is that ratio: the wavenumber range is mapped to
``[0, aspect]`` while the normalised absorbance occupies ``[0, 1]``. Sweeping it
shows how much of the H1 content is a property of the spectrum and how much of
the choice.

The construction is also expensive where the lower-star one is not. Vietoris--
Rips on ``n`` points is quadratic in memory before it is anything else, so the
spectrum must be subsampled; the union-find sweep of :mod:`irtda.persistence`
runs on all 3600 channels in milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["RipsConfig", "point_cloud", "rips_diagrams", "betti_curve",
           "betti_features"]


@dataclass(frozen=True)
class RipsConfig:
    """Parameters of the Vietoris--Rips pipeline.

    ``aspect`` is the ratio between the two axes and has no value the
    measurement determines; ``n_points`` is the subsample the quadratic cost
    forces. Neither has a counterpart in the lower-star construction.
    """

    aspect: float = 1.0
    n_points: int = 600
    maxdim: int = 1
    #: Grid resolution of the Betti curves.
    resolution: int = 100


def point_cloud(intensity: np.ndarray, wavenumber: np.ndarray,
                config: RipsConfig) -> np.ndarray:
    """The spectrum as a subsampled planar point cloud."""
    index = np.linspace(0, len(wavenumber) - 1, config.n_points).astype(int)
    span = wavenumber[-1] - wavenumber[0]
    x = config.aspect * (wavenumber[index] - wavenumber[0]) / span
    return np.column_stack([x, np.asarray(intensity)[index]])


def rips_diagrams(intensity: np.ndarray, wavenumber: np.ndarray,
                  config: RipsConfig) -> list[np.ndarray]:
    """Degree-0 and degree-1 Vietoris--Rips diagrams of one spectrum."""
    import ripser

    cloud = point_cloud(intensity, wavenumber, config)
    diagrams = ripser.ripser(cloud, maxdim=config.maxdim)["dgms"]
    out = []
    for diagram in diagrams:
        diagram = np.asarray(diagram, dtype=float).reshape(-1, 2)
        if diagram.size:
            finite = diagram[np.isfinite(diagram[:, 1])]
            ceiling = finite[:, 1].max() if finite.size else 1.0
            diagram = diagram.copy()
            diagram[~np.isfinite(diagram[:, 1]), 1] = ceiling
        out.append(diagram)
    return out


def betti_curve(diagram: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Number of features alive at each filtration value.

    This is the vectorisation used in the reference pipeline: a curve rather
    than an image, and one that ignores where in the diagram the features sit
    beyond their birth and death.
    """
    if diagram.size == 0:
        return np.zeros(len(grid))
    born = diagram[:, 0][None, :] <= grid[:, None]
    alive = diagram[:, 1][None, :] > grid[:, None]
    return (born & alive).sum(axis=1).astype(float)


def betti_features(diagrams_per_spectrum: list[list[np.ndarray]],
                   config: RipsConfig) -> dict[str, np.ndarray]:
    """Betti curves of every spectrum, per homological degree and concatenated.

    The grid is fitted once over all diagrams so that the curves are comparable
    as vectors, exactly as the image grids are in :mod:`irtda.images`.
    """
    degrees = config.maxdim + 1
    ceiling = max(
        (float(d[:, 1].max()) for spectrum in diagrams_per_spectrum
         for d in spectrum if d.size), default=1.0)
    grid = np.linspace(0.0, ceiling, config.resolution)

    curves = {}
    for degree in range(degrees):
        curves[f"H{degree}"] = np.vstack([
            betti_curve(spectrum[degree], grid)
            for spectrum in diagrams_per_spectrum])
    curves["H0+H1"] = np.hstack([curves[f"H{d}"] for d in range(degrees)])
    return curves
