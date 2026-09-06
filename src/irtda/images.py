"""Persistence images: a vector representation of persistence diagrams.

Implements the construction of Frahi et al. (2020, Sec. 3.2; 2021, Sec. 2.4):
the lifetime diagram is convolved with an isotropic Gaussian kernel, weighted
by a linear ramp in the lifetime, and integrated over the cells of a regular
``resolution x resolution`` partition of its support. The resulting matrix,
flattened, is a Euclidean vector on which ordinary clustering and classification
algorithms operate.

The pixel integrals are evaluated exactly with the Gaussian CDF rather than by
sampling the density at the pixel centre, which keeps the representation stable
when ``sigma`` is small compared with the pixel size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

__all__ = ["PersistenceImager"]


@dataclass
class PersistenceImager:
    """Vectorise lifetime diagrams as persistence images.

    The input of :meth:`fit` and :meth:`transform` is a list of ``(n, 2)``
    arrays whose first column is the horizontal coordinate and whose second
    column is the lifetime. Passing
    :func:`irtda.persistence.lifetime_diagram` reproduces the classical
    persistence image; passing
    :func:`irtda.persistence.position_lifetime_diagram` yields the
    position-aware variant used for the spectra.

    Parameters
    ----------
    resolution
        Number of pixels per axis; the image has ``resolution ** 2`` entries.
    sigma
        Standard deviation of the Gaussian kernel placed at each feature, in
        the units of the horizontal axis. When ``None`` it is set to one pixel
        width, which is the usual default. Anisotropic kernels are obtained by
        passing a pair ``(sigma_x, sigma_y)``.
    weighting
        ``"ramp"``     -- ``w = lifetime / max(lifetime)`` within each diagram,
                          the weight used in the reference papers (default);
        ``"identity"`` -- ``w = lifetime``;
        ``"uniform"``  -- ``w = 1``.
    padding
        Fraction of the fitted ranges added on each side of the grid.
    """

    resolution: int = 20
    sigma: float | tuple[float, float] | None = None
    weighting: str = "ramp"
    padding: float = 0.05

    birth_range: tuple[float, float] = field(default=(0.0, 1.0), init=False)
    lifetime_range: tuple[float, float] = field(default=(0.0, 1.0), init=False)
    _fitted: bool = field(default=False, init=False)

    # -- weights ------------------------------------------------------------
    def _weights(self, lifetimes: np.ndarray) -> np.ndarray:
        if self.weighting == "uniform":
            return np.ones_like(lifetimes)
        if self.weighting == "identity":
            return lifetimes
        if self.weighting == "ramp":
            largest = lifetimes.max()
            return lifetimes / largest if largest > 0 else np.zeros_like(lifetimes)
        raise ValueError(f"unknown weighting: {self.weighting!r}")

    # -- fitting ------------------------------------------------------------
    def fit(self, diagrams: list[np.ndarray]) -> "PersistenceImager":
        """Fit a common image grid covering the support of all lifetime diagrams.

        A single grid shared by the whole data set is required for the images
        to be comparable as vectors.
        """
        births, lifetimes = [], []
        for lt in diagrams:
            if lt.size == 0:
                continue
            births.append(lt[:, 0])
            lifetimes.append(lt[:, 1])
        if not births:
            raise ValueError("no non-empty diagram to fit")

        b = np.concatenate(births)
        l = np.concatenate(lifetimes)

        b_pad = self.padding * (b.max() - b.min() + 1e-12)
        l_pad = self.padding * (l.max() + 1e-12)
        self.birth_range = (float(b.min() - b_pad), float(b.max() + b_pad))
        self.lifetime_range = (0.0, float(l.max() + l_pad))
        self._fitted = True
        return self

    # -- grid ---------------------------------------------------------------
    @property
    def _edges(self) -> tuple[np.ndarray, np.ndarray]:
        bx = np.linspace(*self.birth_range, self.resolution + 1)
        by = np.linspace(*self.lifetime_range, self.resolution + 1)
        return bx, by

    @property
    def _effective_sigma(self) -> tuple[float, float]:
        bx, by = self._edges
        if self.sigma is None:
            return float(bx[1] - bx[0]), float(by[1] - by[0])
        if np.isscalar(self.sigma):
            return float(self.sigma), float(self.sigma)
        sx, sy = self.sigma  # type: ignore[misc]
        return float(sx), float(sy)

    def pixel_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """Centres of the image pixels along the birth and lifetime axes."""
        bx, by = self._edges
        return 0.5 * (bx[:-1] + bx[1:]), 0.5 * (by[:-1] + by[1:])

    # -- transformation -----------------------------------------------------
    def transform_one(self, lt: np.ndarray) -> np.ndarray:
        """Persistence image of a single lifetime diagram, ``(resolution, resolution)``."""
        if not self._fitted:
            raise RuntimeError("call fit() before transform()")

        image = np.zeros((self.resolution, self.resolution))
        if lt.size == 0:
            return image

        weights = self._weights(lt[:, 1])
        sigma_x, sigma_y = self._effective_sigma
        bx, by = self._edges

        # Exact integral of the separable Gaussian over each pixel.
        cx = norm.cdf((bx[None, :] - lt[:, 0:1]) / sigma_x)  # (n_features, res + 1)
        cy = norm.cdf((by[None, :] - lt[:, 1:2]) / sigma_y)
        px = np.diff(cx, axis=1)  # (n_features, res)
        py = np.diff(cy, axis=1)

        return np.einsum("f,fi,fj->ij", weights, px, py)

    def transform(self, diagrams: list[np.ndarray]) -> np.ndarray:
        """Stack of flattened persistence images, shape ``(n_diagrams, resolution**2)``."""
        return np.vstack([self.transform_one(d).ravel() for d in diagrams])

    def fit_transform(self, diagrams: list[np.ndarray]) -> np.ndarray:
        return self.fit(diagrams).transform(diagrams)
