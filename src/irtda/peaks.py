"""Conventional peak descriptors, as a control for the topological ones.

The topological fingerprint image attaches a persistence to the wavenumber of
every absorption band. A spectroscopist reaching for the same information would
instead run a peak picker and tabulate, for each detected band, its position,
intensity, width, area and prominence. Whether persistent homology contributes
anything beyond that table is a question about the *features*, not about the
regression model, and answering it requires a control that differs from the
fingerprint image in one respect only: where the marked points come from.

This module builds that control. :func:`peak_table` runs
:func:`scipy.signal.find_peaks` and returns the five conventional attributes;
:func:`attribute_diagram` pairs any one of them with the peak position, giving a
marked point set of exactly the shape that
:class:`irtda.images.PersistenceImager` consumes. Passing it through the same
imager, with the same anisotropic kernel and the same ramp weighting used for
the fingerprint image, produces a descriptor that is comparable to the
topological one term by term.

One of the five attributes deserves a warning, and it is the point of the
exercise. Topographic prominence is, for a one-dimensional signal, the same
quantity as degree-0 persistence of the superlevel-set filtration; the
"peak prominence" control is therefore expected to behave almost exactly like
the fingerprint image. What separates them is not the quantity but how it is
obtained: the union-find sweep returns every feature of the signal exactly and
without a parameter, whereas ``find_peaks`` returns the features that survive
whatever combination of ``height``, ``prominence``, ``distance`` and ``width``
thresholds the analyst chose. :func:`peak_table` exposes those thresholds so
that the sensitivity of the conventional route can be measured against the
insensitivity of the topological one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths

__all__ = [
    "PeakConfig",
    "PeakTable",
    "ATTRIBUTES",
    "peak_table",
    "attribute_diagram",
]

#: Conventional attributes a peak table carries, in the order used in reporting.
ATTRIBUTES = ("position", "intensity", "prominence", "width", "area")


@dataclass(frozen=True)
class PeakConfig:
    """Thresholds handed to :func:`scipy.signal.find_peaks`.

    The defaults mirror the pruning of the topological pipeline
    (``DescriptorConfig.min_persistence``), so that the two routes start from
    comparable feature sets. Every field is a knob the topological construction
    does not have, which is what the sensitivity analysis exploits.
    """

    prominence: float = 2e-3
    height: float | None = None
    distance: int | None = None
    width: float | None = None
    #: Relative height at which the width is measured; 0.5 gives the FWHM.
    rel_height: float = 0.5


@dataclass
class PeakTable:
    """The conventional description of the bands of one spectrum."""

    position: np.ndarray     # wavenumber of the maximum
    intensity: np.ndarray    # absorbance at the maximum
    prominence: np.ndarray   # topographic prominence
    width: np.ndarray        # full width at ``rel_height``, in wavenumber units
    area: np.ndarray         # integral above the prominence base

    def __len__(self) -> int:
        return len(self.position)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in ATTRIBUTES}


def _integrate_above_base(
    intensity: np.ndarray, wavenumber: np.ndarray,
    left: np.ndarray, right: np.ndarray, base: np.ndarray,
) -> np.ndarray:
    """Trapezoidal integral of ``intensity - base`` between fractional indices.

    The bounds are fractional, so the two partial cells at the ends are added
    separately rather than rounded away --- for a narrow band those cells are a
    large part of the area.
    """
    grid = np.arange(len(intensity), dtype=float)
    areas = np.empty(len(left))
    for k, (lo, hi, floor) in enumerate(zip(left, right, base)):
        inner = np.arange(int(np.ceil(lo)), int(np.floor(hi)) + 1)
        xs = np.concatenate([[lo], inner, [hi]])
        ys = np.interp(xs, grid, intensity) - floor
        # The wavenumber grid is uniform, so a change of variable is a factor.
        step = float(abs(wavenumber[1] - wavenumber[0]))
        areas[k] = float(np.trapezoid(np.clip(ys, 0.0, None), xs) * step)
    return areas


def peak_table(
    intensity: np.ndarray, wavenumber: np.ndarray,
    config: PeakConfig | None = None,
) -> PeakTable:
    """Conventional peak table of a single spectrum.

    Parameters
    ----------
    intensity
        Normalised absorbance on the common wavenumber grid.
    wavenumber
        The grid, assumed uniform and increasing.
    config
        Detection thresholds; see :class:`PeakConfig`.

    Returns
    -------
    PeakTable
        Position, intensity, prominence, width and area of every detected band,
        sorted by decreasing prominence so that the table can be truncated the
        way a pruned diagram is.
    """
    config = config or PeakConfig()
    intensity = np.asarray(intensity, dtype=float).ravel()
    wavenumber = np.asarray(wavenumber, dtype=float).ravel()

    indices, _ = find_peaks(
        intensity,
        height=config.height,
        prominence=config.prominence,
        distance=config.distance,
        width=config.width,
    )
    if indices.size == 0:
        empty = np.empty(0)
        return PeakTable(empty, empty, empty, empty, empty)

    prominences, left_bases, right_bases = peak_prominences(intensity, indices)

    widths, _, _, _ = peak_widths(
        intensity, indices, rel_height=config.rel_height,
        prominence_data=(prominences, left_bases, right_bases),
    )
    step = float(abs(wavenumber[1] - wavenumber[0]))

    # The area is taken above the prominence base over a window of one full
    # width either side of the maximum. Integrating out to the prominence base
    # itself, as peak_widths(rel_height=1) would place it, is not usable here:
    # the base of the tallest band of a spectrum is the global minimum, so its
    # window spans the whole measurement and its "area" becomes the area of
    # every other band as well. A window tied to the band's own width keeps the
    # attribute local while still capturing 98% of a Gaussian profile.
    half_window = widths                       # in samples, i.e. one FWHM
    centres = indices.astype(float)
    left = np.clip(centres - half_window, 0.0, len(intensity) - 1.0)
    right = np.clip(centres + half_window, 0.0, len(intensity) - 1.0)
    base = intensity[indices] - prominences
    areas = _integrate_above_base(intensity, wavenumber, left, right, base)

    order = np.argsort(-prominences, kind="stable")
    return PeakTable(
        position=wavenumber[indices][order],
        intensity=intensity[indices][order],
        prominence=prominences[order],
        width=(widths * step)[order],
        area=areas[order],
    )


def attribute_diagram(table: PeakTable, attribute: str) -> np.ndarray:
    """Pair the peak positions with one attribute, as a marked point set.

    The result has the shape :class:`irtda.images.PersistenceImager` expects ---
    horizontal coordinate first, the weighted quantity second --- so a
    conventional descriptor is obtained by passing it through the same imager
    that produces the fingerprint image. ``"position"`` gives every peak the
    same vertical coordinate, which reduces the descriptor to a smoothed map of
    where the bands are, with no attribute attached.
    """
    if attribute not in ATTRIBUTES:
        raise ValueError(f"unknown attribute: {attribute!r}; expected one of {ATTRIBUTES}")
    if len(table) == 0:
        return np.empty((0, 2))
    values = (np.ones(len(table)) if attribute == "position"
              else getattr(table, attribute))
    return np.column_stack([table.position, values])
