"""Zero-dimensional persistent homology of one-dimensional signals.

For a piecewise-linear function sampled on a line, the Vietoris-Rips machinery
used in the reference papers for 2-D point clouds collapses to the much simpler
*sublevel-set* filtration, whose degree-0 persistence diagram is exactly the
"one-to-one local-minimum/local-maximum pairing" described in Frahi et al.
(2020).  It is computed here in ``O(n log n)`` with a union-find structure, so
no external topology library is required.

Applied to an IR spectrum, the diagram of the *superlevel-set* filtration has a
direct chemical reading: every point is one absorption band, its birth is the
band height and its persistence is the band's topological prominence, i.e. how
far the absorbance must drop before the band merges into a stronger neighbour.

Sign convention
---------------
Superlevel-set diagrams are returned **in the units of the spectrum itself**:
a feature is born at ``b = f(nu_max)``, the height of the band, and dies at
``d = f(nu_merge) < b``, the level of the saddle at which it merges into an
older component. Its persistence is therefore

    ell = b - d > 0,

the topological prominence of the band, and the points of the diagram lie
*below* the diagonal ``d = b``. Sublevel-set diagrams follow the opposite
(standard) convention, ``d > b``. Use :func:`persistence_of` rather than a
hand-written difference of columns: it is correct for both filtration kinds.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "persistence_of",
    "sublevel_persistence",
    "superlevel_persistence",
    "spectrum_diagram",
    "prune_diagram",
    "lifetime_diagram",
    "position_lifetime_diagram",
]


def persistence_of(diagram: np.ndarray) -> np.ndarray:
    """Persistence (lifetime) of every point of ``diagram``, always non-negative.

    Sublevel- and superlevel-set diagrams use opposite sign conventions for the
    ordering of ``birth`` and ``death`` (see the module docstring), so the
    lifetime is the absolute value of their difference in both cases. Every
    consumer of a diagram in this package goes through this function so that no
    downstream code has to know which filtration produced it.
    """
    diagram = np.atleast_2d(np.asarray(diagram, dtype=float))
    if diagram.size == 0:
        return np.empty(0)
    return np.abs(diagram[:, 1] - diagram[:, 0])


def sublevel_persistence(
    f: np.ndarray, essential_death: str = "max", return_locations: bool = False
):
    """Degree-0 persistence diagram of the sublevel-set filtration of ``f``.

    Connected components are born at local minima of ``f`` and die when they
    merge, at a local maximum, into an older component (elder rule).

    Parameters
    ----------
    f
        One-dimensional array of function values sampled on consecutive points.
    essential_death
        Death value assigned to the single essential class (the global
        minimum), which never dies. ``"max"`` uses ``f.max()`` so that the
        diagram stays finite and metrisable; ``"inf"`` uses ``numpy.inf``.
    return_locations
        Also return the sample indices at which each feature is born and dies.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_features, 2)`` with ``(birth, death)`` pairs,
        sorted by decreasing persistence. If ``return_locations`` is set, a
        triple ``(diagram, birth_index, death_index)`` is returned instead.
    """
    f = np.asarray(f, dtype=float).ravel()
    n = f.size
    if n == 0:
        empty = np.empty((0, 2))
        return (empty, np.empty(0, int), np.empty(0, int)) if return_locations else empty

    order = np.argsort(f, kind="stable")

    parent = np.full(n, -1, dtype=np.int64)  # union-find parent; -1 = inactive
    birth = np.full(n, np.nan)  # birth value, valid at component roots
    birth_at = np.full(n, -1, dtype=np.int64)  # index of the generating extremum
    active = np.zeros(n, dtype=bool)

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:  # path compression
            parent[i], i = root, parent[i]
        return root

    pairs: list[tuple[float, float]] = []
    born_at: list[int] = []
    died_at: list[int] = []

    for i in order:
        i = int(i)
        active[i] = True
        parent[i] = i
        birth[i] = f[i]
        birth_at[i] = i

        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < n and active[j]]
        roots = {find(j) for j in neighbours}
        if not roots:
            continue

        roots.add(i)
        # Elder rule: the component born earliest (lowest birth value) survives.
        elder = min(roots, key=lambda r: (birth[r], r))
        for r in roots:
            if r == elder:
                continue
            if r != i:  # a genuine component dies here
                pairs.append((birth[r], f[i]))
                born_at.append(int(birth_at[r]))
                died_at.append(i)
            parent[r] = elder

    # The essential class: born at the global minimum, never dies.
    root = find(int(order[0]))
    death = f.max() if essential_death == "max" else np.inf
    pairs.append((birth[root], death))
    born_at.append(int(birth_at[root]))
    died_at.append(int(np.argmax(f)))

    diagram = np.asarray(pairs, dtype=float).reshape(-1, 2)
    rank = np.argsort(-(diagram[:, 1] - diagram[:, 0]), kind="stable")
    diagram = diagram[rank]

    if not return_locations:
        return diagram
    return diagram, np.asarray(born_at)[rank], np.asarray(died_at)[rank]


def superlevel_persistence(
    f: np.ndarray, essential_death: str = "max", return_locations: bool = False
):
    """Degree-0 diagram of the superlevel-set filtration, in the units of ``f``.

    Computed as ``sublevel_persistence(-f)`` and mapped back by negation, so
    that the diagram is expressed in the units of the spectrum rather than in
    ``-f`` coordinates. Components are born at local *maxima* (spectral bands)
    and die at local minima, hence

        ``birth = f(nu_max)``   -- the height of the band,
        ``death = f(nu_merge)`` -- the level of the merging saddle, ``< birth``,
        ``persistence = birth - death > 0`` -- its topological prominence.

    The essential class, generated by the global maximum, is assigned
    ``death = min f``. Points of the returned diagram lie below the diagonal.
    """
    out = sublevel_persistence(
        -np.asarray(f, dtype=float), essential_death, return_locations
    )
    if not return_locations:
        return -out
    diagram, born_at, died_at = out
    return -diagram, born_at, died_at


def spectrum_diagram(
    intensity: np.ndarray, kind: str = "peaks", return_locations: bool = False
):
    """Persistence diagram of a single spectrum.

    Parameters
    ----------
    intensity
        Normalised absorbance values on the common wavenumber grid.
    kind
        ``"peaks"``   -- superlevel-set filtration; features are absorption
                         bands and persistence is band prominence (default);
        ``"valleys"`` -- sublevel-set filtration; features are the troughs
                         between bands.
    return_locations
        Also return the grid indices where each feature is born and dies.
    """
    if kind == "peaks":
        return superlevel_persistence(intensity, return_locations=return_locations)
    if kind == "valleys":
        return sublevel_persistence(intensity, return_locations=return_locations)
    raise ValueError(f"unknown diagram kind: {kind!r}")


def prune_diagram(
    diagram: np.ndarray,
    min_persistence: float = 0.0,
    max_features: int | None = None,
    birth_index: np.ndarray | None = None,
):
    """Discard low-persistence features, which encode instrumental noise.

    ``diagram`` is assumed to be sorted by decreasing persistence, as produced
    by :func:`sublevel_persistence`. When ``birth_index`` is supplied it is
    filtered consistently and returned alongside the pruned diagram.
    """
    if diagram.size == 0:
        return diagram if birth_index is None else (diagram, birth_index)

    keep = persistence_of(diagram) > min_persistence
    if max_features is not None:
        cut = np.flatnonzero(keep)[:max_features]
        keep = np.zeros_like(keep)
        keep[cut] = True

    if birth_index is None:
        return diagram[keep]
    return diagram[keep], birth_index[keep]


def lifetime_diagram(diagram: np.ndarray) -> np.ndarray:
    """Map ``(birth, death)`` to the lifetime coordinates ``(birth, lifetime)``.

    This is the bijection ``T`` of Frahi et al. (2020, Eq. 5), written here in
    the sign convention of each filtration: ``T(b, d) = (b, b - d)`` for a
    superlevel-set diagram, whose lifetime is the prominence of the band, and
    ``T(b, d) = (b, d - b)`` for a sublevel-set one. In both cases the second
    coordinate is non-negative.
    """
    if diagram.size == 0:
        return diagram.reshape(-1, 2)
    return np.column_stack([diagram[:, 0], persistence_of(diagram)])


def position_lifetime_diagram(
    diagram: np.ndarray, birth_index: np.ndarray, wavenumber: np.ndarray
) -> np.ndarray:
    """Replace the birth coordinate by the wavenumber at which the feature is born.

    The persistence diagram of a one-dimensional filtration is invariant under
    any reparametrisation of the domain, so it deliberately forgets *where*
    along the spectrum each band lies. That invariance is what makes the
    descriptor powerful for rough surfaces or robot trajectories, but for
    vibrational spectroscopy the position of a band is precisely its chemical
    identity: 1730 cm^-1 is an ester carbonyl whatever its prominence.

    Substituting the wavenumber of the generating extremum for the birth value
    keeps the topological notion of prominence -- which is what makes the
    descriptor robust to baseline drift and to peak misalignment -- while
    restoring the chemical information. The resulting point set is fed to the
    same weighted Gaussian vectorisation as an ordinary lifetime diagram.
    """
    if diagram.size == 0:
        return diagram.reshape(-1, 2)
    lifetimes = persistence_of(diagram)
    return np.column_stack([np.asarray(wavenumber)[birth_index], lifetimes])
