"""Chemometric pre-processing and alignment, as baselines for the comparison.

The introduction of this work argues that a point-by-point comparison of
absorbance values is fragile once spectra are pooled across instruments or
calibrations, and Sect. 5 measures that fragility. The argument is only honest
if the raw spectra are given the treatment they would actually receive: the
pre-processing chain of \\citet{rinnan2009review}, and, when the artefact is a
misalignment, an alignment step.

This module supplies both. The pre-processing routines are the standard ones --
a Savitzky-Golay derivative, an asymmetric-least-squares baseline, extended
multiplicative scatter correction. The alignment routines are the two that a
spectroscopist would reach for against a wavenumber miscalibration: a single
shift estimated by cross-correlation, and a piecewise version of the same idea
in the spirit of interval-correlation-optimised shifting.

Alignment is deliberately given the best case available to it. In the retrieval
experiment each query is aligned *to the library spectrum it is being compared
with*, one pair at a time, rather than to a single global reference: no analyst
has that much information, so a raw-spectrum baseline that still loses under
this treatment has lost on the merits.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.signal import correlate, savgol_filter
from scipy.sparse.linalg import spsolve

__all__ = [
    "savitzky_golay",
    "als_baseline",
    "emsc",
    "align_global",
    "align_intervals",
    "fourier_magnitude",
    "CHAINS",
]


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------
def savitzky_golay(X: np.ndarray, window: int = 15, poly: int = 2,
                   derivative: int = 1) -> np.ndarray:
    """Savitzky-Golay smoothing derivative, row-wise.

    The first derivative removes an additive constant and the second an
    additive slope, which is why derivative filtering is the usual answer to a
    drifting baseline.
    """
    return savgol_filter(np.atleast_2d(X), window_length=window, polyorder=poly,
                         deriv=derivative, axis=1)


def _als_one(y: np.ndarray, lam: float, p: float, iterations: int) -> np.ndarray:
    """Asymmetric least squares baseline of one signal (Eilers and Boelens).

    Minimises ``sum w_i (y_i - z_i)^2 + lam * ||D2 z||^2`` with weights that are
    ``p`` above the current estimate and ``1 - p`` below it, so the fit is
    pulled towards the lower envelope of the signal. The normal equations are
    pentadiagonal; they are assembled as a sparse matrix and solved directly,
    which is O(n) per iteration and costs nothing to get right.
    """
    n = len(y)
    differences = sparse.diags(
        [1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    penalty = lam * (differences.T @ differences)

    w = np.ones(n)
    z = y.copy()
    for _ in range(iterations):
        system = penalty + sparse.diags(w, 0, format="csc")
        z = spsolve(system, w * y)
        w = np.where(y > z, p, 1.0 - p)
    return z


def als_baseline(X: np.ndarray, lam: float = 1e5, p: float = 0.01,
                 iterations: int = 10) -> np.ndarray:
    """Remove an asymmetric-least-squares baseline from every spectrum."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.vstack([row - _als_one(row, lam, p, iterations) for row in X])


def emsc(X: np.ndarray, wavenumber: np.ndarray, degree: int = 2) -> np.ndarray:
    """Extended multiplicative scatter correction against the mean spectrum.

    Each spectrum is regressed on the mean spectrum and a polynomial in the
    wavenumber; the polynomial part is subtracted and the multiplicative
    coefficient divided out, which removes the scattering and path-length
    effects that an ATR contact-pressure variation produces.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    reference = X.mean(axis=0)
    t = np.linspace(-1.0, 1.0, X.shape[1])
    basis = np.vstack([t**k for k in range(degree + 1)] + [reference]).T

    corrected = np.empty_like(X)
    for i, row in enumerate(X):
        coefficients, *_ = np.linalg.lstsq(basis, row, rcond=None)
        scale = coefficients[-1]
        polynomial = basis[:, :-1] @ coefficients[:-1]
        corrected[i] = (row - polynomial) / (scale if abs(scale) > 1e-12 else 1.0)
    return corrected


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def _best_shift(query: np.ndarray, reference: np.ndarray, max_shift: int) -> int:
    """Integer shift of ``query`` that maximises its correlation with ``reference``.

    Evaluated by FFT cross-correlation rather than by trying each shift in turn:
    the retrieval experiment aligns every query against every candidate, so this
    is called of the order of ten thousand times per artefact level.
    """
    q = np.asarray(query, dtype=float) - np.mean(query)
    r = np.asarray(reference, dtype=float) - np.mean(reference)
    correlation = correlate(r, q, mode="full", method="fft")
    centre = len(q) - 1
    lo = max(0, centre - max_shift)
    hi = min(len(correlation), centre + max_shift + 1)
    return int(np.argmax(correlation[lo:hi]) + lo - centre)


def align_global(query: np.ndarray, reference: np.ndarray,
                 max_shift: int = 40) -> np.ndarray:
    """Shift ``query`` rigidly onto ``reference`` by cross-correlation."""
    shift = _best_shift(query, reference, max_shift)
    out = np.roll(query, shift)
    if shift > 0:
        out[:shift] = query[0]
    elif shift < 0:
        out[shift:] = query[-1]
    return out


def align_intervals(query: np.ndarray, reference: np.ndarray,
                    n_intervals: int = 20, max_shift: int = 40) -> np.ndarray:
    """Piecewise alignment: an independent shift per interval, then stitched.

    This is the idea behind interval-correlation-optimised shifting: a single
    rigid shift cannot follow a miscalibration that varies along the axis, so
    the axis is cut into intervals and each is shifted on its own.
    """
    n = len(query)
    edges = np.linspace(0, n, n_intervals + 1).astype(int)
    out = query.copy()
    for lo, hi in zip(edges[:-1], edges[1:]):
        pad = min(max_shift, lo, n - hi)
        window = slice(lo - pad, hi + pad)
        shift = _best_shift(query[window], reference[window], max_shift)
        segment = np.roll(query[window], shift)
        out[lo:hi] = segment[pad:pad + (hi - lo)] if pad else segment
    return out


def fourier_magnitude(X: np.ndarray, n_components: int = 400) -> np.ndarray:
    """Magnitude spectrum along the wavenumber axis, truncated.

    The modulus of the Fourier transform is invariant under a rigid shift of the
    axis by construction, so it is the strongest non-topological answer to a
    wavenumber miscalibration and belongs in the comparison of Sect. 5.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.abs(np.fft.rfft(X, axis=1))[:, :n_components]


#: Pre-processing chains compared against the topological descriptors. Each maps
#: ``(intensity, wavenumber)`` to a feature matrix.
CHAINS = {
    "SNV": lambda X, w: _snv(X),
    "SG 1st derivative": lambda X, w: _snv(savitzky_golay(X, derivative=1)),
    "SG 2nd derivative": lambda X, w: _snv(savitzky_golay(X, derivative=2)),
    "ALS baseline + SNV": lambda X, w: _snv(als_baseline(X)),
    "EMSC": lambda X, w: emsc(X, w),
    "Fourier magnitude": lambda X, w: fourier_magnitude(X),
}


def _snv(X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=float))
    centred = X - X.mean(axis=1, keepdims=True)
    scale = centred.std(axis=1, keepdims=True)
    return centred / np.where(scale > 0, scale, 1.0)
