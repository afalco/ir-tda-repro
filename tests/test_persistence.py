"""Correctness checks for the persistence computation and its vectorisation."""

from pathlib import Path

import numpy as np
import pytest

from irtda import clustering, features, images, persistence


def test_single_peak_has_one_finite_class():
    """A unimodal signal has exactly one connected component throughout."""
    x = np.linspace(-3, 3, 201)
    f = np.exp(-(x**2))
    diagram = persistence.superlevel_persistence(f)
    assert len(diagram) == 1
    # The essential class spans the full range of the signal.
    assert persistence.persistence_of(diagram)[0] == pytest.approx(f.max() - f.min())


def test_two_peaks_pair_by_prominence():
    """Two peaks separated by a valley give one finite class of known prominence."""
    #        peak 1      valley     peak 2
    f = np.array([0.0, 1.0, 0.4, 0.8, 0.0])
    diagram = persistence.superlevel_persistence(f)
    lifetimes = np.sort(persistence.persistence_of(diagram))
    # The lower peak (0.8) dies at the valley (0.4): prominence 0.4.
    # The higher peak (1.0) is essential: prominence 1.0.
    assert lifetimes == pytest.approx([0.4, 1.0])


def test_number_of_classes_matches_number_of_local_maxima():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 8 * np.pi, 2000)
    f = np.sin(x) + 0.3 * np.sin(3.7 * x) + 0.01 * rng.normal(size=x.size)
    diagram = persistence.superlevel_persistence(f)
    interior = f[1:-1]
    n_maxima = int(((interior > f[:-2]) & (interior > f[2:])).sum())
    # Every local maximum generates exactly one class (boundary maxima included).
    assert n_maxima <= len(diagram) <= n_maxima + 2


def test_diagram_is_invariant_under_reparametrisation():
    """Persistence of a 1-D signal does not depend on the sampling positions."""
    x = np.linspace(0, 10, 500)
    f = np.sin(x) + 0.5 * np.sin(2.3 * x)
    stretched = np.interp(np.linspace(0, 10, 500), x**1.2 / 10**0.2, f)

    a = persistence.superlevel_persistence(f)
    b = persistence.superlevel_persistence(stretched)
    top_a = np.sort(persistence.persistence_of(a))[-3:]
    top_b = np.sort(persistence.persistence_of(b))[-3:]
    assert top_a == pytest.approx(top_b, abs=0.05)


def test_pruning_keeps_the_most_persistent_features():
    f = np.array([0.0, 1.0, 0.4, 0.8, 0.0, 0.81, 0.79, 0.9, 0.0])
    diagram = persistence.superlevel_persistence(f)
    pruned = persistence.prune_diagram(diagram, min_persistence=0.05)
    kept = persistence.persistence_of(pruned)
    assert (kept > 0.05).all()
    assert len(pruned) < len(diagram)


def test_birth_locations_point_at_the_maxima():
    f = np.array([0.0, 1.0, 0.4, 0.8, 0.0])
    _, born_at, _ = persistence.superlevel_persistence(f, return_locations=True)
    assert set(born_at.tolist()) == {1, 3}


def test_persistence_image_is_translation_covariant():
    """Shifting a diagram along the birth axis shifts its image, not its mass."""
    diagram = np.array([[0.0, 0.5], [0.1, 0.9]])
    shifted = diagram + 0.2

    # Generous padding so that no Gaussian mass falls outside the grid.
    imager = images.PersistenceImager(resolution=16, sigma=0.02, padding=0.5)
    imager.fit([persistence.lifetime_diagram(d) for d in (diagram, shifted)])
    a = imager.transform_one(persistence.lifetime_diagram(diagram))
    b = imager.transform_one(persistence.lifetime_diagram(shifted))
    assert a.sum() == pytest.approx(b.sum(), rel=1e-6)
    assert not np.allclose(a, b)


def test_persistence_image_mass_matches_total_weight():
    """With a narrow kernel well inside the grid, the image integrates the weights."""
    diagram = np.array([[0.0, 1.0], [0.2, 0.6]])
    lt = persistence.lifetime_diagram(diagram)
    imager = images.PersistenceImager(resolution=64, sigma=0.01, padding=0.3)
    imager.fit([lt])
    expected = (lt[:, 1] / lt[:, 1].max()).sum()
    assert imager.transform_one(lt).sum() == pytest.approx(expected, rel=1e-3)


def test_wasserstein_is_zero_for_identical_diagrams():
    diagram = np.array([[0.0, 1.0], [0.2, 0.6], [0.3, 0.35]])
    assert clustering.wasserstein(diagram, diagram) == pytest.approx(0.0, abs=1e-9)
    assert clustering.sliced_wasserstein(diagram, diagram) == pytest.approx(0.0, abs=1e-9)


def test_wasserstein_matches_a_hand_computed_case():
    """One point moved by a known amount, everything else identical."""
    a = np.array([[0.0, 1.0]])
    b = np.array([[0.0, 1.3]])
    assert clustering.wasserstein(a, b, order=2) == pytest.approx(0.3, rel=1e-6)


def test_distance_matrix_is_a_metric_shape():
    rng = np.random.default_rng(1)
    diagrams = [np.sort(rng.random((5, 2)), axis=1) for _ in range(4)]
    d = clustering.distance_matrix(diagrams)
    assert d.shape == (4, 4)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)
    # Triangle inequality on every triple.
    for i in range(4):
        for j in range(4):
            for k in range(4):
                assert d[i, j] <= d[i, k] + d[k, j] + 1e-9


def test_noise_estimate_recovers_the_injected_level():
    rng = np.random.default_rng(2)
    x = np.linspace(0, 20, 5000)
    clean = np.sin(x)
    sigma = 0.02
    noisy = clean + sigma * rng.normal(size=x.size)
    assert features.estimate_noise(noisy) == pytest.approx(sigma, rel=0.1)


def test_multi_config_matches_single_config():
    rng = np.random.default_rng(3)
    wavenumber = np.linspace(500, 4000, 800)
    spectra = np.abs(rng.normal(size=(3, 800)).cumsum(axis=1))
    spectra /= spectra.max(axis=1, keepdims=True)

    a = features.DescriptorConfig(min_persistence=1e-3, adaptive_threshold=False)
    b = features.DescriptorConfig(min_persistence=1e-2, adaptive_threshold=False)
    together = features.compute_diagrams_multi(spectra, wavenumber, [a, b])
    for config, expected in zip((a, b), together):
        got = features.compute_diagrams(spectra, wavenumber, config)
        for x, y in zip(got.diagrams, expected.diagrams):
            assert np.allclose(x, y)


def test_superlevel_diagram_is_expressed_in_spectrum_units():
    """Birth is the band height and death is the merging level, with d < b.

    This pins the sign convention stated in Eqs. (1)-(2) of the paper: the
    superlevel-set diagram lives in the units of ``f`` itself, so its points lie
    strictly below the diagonal and the lifetime is ``ell = b - d > 0``.
    """
    #             peak 1      valley     peak 2
    f = np.array([0.0, 1.0, 0.4, 0.8, 0.0])
    diagram = persistence.superlevel_persistence(f)

    births, deaths = diagram[:, 0], diagram[:, 1]
    assert (deaths < births).all()                      # strictly below diagonal
    assert set(np.round(births, 12)) == {1.0, 0.8}      # births are band heights
    # The lower band (height 0.8) dies at the valley; the essential one at min f.
    assert dict(zip(np.round(births, 12), np.round(deaths, 12))) == {0.8: 0.4, 1.0: 0.0}
    assert persistence.persistence_of(diagram) == pytest.approx([1.0, 0.4])


def test_superlevel_and_sublevel_use_opposite_conventions():
    """``superlevel_persistence(f)`` is the reflection of ``sublevel_persistence(-f)``."""
    rng = np.random.default_rng(7)
    f = np.abs(rng.normal(size=400).cumsum())
    f /= f.max()

    sup = persistence.superlevel_persistence(f)
    sub = persistence.sublevel_persistence(-f)

    assert np.allclose(sup, -sub)
    assert (sup[:, 1] < sup[:, 0]).all()      # superlevel: d < b
    assert (sub[:, 1] > sub[:, 0]).all()      # sublevel:   d > b
    assert np.allclose(persistence.persistence_of(sup), persistence.persistence_of(sub))


def test_lifetime_is_non_negative_for_both_filtrations():
    x = np.linspace(0, 12, 900)
    f = np.sin(x) + 0.4 * np.sin(2.7 * x)

    for diagram in (persistence.superlevel_persistence(f),
                    persistence.sublevel_persistence(f)):
        lt = persistence.lifetime_diagram(diagram)
        assert (lt[:, 1] >= 0).all()
        assert np.allclose(lt[:, 0], diagram[:, 0])


def test_fingerprint_lifetimes_are_positive_prominences():
    wavenumber = np.linspace(500, 4000, 500)
    f = np.zeros(500)
    for centre, height in ((1730.0, 1.0), (2900.0, 0.6), (1100.0, 0.35)):
        f += height * np.exp(-((wavenumber - centre) ** 2) / (2 * 20.0**2))
    f /= f.max()

    diagram, born_at, _ = persistence.superlevel_persistence(f, return_locations=True)
    tfi = persistence.position_lifetime_diagram(diagram, born_at, wavenumber)

    assert (tfi[:, 1] > 0).all()
    assert tfi[:, 0].min() >= wavenumber.min() and tfi[:, 0].max() <= wavenumber.max()
    # The most prominent feature sits at the strongest band.
    assert tfi[np.argmax(tfi[:, 1]), 0] == pytest.approx(1730.0, abs=15.0)


def test_sliced_wasserstein_is_invariant_under_the_sign_convention():
    """Changing convention reflects every diagram, which no distance should see."""
    rng = np.random.default_rng(11)
    f = np.abs(rng.normal(size=600).cumsum()); f /= f.max()
    g = np.abs(rng.normal(size=600).cumsum()); g /= g.max()

    a, b = (persistence.superlevel_persistence(x) for x in (f, g))
    assert clustering.sliced_wasserstein(a, b) == pytest.approx(
        clustering.sliced_wasserstein(-a, -b), rel=1e-9
    )
    assert clustering.wasserstein(a, b) == pytest.approx(
        clustering.wasserstein(-a, -b), rel=1e-9
    )


def test_descriptions_are_translated_into_english():
    """Every Spanish batch description of the study renders in English."""
    from irtda import descriptions

    cases = {
        "5 planchas PUR negro BK sin lavar": "5 black PUR sheets, BK, unwashed",
        "8 pares TR marrón CLO/7922/65": "8 brown TR pairs, CLO/7922/65",
        "4 Planchas de Eva amarillo": "4 yellow EVA sheets",
        "9 muestras PVC negro": "9 black PVC samples",
        "3 planchas caucho marrón LATEX": "3 brown rubber sheets, LATEX",
        "6 planchas EVA negro MII relleno": "6 black EVA sheets, MII, filled",
        # Everything after "marcada como" is a quoted trade name, kept verbatim,
        # except the supplying company's own brand, which names a customer.
        "2 Planchas C/gris marcada como Hi-react PU Pikolinos":
            "2 grey sheets, labelled Hi-react PU",
        "2 Planchas C/crudo marcada como Hi-react PU Pikolinos":
            "2 natural sheets, labelled Hi-react PU",
    }
    for spanish, english in cases.items():
        assert descriptions.translate(spanish) == english


def test_translation_refuses_an_unknown_spanish_word():
    """A description added later cannot reach a figure untranslated."""
    from irtda import descriptions

    with pytest.raises(descriptions.UnknownTerm):
        descriptions.translate("5 planchas TPU verde XYZ")


def test_every_description_in_the_data_set_is_translatable():
    import pandas as pd
    from irtda import descriptions

    path = Path(__file__).resolve().parents[1] / "data/processed/labels.csv"
    if not path.exists():                       # data not extracted yet
        pytest.skip("data/processed/labels.csv not present")
    for text in pd.read_csv(path)["description"]:
        assert descriptions.translate(text)


# ---------------------------------------------------------------------------
# Conventional peak descriptors (the control of Sect. 9)
# ---------------------------------------------------------------------------
def test_peak_table_recovers_gaussian_width_and_area():
    """Width and area are exact on a profile whose values are known."""
    from irtda import peaks

    wavenumber = np.linspace(500, 4000, 3600)
    centres, heights, sigmas = (1730.0, 2900.0, 1100.0), (1.0, 0.6, 0.35), (12.0, 25.0, 40.0)
    f = np.zeros_like(wavenumber)
    for c, h, s in zip(centres, heights, sigmas):
        f += h * np.exp(-((wavenumber - c) ** 2) / (2 * s**2))

    table = peaks.peak_table(f, wavenumber, peaks.PeakConfig(prominence=1e-3))
    assert len(table) == 3

    order = np.argsort(table.position)
    fwhm = 2 * np.sqrt(2 * np.log(2))
    expected_width = fwhm * np.array(sigmas)
    expected_position = np.array(centres)
    # The area is integrated over one full width either side of the maximum,
    # which captures erf(fwhm / sqrt 2) of a Gaussian.
    from scipy.special import erf
    captured = erf(fwhm / np.sqrt(2))
    expected_area = np.array(heights) * np.array(sigmas) * np.sqrt(2 * np.pi) * captured

    by_position = np.argsort(expected_position)
    assert table.position[order] == pytest.approx(expected_position[by_position], abs=1.0)
    assert table.width[order] == pytest.approx(expected_width[by_position], rel=1e-3)
    assert table.area[order] == pytest.approx(expected_area[by_position], rel=1e-2)


def test_prominence_and_persistence_agree_on_an_interior_band():
    """Degree-0 persistence is the topographic prominence the peak picker reports.

    They are the same quantity, which is the point of the control: what the
    topological construction supplies is not a different number but the same
    one, computed exactly and without a detection threshold.
    """
    from irtda import peaks

    wavenumber = np.linspace(0.0, 100.0, 1001)
    #  a dominant band, a lesser one beside it, and a shoulder on the lesser
    f = (1.0 * np.exp(-((wavenumber - 50) ** 2) / 8)
         + 0.6 * np.exp(-((wavenumber - 62) ** 2) / 8)
         + 0.2 * np.exp(-((wavenumber - 70) ** 2) / 4))

    diagram, born_at, _ = persistence.superlevel_persistence(f, return_locations=True)
    topological = persistence.position_lifetime_diagram(diagram, born_at, wavenumber)
    table = peaks.peak_table(f, wavenumber, peaks.PeakConfig(prominence=1e-6))

    for position, lifetime in topological:
        j = int(np.argmin(np.abs(table.position - position)))
        if abs(table.position[j] - position) > 1e-9:
            continue
        # The essential class is the exception: the peak picker bounds its
        # search at the ends of the array, the filtration does not.
        if lifetime == pytest.approx(f.max() - f.min()):
            continue
        assert lifetime == pytest.approx(table.prominence[j], abs=1e-9)


def test_attribute_diagram_has_the_shape_the_imager_consumes():
    from irtda import images, peaks

    wavenumber = np.linspace(500, 4000, 800)
    f = np.exp(-((wavenumber - 1700) ** 2) / 400) + 0.4 * np.exp(-((wavenumber - 2900) ** 2) / 900)
    table = peaks.peak_table(f, wavenumber, peaks.PeakConfig(prominence=1e-3))

    for attribute in peaks.ATTRIBUTES:
        diagram = peaks.attribute_diagram(table, attribute)
        assert diagram.shape == (len(table), 2)
        assert (diagram[:, 1] >= 0).all()
        imager = images.PersistenceImager(resolution=8)
        imager.fit([diagram])
        assert imager.transform_one(diagram).size == 64

    with pytest.raises(ValueError):
        peaks.attribute_diagram(table, "curvature")


# ---------------------------------------------------------------------------
# Pre-processing and alignment (the baselines of Sect. 10)
# ---------------------------------------------------------------------------
def _two_band_spectrum(wavenumber):
    return (np.exp(-((wavenumber - 1700) ** 2) / 800)
            + 0.6 * np.exp(-((wavenumber - 2900) ** 2) / 2000))


def test_als_removes_a_curved_baseline():
    from irtda import preprocess

    wavenumber = np.linspace(500, 4000, 3600)
    clean = _two_band_spectrum(wavenumber)
    drift = 0.3 * ((wavenumber - wavenumber[0]) / (wavenumber[-1] - wavenumber[0])) ** 2

    corrected_clean = preprocess.als_baseline(clean[None, :])[0]
    corrected_drifted = preprocess.als_baseline((clean + drift)[None, :])[0]
    # The correction maps both to the same signal: the drift is gone.
    assert np.abs(corrected_clean - corrected_drifted).max() < 1e-3


def test_cross_correlation_alignment_undoes_a_rigid_shift():
    from irtda import preprocess

    wavenumber = np.linspace(500, 4000, 3600)
    clean = _two_band_spectrum(wavenumber)
    for shift in (4.0, 8.0, 16.0):
        moved = np.interp(wavenumber, wavenumber + shift, clean)
        before = np.abs(moved - clean).max()
        after = np.abs(preprocess.align_global(moved, clean) - clean).max()
        assert after < before / 10.0


def test_fourier_magnitude_is_shift_invariant():
    from irtda import preprocess

    wavenumber = np.linspace(500, 4000, 3600)
    clean = _two_band_spectrum(wavenumber)
    moved = np.interp(wavenumber, wavenumber + 12.0, clean)
    a = preprocess.fourier_magnitude(clean[None, :])
    b = preprocess.fourier_magnitude(moved[None, :])
    assert np.abs(a - b).max() / a.max() < 1e-3


def test_every_chain_returns_one_row_per_spectrum():
    from irtda import preprocess

    wavenumber = np.linspace(500, 4000, 1000)
    X = np.vstack([_two_band_spectrum(wavenumber),
                   _two_band_spectrum(wavenumber) * 0.7 + 0.05])
    for name, chain in preprocess.CHAINS.items():
        out = chain(X, wavenumber)
        assert out.shape[0] == 2, name
        assert np.isfinite(out).all(), name


# ---------------------------------------------------------------------------
# The Vietoris-Rips control of Sect. 13
# ---------------------------------------------------------------------------
def test_rips_point_cloud_respects_the_aspect_ratio():
    from irtda import rips

    wavenumber = np.linspace(500, 4000, 2000)
    intensity = np.exp(-((wavenumber - 1700) ** 2) / 4000)
    for aspect in (0.5, 1.0, 4.0):
        cloud = rips.point_cloud(intensity, wavenumber,
                                 rips.RipsConfig(aspect=aspect, n_points=200))
        assert cloud.shape == (200, 2)
        assert cloud[:, 0].min() == pytest.approx(0.0)
        assert cloud[:, 0].max() == pytest.approx(aspect, rel=1e-3)
        assert 0.0 <= cloud[:, 1].min() and cloud[:, 1].max() <= 1.0


def test_betti_curve_counts_the_features_alive():
    from irtda import rips

    #                     born  dies
    diagram = np.array([[0.0, 2.0],
                        [1.0, 3.0],
                        [1.5, 1.8]])
    grid = np.array([-0.5, 0.5, 1.2, 1.6, 1.9, 2.5, 3.5])
    expected = [0, 1, 2, 3, 2, 1, 0]
    assert list(rips.betti_curve(diagram, grid)) == pytest.approx(expected)
    assert list(rips.betti_curve(np.empty((0, 2)), grid)) == pytest.approx([0] * 7)


def test_rips_diagrams_are_finite_and_h1_depends_on_the_aspect():
    """The dependence on the axis ratio is the point of the comparison."""
    ripser = pytest.importorskip("ripser")

    wavenumber = np.linspace(500, 4000, 1200)
    rng = np.random.default_rng(0)
    intensity = np.zeros_like(wavenumber)
    for centre in (900.0, 1400.0, 1700.0, 2900.0, 3300.0):
        intensity += rng.uniform(0.3, 1.0) * np.exp(
            -((wavenumber - centre) ** 2) / 2000.0)
    intensity /= intensity.max()

    from irtda import rips

    counts = []
    for aspect in (0.2, 5.0):
        diagrams = rips.rips_diagrams(
            intensity, wavenumber, rips.RipsConfig(aspect=aspect, n_points=200))
        assert all(np.isfinite(d).all() for d in diagrams)
        counts.append(len(diagrams[1]))
    assert counts[0] != counts[1]
