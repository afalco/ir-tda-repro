#!/usr/bin/env python3
"""Step 9 -- is the topological fingerprint more than a peak table?

Referee 3 put the question that this script answers. The fingerprint image
attaches a persistence to the wavenumber of every absorption band, and a
spectroscopist would obtain something of the same shape by running a peak
picker and tabulating position, intensity, width, area and prominence. Adding
more regression models cannot settle whether persistent homology contributes
anything beyond that table; only a comparison at the level of the *features*
can, and it has to hold everything else fixed.

It is held fixed here. Every conventional descriptor is a marked point set
``(position, attribute)`` pushed through the same imager, with the same
anisotropic kernel, the same resolution and the same ramp weighting as the
fingerprint image, so the only thing that varies is where the marked points
come from and what is attached to them. The same three tasks as in the rest of
the paper are then run on all of them: prediction of the thermal targets,
recovery of the material families, and self-retrieval under measurement
artefacts.

Five things are measured:

    A  agreement    how far the topological feature set and the peak table
                    actually differ, feature by feature, at matched thresholds.
    B  regression   cross-validated Q^2 on the thermogravimetric targets.
    C  clustering   ARI and AMI against the material families.
    D  robustness   self-retrieval under the four artefacts of Sect. 5.
    E  sensitivity  the spread of B and C over the detection thresholds that
                    the peak picker requires and the topological route does not.

Writes:
    results/peak_agreement.csv
    results/peak_regression.csv
    results/peak_clustering.csv
    results/peak_robustness.csv
    results/peak_sensitivity.csv
    figures/09_peak_features.png
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irtda import clustering, dataset, features, images, peaks  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# The regression and robustness protocols are the ones already used in the
# paper; they are imported rather than reimplemented so that the numbers here
# are comparable with Tables 2 and 4 term by term.
sys.path.insert(0, str(ROOT / "scripts"))
_reg = __import__("07_property_regression")
_rob = __import__("04_robustness")

TARGETS = ["T5", "T10", "T50", "T_dtg_peak", "residue"]
_ARTEFACT_NAMES = ["baseline drift", "wavenumber shift",
                   "additive noise", "intensity envelope"]

#: Conventional descriptors, in reporting order. The label is what appears in
#: the tables of the manuscript.
PEAK_DESCRIPTORS = {
    "peaks: position": "position",
    "peaks: + intensity": "intensity",
    "peaks: + width": "width",
    "peaks: + area": "area",
    "peaks: + prominence": "prominence",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=3,
                   help="random draws averaged per artefact level")
    p.add_argument("--parts", default="ABCDE",
                   help="subset of the five analyses to run")
    p.add_argument("--targets", default=",".join(TARGETS),
                   help="comma-separated regression targets for part B")
    p.add_argument("--artefacts", default=",".join(_ARTEFACT_NAMES),
                   help="comma-separated artefacts for part D")
    return p.parse_args()


def _append(frame: pd.DataFrame, path: Path) -> None:
    """Add rows to a results file, replacing any earlier run of the same keys.

    Parts B and D are expensive enough to be run one target or one artefact at
    a time, so each invocation contributes its rows to the same table rather
    than overwriting it.
    """
    if path.exists():
        previous = pd.read_csv(path)
        keys = [c for c in ("target", "artefact", "representation", "model", "level")
                if c in frame.columns and c in previous.columns]
        merged = pd.concat([previous, frame], ignore_index=True)
        frame = merged.drop_duplicates(subset=keys, keep="last")
    frame.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------
def peak_tables(intensity, wavenumber, config):
    return [peaks.peak_table(row, wavenumber, config) for row in np.atleast_2d(intensity)]


def fit_peak_imagers(tables, config):
    """One imager per attribute, fitted on the reference (clean) tables.

    The kernel matches the fingerprint image: one band width along wavenumber
    and one pixel along the attribute axis. As in the topological pipeline the
    grids are fitted once and reused for every perturbed copy, otherwise the
    resulting vectors would not be comparable across perturbations.
    """
    imagers = {}
    for label, attribute in PEAK_DESCRIPTORS.items():
        diagrams = [peaks.attribute_diagram(t, attribute) for t in tables]
        probe = images.PersistenceImager(resolution=config.resolution)
        probe.fit(diagrams)
        pixel = probe.lifetime_range[1] / config.resolution
        imager = images.PersistenceImager(
            resolution=config.resolution,
            sigma=(config.sigma_wavenumber, max(pixel, 1e-6)),
        )
        imager.fit(diagrams)
        imagers[label] = imager
    return imagers


def peak_images(tables, imagers):
    out = {}
    for label, attribute in PEAK_DESCRIPTORS.items():
        diagrams = [peaks.attribute_diagram(t, attribute) for t in tables]
        out[label] = imagers[label].transform(diagrams)
    return out


def _unit_frobenius(block: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(block)
    return block / norm if norm > 0 else block


def combine(blocks: list[np.ndarray]) -> np.ndarray:
    """Concatenate descriptor blocks, each scaled so none dominates by scale.

    The same normalisation is used for the multimodal descriptors of Sect. 7.
    """
    return np.hstack([_unit_frobenius(b) for b in blocks])


# ---------------------------------------------------------------------------
# A -- how far do the two feature sets actually differ?
# ---------------------------------------------------------------------------
def agreement(intensity, wavenumber, descriptors, config) -> pd.DataFrame:
    """Match the topological features against the peak table, one spectrum at a time.

    Both routes mark maxima of the same sampled function, so a feature and a
    peak correspond exactly when they sit on the same grid point; the match is
    made on the position and is therefore exact rather than approximate.
    """
    rows = []
    step = float(abs(wavenumber[1] - wavenumber[0]))
    for i, spectrum in enumerate(np.atleast_2d(intensity)):
        fingerprint = descriptors.fingerprints[i]
        table = peaks.peak_table(spectrum, wavenumber, peaks.PeakConfig(
            prominence=config.min_persistence))
        essential = int(np.argmax(fingerprint[:, 1])) if len(fingerprint) else -1

        for k, (nu, lifetime) in enumerate(fingerprint):
            if len(table) == 0:
                rows.append(dict(spectrum=i, wavenumber=float(nu),
                                 persistence=float(lifetime), prominence=np.nan,
                                 matched=False, essential=k == essential))
                continue
            j = int(np.argmin(np.abs(table.position - nu)))
            hit = abs(table.position[j] - nu) <= 0.5 * step
            rows.append(dict(
                spectrum=i, wavenumber=float(nu), persistence=float(lifetime),
                prominence=float(table.prominence[j]) if hit else np.nan,
                matched=bool(hit), essential=k == essential,
                edge=bool(nu < wavenumber[0] + 20 or nu > wavenumber[-1] - 20),
            ))
        rows.append(dict(spectrum=i, n_topological=len(fingerprint),
                         n_peaks=len(table), summary=True))
    return pd.DataFrame(rows)


def report_agreement(frame: pd.DataFrame) -> None:
    summary = frame[frame.get("summary").fillna(False)] if "summary" in frame else frame
    detail = frame[frame["persistence"].notna()]
    matched = detail[detail["matched"]]
    interior = matched[~matched["essential"] & ~matched["edge"].fillna(False)]
    delta = (interior["persistence"] - interior["prominence"]).abs()

    print(f"  topological features per spectrum : {summary['n_topological'].mean():.1f}")
    print(f"  peaks per spectrum                : {summary['n_peaks'].mean():.1f}")
    print(f"  features matched to a peak        : {100 * matched.shape[0] / detail.shape[0]:.1f}%")
    print(f"  interior features, |persistence - prominence|:")
    print(f"      exactly equal   {100 * (delta < 1e-9).mean():.2f}%  of {len(delta)}")
    print(f"      median          {delta.median():.2e}")
    print(f"      maximum         {delta.max():.2e}")


# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(PROCESSED / "labels.csv")
    targets = pd.read_csv(PROCESSED / "targets.csv")
    data = np.load(PROCESSED / "spectra.npz", allow_pickle=True)
    wavenumber, raw = data["wavenumber"], data["intensity"]
    families = labels["family"].to_numpy()

    config = features.DescriptorConfig(adaptive_threshold=False)
    clean = dataset.normalise(raw, "minmax")
    descriptors = features.compute_diagrams(clean, wavenumber, config)
    imager, fingerprint_imager = features.build_imagers(descriptors, config)

    tables = peak_tables(clean, wavenumber, peaks.PeakConfig(
        prominence=config.min_persistence))
    peak_imagers = fit_peak_imagers(tables, config)

    representations: dict[str, np.ndarray] = {
        "TFI": fingerprint_imager.transform(descriptors.fingerprints),
        "PI": imager.transform(descriptors.lifetimes),
        **peak_images(tables, peak_imagers),
    }
    representations["peaks: all five"] = combine(
        [representations[label] for label in PEAK_DESCRIPTORS])
    representations["raw spectra"] = dataset.normalise(raw, "snv")

    # -- A ------------------------------------------------------------------
    if "A" in args.parts:
        print("\nA. agreement between the two feature sets")
        frame = agreement(clean, wavenumber, descriptors, config)
        frame.to_csv(RESULTS / "peak_agreement.csv", index=False)
        report_agreement(frame)

    # -- B ------------------------------------------------------------------
    if "B" in args.parts:
        print("\nB. prediction of the thermogravimetric targets")
        rows, predictions = [], {}
        for target in args.targets.split(","):
            y_all = targets[target].to_numpy(dtype=float)
            keep = np.isfinite(y_all)
            y = y_all[keep]
            y_hat = _reg.family_mean_predict(families[keep], y)
            rows.append(dict(target=target, representation="family mean",
                             model="-", **_reg.scores(y, y_hat)))
            for name, X in representations.items():
                for model, fn in (("ridge", _reg.ridge_predict),
                                  ("PLS", _reg.pls_predict)):
                    y_hat = fn(X[keep], y)
                    predictions[(target, name, model)] = (y, y_hat)
                    rows.append(dict(target=target, representation=name,
                                     model=model, **_reg.scores(y, y_hat)))
            best = max((r for r in rows if r["target"] == target
                        and r["model"] != "-"), key=lambda r: r["Q2"])
            print(f"  {target:11s} best {best['representation']:20s} "
                  f"{best['model']:5s} Q2={best['Q2']:6.3f}")
        _append(pd.DataFrame(rows), RESULTS / "peak_regression.csv")

        # Paired bootstrap of the fingerprint image against every control, on
        # the target for which the manuscript claims an advantage.
        comparisons = []
        for target in [t for t in args.targets.split(",") if t == "T5"]:
            for model in ("ridge",):
                y, tfi_hat = predictions[(target, "TFI", model)]
                for name in representations:
                    if name == "TFI":
                        continue
                    _, other = predictions[(target, name, model)]
                    lo, hi, above = _reg.bootstrap_delta_q2(y, tfi_hat, other)
                    comparisons.append(dict(
                        target=target, model=model, against=name,
                        delta_Q2=float(_reg.scores(y, tfi_hat)["Q2"]
                                       - _reg.scores(y, other)["Q2"]),
                        lo=lo, hi=hi, p_above_zero=above))
        if comparisons:
            _append(pd.DataFrame(comparisons),
                    RESULTS / "peak_regression_bootstrap.csv")
        if comparisons:
            print("\n  paired bootstrap, TFI against each control on T5 (ridge):")
        for c in comparisons:
            print(f"    vs {c['against']:22s} dQ2={c['delta_Q2']:+.3f}  "
                  f"[{c['lo']:+.3f}, {c['hi']:+.3f}]  P(>0)={c['p_above_zero']:.3f}")

    # -- C ------------------------------------------------------------------
    if "C" in args.parts:
        print("\nC. recovery of the material families")
        rows = []
        main_subset = np.isin(families, ["TPU", "PUR", "TR"])
        for subset, mask, k in (("all (n=39)", np.ones(len(families), bool), 6),
                                ("TPU/PUR/TR (n=35)", main_subset, 3)):
            for name, X in representations.items():
                predicted = clustering.kmeans_labels(X[mask], k, seed=args.seed)
                rows.append(dict(subset=subset, representation=name,
                                 **clustering.evaluate(families[mask], predicted, X[mask])))
        table = pd.DataFrame(rows)
        table.to_csv(RESULTS / "peak_clustering.csv", index=False)
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # -- D ------------------------------------------------------------------
    if "D" in args.parts:
        print("\nD. self-retrieval under measurement artefacts")
        rng = np.random.default_rng(args.seed)
        reference = {name: X for name, X in representations.items()}
        rows = []
        for artefact in args.artefacts.split(","):
            meta = _rob.ARTEFACTS[artefact]
            for level in _rob.LEVELS:
                amplitude = level * meta["scale"]
                scores = {name: [] for name in representations}
                for _ in range(args.repeats):
                    perturbed = _rob.perturb(raw, wavenumber, artefact, amplitude, rng)
                    normalised = dataset.normalise(perturbed, "minmax")
                    d = features.compute_diagrams(normalised, wavenumber, config)
                    t = peak_tables(normalised, wavenumber, peaks.PeakConfig(
                        prominence=config.min_persistence))
                    query = {
                        "TFI": fingerprint_imager.transform(d.fingerprints),
                        "PI": imager.transform(d.lifetimes),
                        **peak_images(t, peak_imagers),
                        "raw spectra": dataset.normalise(perturbed, "snv"),
                    }
                    query["peaks: all five"] = combine(
                        [query[label] for label in PEAK_DESCRIPTORS])
                    for name in representations:
                        scores[name].append(
                            _rob.self_retrieval(reference[name], query[name]))
                for name, values in scores.items():
                    rows.append(dict(artefact=artefact, level=float(level),
                                     amplitude=float(amplitude),
                                     representation=name,
                                     retrieval=float(np.mean(values)),
                                     sd=float(np.std(values))))
                print(f"  {artefact:19s} level {level:.2f}  " + "  ".join(
                    f"{n.split(':')[-1].strip()[:9]}={np.mean(v):.2f}"
                    for n, v in scores.items()))
        _append(pd.DataFrame(rows), RESULTS / "peak_robustness.csv")

    # -- E ------------------------------------------------------------------
    if "E" in args.parts:
        print("\nE. sensitivity of the peak picker to its thresholds")
        y_all = targets["T5"].to_numpy(dtype=float)
        keep = np.isfinite(y_all)
        y = y_all[keep]
        main_subset = np.isin(families, ["TPU", "PUR", "TR"])
        rows = []
        for prominence in (1e-3, 2e-3, 5e-3, 1e-2, 2e-2):
            for distance in (None, 3, 10, 30):
                cfg = peaks.PeakConfig(prominence=prominence, distance=distance)
                t = peak_tables(clean, wavenumber, cfg)
                diagrams = [peaks.attribute_diagram(x, "prominence") for x in t]
                probe = images.PersistenceImager(resolution=config.resolution)
                probe.fit(diagrams)
                pixel = probe.lifetime_range[1] / config.resolution
                im = images.PersistenceImager(
                    resolution=config.resolution,
                    sigma=(config.sigma_wavenumber, max(pixel, 1e-6)))
                im.fit(diagrams)
                X = im.transform(diagrams)
                q2 = _reg.scores(y, _reg.ridge_predict(X[keep], y))["Q2"]
                ari = clustering.evaluate(
                    families[main_subset],
                    clustering.kmeans_labels(X[main_subset], 3, seed=args.seed),
                    X[main_subset])["ARI"]
                rows.append(dict(prominence=prominence,
                                 distance=-1 if distance is None else distance,
                                 n_peaks=float(np.mean([len(x) for x in t])),
                                 Q2_T5=q2, ARI=ari))
                print(f"  prominence={prominence:<7g} distance={str(distance):<5s} "
                      f"peaks={rows[-1]['n_peaks']:5.1f}  Q2={q2:6.3f}  ARI={ari:.3f}")
        frame = pd.DataFrame(rows)
        frame.to_csv(RESULTS / "peak_sensitivity.csv", index=False)
        print(f"\n  peak picker : Q2 in [{frame.Q2_T5.min():.3f}, {frame.Q2_T5.max():.3f}], "
              f"ARI in [{frame.ARI.min():.3f}, {frame.ARI.max():.3f}] over "
              f"{len(frame)} threshold settings")

    print(f"\nwritten to {RESULTS}")


if __name__ == "__main__":
    main()
