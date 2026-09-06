#!/usr/bin/env python3
"""Step 10 -- does pre-processing, or alignment, rescue the raw spectra?

Section 5 reports that a point-by-point comparison identifies 67% of the
compounds under a wavenumber miscalibration where the band-based descriptors
identify essentially all of them. That comparison is only fair if the raw
spectra are given the treatment they would actually receive. Referee 3 asked
for the state-of-the-art spectral-processing methods, and against a wavenumber
artefact the obvious one is not a filter at all but an alignment step.

Two questions are answered.

    B  Do the standard pre-processing chains -- a Savitzky-Golay derivative, an
       asymmetric-least-squares baseline, extended multiplicative scatter
       correction -- change the picture for prediction and identification?

    D  Under each artefact, does aligning the query to the library before
       matching restore the raw spectra?

Alignment is given the best case available to it: each query is aligned to the
library spectrum it is being compared with, one pair at a time, which is more
information than any analyst has. A representation that still loses under that
treatment has lost on the merits. The shift-invariant Fourier magnitude is
included for the same reason: it is the strongest non-topological answer to a
rigid miscalibration.

Writes:
    results/alignment_regression.csv
    results/alignment_clustering.csv
    results/alignment_robustness.csv
    figures/10_alignment.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from irtda import clustering, dataset, features, peaks, preprocess  # noqa: E402
from irtda import images  # noqa: E402

_reg = __import__("07_property_regression")
_rob = __import__("04_robustness")

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

ALIGNERS = {
    "raw, aligned (global)": preprocess.align_global,
    "raw, aligned (intervals)": preprocess.align_intervals,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--parts", default="BD")
    p.add_argument("--targets", default="T5")
    p.add_argument("--artefacts", default="wavenumber shift")
    p.add_argument("--adaptive", action="store_true",
                   help="tie the pruning threshold to the per-spectrum noise "
                        "estimate, as Sect. 5 does for the noise artefact")
    p.add_argument("--suffix", default="",
                   help="label appended to the representation names, so that a "
                        "second setting can be added to the same table")
    return p.parse_args()


def _append(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        previous = pd.read_csv(path)
        keys = [c for c in ("target", "artefact", "representation", "model", "level")
                if c in frame.columns and c in previous.columns]
        frame = pd.concat([previous, frame], ignore_index=True).drop_duplicates(
            subset=keys, keep="last")
    frame.to_csv(path, index=False)


def aligned_retrieval(reference: np.ndarray, query: np.ndarray, aligner) -> float:
    """Self-retrieval when every query is aligned to each candidate in turn.

    The alignment is done inside the comparison rather than once against a
    global reference, which is the most favourable protocol possible for the
    raw spectra: each pair is given its own optimal shift.
    """
    hits = 0
    for i, q in enumerate(query):
        best, best_distance = -1, np.inf
        for j, r in enumerate(reference):
            distance = float(np.linalg.norm(aligner(q, r) - r))
            if distance < best_distance:
                best, best_distance = j, distance
        hits += int(best == i)
    return hits / len(query)


def topological_descriptors(intensity, wavenumber, config, imagers=None):
    normalised = dataset.normalise(intensity, "minmax")
    d = features.compute_diagrams(normalised, wavenumber, config)
    tables = [peaks.peak_table(row, wavenumber,
                               peaks.PeakConfig(prominence=config.min_persistence))
              for row in normalised]
    diagrams = [peaks.attribute_diagram(t, "prominence") for t in tables]
    if imagers is None:
        fp = features.build_imagers(d, config)[1]
        probe = images.PersistenceImager(resolution=config.resolution)
        probe.fit(diagrams)
        pixel = probe.lifetime_range[1] / config.resolution
        pk = images.PersistenceImager(
            resolution=config.resolution,
            sigma=(config.sigma_wavenumber, max(pixel, 1e-6)))
        pk.fit(diagrams)
        imagers = (fp, pk)
    fp, pk = imagers
    return {"TFI": fp.transform(d.fingerprints),
            "peaks: + prominence": pk.transform(diagrams)}, imagers


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(PROCESSED / "labels.csv")
    targets = pd.read_csv(PROCESSED / "targets.csv")
    data = np.load(PROCESSED / "spectra.npz", allow_pickle=True)
    wavenumber, raw = data["wavenumber"], data["intensity"]
    families = labels["family"].to_numpy()

    config = features.DescriptorConfig(adaptive_threshold=args.adaptive)
    topological, imagers = topological_descriptors(raw, wavenumber, config)

    chains = {name: fn(raw, wavenumber) for name, fn in preprocess.CHAINS.items()}
    representations = {**chains, **topological}

    # -- B ------------------------------------------------------------------
    if "B" in args.parts:
        print("\nB. prediction and identification after pre-processing")
        rows = []
        for target in args.targets.split(","):
            y_all = targets[target].to_numpy(dtype=float)
            keep = np.isfinite(y_all)
            y = y_all[keep]
            for name, X in representations.items():
                for model, fn in (("ridge", _reg.ridge_predict),
                                  ("PLS", _reg.pls_predict)):
                    rows.append(dict(target=target, representation=name, model=model,
                                     **_reg.scores(y, fn(X[keep], y))))
        frame = pd.DataFrame(rows)
        _append(frame, RESULTS / "alignment_regression.csv")
        best = frame.loc[frame.groupby("representation").Q2.idxmax()]
        print(best[["representation", "model", "Q2"]].to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))

        rows = []
        main_subset = np.isin(families, ["TPU", "PUR", "TR"])
        for subset, mask, k in (("all (n=39)", np.ones(len(families), bool), 6),
                                ("TPU/PUR/TR (n=35)", main_subset, 3)):
            for name, X in representations.items():
                rows.append(dict(subset=subset, representation=name,
                                 **clustering.evaluate(
                                     families[mask],
                                     clustering.kmeans_labels(X[mask], k, seed=args.seed),
                                     X[mask])))
        pd.DataFrame(rows).to_csv(RESULTS / "alignment_clustering.csv", index=False)
        print()
        print(pd.DataFrame(rows).to_string(index=False,
                                           float_format=lambda v: f"{v:.3f}"))

    # -- D ------------------------------------------------------------------
    if "D" in args.parts:
        print("\nD. self-retrieval, with the query aligned to each candidate")
        rng = np.random.default_rng(args.seed)
        snv_reference = dataset.normalise(raw, "snv")
        rows = []
        for artefact in args.artefacts.split(","):
            meta = _rob.ARTEFACTS[artefact]
            for level in _rob.LEVELS:
                amplitude = level * meta["scale"]
                scores: dict[str, list[float]] = {}
                for _ in range(args.repeats):
                    perturbed = _rob.perturb(raw, wavenumber, artefact, amplitude, rng)
                    query_topological, _ = topological_descriptors(
                        perturbed, wavenumber, config, imagers)
                    query_chains = {name: fn(perturbed, wavenumber)
                                    for name, fn in preprocess.CHAINS.items()}
                    for name, X in {**query_chains, **query_topological}.items():
                        scores.setdefault(name + args.suffix, []).append(
                            _rob.self_retrieval(representations[name], X))
                    for name, aligner in ALIGNERS.items():
                        scores.setdefault(name, []).append(
                            aligned_retrieval(snv_reference,
                                              dataset.normalise(perturbed, "snv"),
                                              aligner))
                for name, values in scores.items():
                    rows.append(dict(artefact=artefact, level=float(level),
                                     amplitude=float(amplitude), representation=name,
                                     retrieval=float(np.mean(values)),
                                     sd=float(np.std(values))))
                print(f"  {artefact} level {level:.2f}  " + "  ".join(
                    f"{n[:14]}={np.mean(v):.2f}" for n, v in scores.items()))
        _append(pd.DataFrame(rows), RESULTS / "alignment_robustness.csv")

    print(f"\nwritten to {RESULTS}")


if __name__ == "__main__":
    main()
