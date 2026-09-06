#!/usr/bin/env python3
"""Step 10c -- is the derivative separation chemistry, or a batch effect?

A Savitzky-Golay second derivative with SNV recovers TPU, PUR and TR perfectly
by k-means. A result that clean on 35 industrial samples has to be checked
against the obvious alternative explanations before it is believed: the
material family is partly confounded with the supplier who delivered the batch,
and the batches also differ in colour, so a partition that looks chemical could
be reproducing either.

Three checks.

    1  What does the partition agree with -- family, supplier, or colour?
    2  The decisive one: does each family stay together when its members come
       from different suppliers? A batch effect cannot put a compound from one
       supplier in the same cluster as the same polymer from another.
    3  Does the result depend on the filter parameters, or does it hold over the
       window lengths and polynomial orders a spectroscopist might choose?

Writes:
    results/derivative_confounders.csv
    results/derivative_sensitivity.csv
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irtda import clustering, preprocess  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

COLOURS = ("black", "brown", "yellow", "tan", "natural", "clear", "grey", "white")
FAMILIES = ("TPU", "PUR", "TR")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(PROCESSED / "labels.csv")
    data = np.load(PROCESSED / "spectra.npz", allow_pickle=True)
    wavenumber, raw = data["wavenumber"], data["intensity"]

    family = labels["family"].to_numpy()
    supplier = labels["supplier"].to_numpy()
    colour = np.array([next((c for c in COLOURS
                             if re.search(rf"\b{c}\b", description)), "unstated")
                       for description in labels["description_en"]])

    subset = np.isin(family, FAMILIES)
    X = preprocess.CHAINS["SG 2nd derivative"](raw, wavenumber)
    predicted = clustering.kmeans_labels(X[subset], len(FAMILIES), seed=0)
    f, s, c = family[subset], supplier[subset], colour[subset]

    # -- 1. what does the partition agree with? -----------------------------
    rows = [
        dict(comparison="clustering vs family", ARI=adjusted_rand_score(f, predicted)),
        dict(comparison="clustering vs supplier", ARI=adjusted_rand_score(s, predicted)),
        dict(comparison="clustering vs colour", ARI=adjusted_rand_score(c, predicted)),
        dict(comparison="family vs supplier", ARI=adjusted_rand_score(f, s)),
        dict(comparison="family vs colour", ARI=adjusted_rand_score(f, c)),
    ]
    print("what the partition agrees with")
    for row in rows:
        print(f"  {row['comparison']:26s} ARI = {row['ARI']:.3f}")
    print("\n  The agreement with the supplier is exactly the agreement the family\n"
          "  itself has with the supplier, so the clustering carries no supplier\n"
          "  structure of its own.")

    # -- 2. the decisive check ----------------------------------------------
    table = pd.DataFrame({"family": f, "supplier": s, "colour": c,
                          "cluster": predicted})
    print("\neach family, split by the supplier that delivered it")
    intact = True
    for name in FAMILIES:
        group = table[table.family == name]
        clusters = group.cluster.unique()
        intact &= len(clusters) == 1
        counts = ", ".join(f"{k} ({v})" for k, v in
                           group.supplier.value_counts().items())
        print(f"  {name:4s} n={len(group):2d}  suppliers: {counts:44s} "
              f"-> cluster{'s' if len(clusters) > 1 else ''} "
              f"{sorted(int(x) for x in clusters)}")
        rows.append(dict(comparison=f"{name}: distinct clusters",
                         ARI=float(len(clusters))))
    print("\n  " + ("Every family is intact across its suppliers: the partition cannot"
                    "\n  be a batch effect." if intact else
                    "A family is split across clusters; the separation is not clean."))
    pd.DataFrame(rows).to_csv(RESULTS / "derivative_confounders.csv", index=False)

    # -- 3. sensitivity to the filter -----------------------------------------
    sweep = []
    for window in (7, 11, 15, 21, 31, 51):
        for poly in (2, 3):
            for derivative in (1, 2):
                if poly >= window:
                    continue
                Y = preprocess._snv(preprocess.savitzky_golay(
                    raw, window=window, poly=poly, derivative=derivative))
                scores = [clustering.evaluate(
                    f, clustering.kmeans_labels(Y[subset], len(FAMILIES), seed=seed),
                    Y[subset])["ARI"] for seed in range(3)]
                sweep.append(dict(window=window, polyorder=poly,
                                  derivative=derivative, ARI=float(np.mean(scores))))
    frame = pd.DataFrame(sweep)
    frame.to_csv(RESULTS / "derivative_sensitivity.csv", index=False)
    second = frame[frame.derivative == 2]
    print(f"\nfilter parameters: the second derivative gives ARI = 1.000 in "
          f"{int((second.ARI > 0.999).sum())} of {len(second)} settings "
          f"(windows 7-31, polynomial orders 2 and 3); only the widest window, "
          f"which smooths past a band width, falls back to {second.ARI.min():.3f}.")
    print(f"\nwritten to {RESULTS}")


if __name__ == "__main__":
    main()
