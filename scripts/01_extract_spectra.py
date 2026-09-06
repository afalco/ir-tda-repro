#!/usr/bin/env python3
"""Step 1 -- extract the IR/ATR spectra and the material labels from the raw data.

Reads ``data/raw/Resultados Informe.xlsx`` (one worksheet per material
reference) together with ``data/raw/references.txt`` or, when the workbook is
absent, the redistributable extract written by step 0
(``data/raw/spectra_native.npz`` and ``data/raw/references.csv``). It resamples
every spectrum onto a common wavenumber grid and writes:

    data/processed/spectra.npz   wavenumber grid and raw absorbance matrix
    data/processed/labels.csv    sample identifier, description, supplier, family
    figures/01_spectra_by_family.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irtda import dataset, plotting  # noqa: E402

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"


def main() -> int:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    workbook = RAW / "Resultados Informe.xlsx"
    references = RAW / "references.txt"
    native = RAW / "spectra_native.npz"
    references_csv = RAW / "references.csv"

    if workbook.exists() and references.exists():
        source = "the characterisation workbook"
        spectra = dataset.load_workbook_spectra(workbook, references)
    elif native.exists() and references_csv.exists():
        source = f"{native.name} and {references_csv.name}"
        spectra = dataset.load_native_spectra(native, references_csv)
    else:
        print("nothing to read in data/raw.\n", file=sys.stderr)
        print("This step reads either the characterisation workbook "
              "('Resultados Informe.xlsx'", file=sys.stderr)
        print("with 'references.txt'), which is the property of the laboratory "
              "that produced", file=sys.stderr)
        print("the study and is not redistributed, or the extract of it that is "
              "included here", file=sys.stderr)
        print(f"('{native.name}' with '{references_csv.name}'), which carries "
              "the same two columns", file=sys.stderr)
        print("per worksheet. Step 0 writes the second from the first.\n",
              file=sys.stderr)
        print("Steps 2 onwards need neither: they read data/processed, which "
              "is included.", file=sys.stderr)
        return 1

    np.savez_compressed(
        PROCESSED / "spectra.npz",
        wavenumber=spectra.wavenumber,
        intensity=spectra.intensity,
        sheet=spectra.metadata["sheet"].to_numpy(),
    )
    spectra.metadata.to_csv(PROCESSED / "labels.csv", index=False)

    normalised = dataset.normalise(spectra.intensity, "minmax")
    plotting.plot_spectra_by_family(
        spectra.wavenumber,
        normalised,
        spectra.metadata["family"].tolist(),
        FIGURES / "01_spectra_by_family.png",
    )

    print(f"{len(spectra)} spectra x {spectra.intensity.shape[1]} points "
          f"({spectra.wavenumber[0]:.0f}-{spectra.wavenumber[-1]:.0f} cm-1), "
          f"read from {source}")
    print(spectra.metadata["family"].value_counts().to_string())
    print(f"\nwritten to {PROCESSED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
