"""The redistributable extract reproduces data/processed.

Steps 1 and 6 read the characterisation workbook when it is present and the
extract written by step 0 when it is not. The workbook is not redistributed, so
what a third party can check is the second route: that reading
``data/raw/spectra_native.npz``, ``data/raw/tg_native.npz`` and
``data/raw/references.csv`` reproduces the ``data/processed`` files shipped with
this repository. That is what these tests do.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from irtda import dataset, thermal

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

NATIVE_SPECTRA = RAW / "spectra_native.npz"
NATIVE_TG = RAW / "tg_native.npz"
REFERENCES = RAW / "references.csv"

needs_extract = pytest.mark.skipif(
    not (NATIVE_SPECTRA.exists() and NATIVE_TG.exists() and REFERENCES.exists()),
    reason="the extract of step 0 is not present in data/raw",
)


@needs_extract
def test_native_spectra_reproduce_processed():
    spectra = dataset.load_native_spectra(NATIVE_SPECTRA, REFERENCES)
    shipped = np.load(PROCESSED / "spectra.npz", allow_pickle=True)

    # Not bit-for-bit: np.interp differs in the last bit between platforms and
    # numpy builds, which is irrelevant to every quantity computed from it.
    np.testing.assert_allclose(spectra.wavenumber, shipped["wavenumber"],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(spectra.intensity, shipped["intensity"],
                               rtol=1e-12, atol=1e-12)
    assert list(spectra.metadata["sheet"]) == [str(s) for s in shipped["sheet"]]


@needs_extract
def test_native_labels_reproduce_processed():
    spectra = dataset.load_native_spectra(NATIVE_SPECTRA, REFERENCES)
    shipped = pd.read_csv(PROCESSED / "labels.csv")

    rebuilt = spectra.metadata.copy()
    rebuilt["sheet"] = rebuilt["sheet"].astype(str)
    shipped["sheet"] = shipped["sheet"].astype(str)
    pd.testing.assert_frame_equal(
        rebuilt.reset_index(drop=True),
        shipped.reset_index(drop=True),
        check_dtype=False,
    )


@needs_extract
def test_native_tg_reproduces_targets(tmp_path):
    labels = pd.read_csv(PROCESSED / "labels.csv")
    curves = thermal.read_tg_curves_native(NATIVE_TG)
    assert set(curves) >= {str(s) for s in labels["sheet"]}

    # Compared after the same round trip through CSV that step 6 performs, so
    # that missing values and float formatting are those of the shipped file.
    rebuilt_path = tmp_path / "targets.csv"
    thermal.build_targets(NATIVE_TG, labels).to_csv(rebuilt_path, index=False)

    pd.testing.assert_frame_equal(
        pd.read_csv(rebuilt_path),
        pd.read_csv(PROCESSED / "targets.csv"),
        check_dtype=False,
    )
