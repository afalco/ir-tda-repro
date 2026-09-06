#!/usr/bin/env python3
"""Step 0 -- export the redistributable extract of the raw characterisation data.

The characterisation workbook ``data/raw/Resultados Informe.xlsx`` is the
laboratory's own report document and is not redistributed. Of it, step 1 reads
only two columns per worksheet -- the wavenumber grid and the ATR absorbance --
together with the reference table of ``data/raw/references.txt``. This script
writes exactly those, and nothing else, in open formats:

    data/raw/spectra_native.npz   'sheet': the worksheet names, in workbook
                                  order; 'k_<sheet>' and 'y_<sheet>': that
                                  worksheet's wavenumber [cm^-1] and absorbance
                                  on the instrument's own grid
    data/raw/tg_native.npz        'sheet' as above; 'T_<sheet>' and 'm_<sheet>':
                                  the temperature [C] and the mass [mg] of the
                                  TG run recorded on the same specimen, as the
                                  instrument reported them, present only for the
                                  worksheets that hold a run
    data/raw/references.csv       reference, description, supplier

With those three files present, steps 1 and 6 reproduce ``data/processed``
without the workbook, and everything they do to the raw records --- the
interpolation onto the common grid, the conversion of the mass to a percentage
of the initial load --- becomes checkable. Only the holder of the workbook
needs to run this step; everyone else starts at step 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irtda import dataset, thermal  # noqa: E402

RAW = ROOT / "data" / "raw"


def main() -> int:
    workbook_path = RAW / "Resultados Informe.xlsx"
    references_path = RAW / "references.txt"
    missing = [p.name for p in (workbook_path, references_path) if not p.exists()]
    if missing:
        print(f"missing from data/raw: {', '.join(missing)}", file=sys.stderr)
        print("This step is for the holder of the characterisation workbook; "
              "the extract it writes is already included.", file=sys.stderr)
        return 1

    workbook = openpyxl.load_workbook(workbook_path, read_only=True, data_only=False)

    arrays: dict[str, np.ndarray] = {}
    names: list[str] = []
    for sheet_name in workbook.sheetnames:
        k, y = dataset._read_sheet(workbook[sheet_name])
        if len(k) == 0:
            continue
        if f"k_{sheet_name}" in arrays:
            raise ValueError(f"duplicate worksheet name: {sheet_name!r}")
        names.append(sheet_name)
        arrays[f"k_{sheet_name}"] = k
        arrays[f"y_{sheet_name}"] = y

    # Unicode rather than object dtype, so that the archived file loads with
    # allow_pickle=False.
    arrays["sheet"] = np.asarray(names, dtype=np.str_)

    out_npz = RAW / "spectra_native.npz"
    np.savez_compressed(out_npz, **arrays)

    # The TG run recorded next to each spectrum, read exactly as step 6 reads
    # it: the header row that carries 't', 'Ts' and 'Value' locates the two
    # columns. Sheets without a run simply contribute no T_/m_ pair.
    tg_book = openpyxl.load_workbook(workbook_path, read_only=True, data_only=False)
    tg_arrays: dict[str, np.ndarray] = {"sheet": np.asarray(names, dtype=np.str_)}
    n_runs = 0
    for sheet_name in names:
        rows = list(tg_book[sheet_name].iter_rows(values_only=True))
        located = thermal._locate_tg_columns(rows)
        if located is None:
            continue
        start, columns = located
        temperature, mass = [], []
        for row in rows[start + 1:]:
            t, m = row[columns["Ts"]], row[columns["Value"]]
            if isinstance(t, (int, float)) and isinstance(m, (int, float)):
                temperature.append(float(t))
                mass.append(float(m))
        if not temperature:
            continue
        tg_arrays[f"T_{sheet_name}"] = np.asarray(temperature, dtype=float)
        tg_arrays[f"m_{sheet_name}"] = np.asarray(mass, dtype=float)
        n_runs += 1

    out_tg = RAW / "tg_native.npz"
    np.savez_compressed(out_tg, **tg_arrays)

    refs = dataset.read_references(references_path)
    table = pd.DataFrame(
        [{"reference": n, "description": d, "supplier": s}
         for n, (d, s) in sorted(refs.items())]
    )
    out_csv = RAW / "references.csv"
    table.to_csv(out_csv, index=False)

    print(f"{len(names)} worksheets exported, {n_runs} of them with a TG run")
    print(f"  {out_npz.name}  {out_npz.stat().st_size / 1e6:.2f} MB")
    print(f"  {out_tg.name}  {out_tg.stat().st_size / 1e6:.2f} MB")
    print(f"  {out_csv.name}  {len(table)} references, "
          f"{out_csv.stat().st_size / 1e3:.1f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
