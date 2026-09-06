"""Thermogravimetric targets and hardness grades.

The characterisation workbook records, next to each IR spectrum, the
thermogravimetric run of the same specimen: elapsed time, sample temperature
and residual mass. From those curves this module derives the quantities that
bound the thermal processing window of a compound, and which are used as
regression targets for the topological descriptors of the spectrum.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from scipy.signal import savgol_filter

__all__ = [
    "read_tg_curves",
    "thermal_features",
    "parse_hardness",
    "TG_TARGETS",
]

# Regression targets derived from the TG/DTG curves, with the label used in
# figures and tables.
TG_TARGETS = {
    "T5": "5 % mass-loss temperature [C]",
    "T10": "10 % mass-loss temperature [C]",
    "T50": "50 % mass-loss temperature [C]",
    "T_dtg_peak": "peak mass-loss-rate temperature [C]",
    "residue": "residue at 800 C [% w/w]",
    "n_steps": "number of decomposition steps",
}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _locate_tg_columns(rows) -> tuple[int, dict[str, int]] | None:
    """Find the header row of the TG block and the index of its columns.

    The block is not at a fixed position: some worksheets carry an extra
    column before it, so the header text is matched instead of the column
    letter.
    """
    for i, row in enumerate(rows[:10]):
        cells = [str(c).strip() if c is not None else "" for c in row]
        if "t" in cells and "Ts" in cells and "Value" in cells:
            return i, {c: j for j, c in enumerate(cells) if c}
    return None


def read_tg_curves(xlsx_path: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return ``{sheet: (temperature [C], residual mass [% of initial])}``.

    The recorded mass is in milligrams and depends on how much material was
    loaded, so it is expressed as a percentage of the initial mass; the
    reference is the mean of the first five points, which is more stable than
    the single first reading.
    """
    workbook = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=False)
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for sheet_name in workbook.sheetnames:
        rows = list(workbook[sheet_name].iter_rows(values_only=True))
        located = _locate_tg_columns(rows)
        if located is None:
            continue
        start, columns = located

        temperature, mass = [], []
        for row in rows[start + 1:]:
            t, m = row[columns["Ts"]], row[columns["Value"]]
            if isinstance(t, (int, float)) and isinstance(m, (int, float)):
                temperature.append(float(t))
                mass.append(float(m))

        curve = _finalise_tg(np.asarray(temperature), np.asarray(mass))
        if curve is not None:
            curves[sheet_name] = curve

    return curves


def _finalise_tg(
    temperature: np.ndarray, mass: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Sort a raw TG record by temperature and express the mass as a percentage.

    Returns ``None`` for a record too short to be a run. Shared by the two
    readers so that the workbook and the redistributable extract of step 0 are
    treated identically.
    """
    if len(temperature) < 50:
        return None
    order = np.argsort(temperature)
    T, W = temperature[order], mass[order]
    return T, 100.0 * W / W[:5].mean()


def read_tg_curves_native(
    npz_path: str | Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read the TG runs from the redistributable extract written by step 0.

    The file holds one ``T_<sheet>`` / ``m_<sheet>`` pair per worksheet that
    records a run, the temperature in degrees Celsius and the mass in
    milligrams as the instrument reported them, so that the conversion to a
    percentage of the initial mass stays in the code and remains checkable.
    """
    data = np.load(npz_path, allow_pickle=False)
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in (str(sheet) for sheet in data["sheet"]):
        if f"T_{name}" not in data:
            continue
        curve = _finalise_tg(data[f"T_{name}"], data[f"m_{name}"])
        if curve is not None:
            curves[name] = curve
    return curves


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _crossing(T: np.ndarray, w: np.ndarray, level: float) -> float:
    """Temperature at which the mass first falls below ``level`` percent."""
    below = np.flatnonzero(w <= level)
    if below.size == 0:
        return float("nan")
    i = int(below[0])
    if i == 0:
        return float(T[0])
    # Linear interpolation between the bracketing samples.
    w0, w1 = w[i - 1], w[i]
    t0, t1 = T[i - 1], T[i]
    if w1 == w0:
        return float(t1)
    return float(t0 + (level - w0) * (t1 - t0) / (w1 - w0))


def thermal_features(
    T: np.ndarray, w: np.ndarray, step_threshold: float = 0.15
) -> dict[str, float]:
    """Derive the processing-window descriptors of one TG curve.

    ``step_threshold`` is the fraction of the maximum mass-loss rate above
    which a DTG maximum counts as a separate decomposition step.
    """
    # Interpolate onto a regular temperature grid so that the derivative is
    # taken with a constant step.
    grid = np.linspace(max(T.min(), 40.0), min(T.max(), 800.0), 600)
    w_grid = np.interp(grid, T, w)
    smooth = savgol_filter(w_grid, window_length=31, polyorder=3)

    dtg = -np.gradient(smooth, grid)  # % per degree, positive on mass loss
    dtg_smooth = savgol_filter(dtg, window_length=31, polyorder=3)

    peak = float(grid[int(np.argmax(dtg_smooth))])

    # Count interior maxima of the DTG curve that exceed the threshold.
    interior = dtg_smooth[1:-1]
    is_max = (interior > dtg_smooth[:-2]) & (interior > dtg_smooth[2:])
    tall = interior > step_threshold * dtg_smooth.max()
    n_steps = int((is_max & tall).sum())

    return {
        "T5": _crossing(grid, smooth, 95.0),
        "T10": _crossing(grid, smooth, 90.0),
        "T50": _crossing(grid, smooth, 50.0),
        "T_dtg_peak": peak,
        "residue": float(smooth[-1]),
        "n_steps": n_steps,
    }


# ---------------------------------------------------------------------------
# Hardness grades
# ---------------------------------------------------------------------------
# Shore A values are converted to the Shore D scale so that grades quoted on
# either scale can be regressed together. The relation is the usual empirical
# correspondence and is only approximate; conversions are therefore reported
# separately from the thermal targets.
_SHORE_A = np.array([40, 50, 60, 70, 80, 85, 90, 95, 100])
_SHORE_D = np.array([8, 12, 16, 22, 30, 35, 40, 50, 58])


def shore_a_to_d(value: float) -> float:
    return float(np.interp(value, _SHORE_A, _SHORE_D))


def parse_hardness(description: str) -> tuple[float, str] | tuple[None, None]:
    """Extract a Shore hardness grade from a supplier description.

    Grades appear either as a bare two-digit number at the end of a trade name
    ("TR negro R-C42 68"), or prefixed by the scale ("TPU crudo D65",
    "TPU negro A70NP50"). Returns the value and the scale it was quoted on.
    """
    text = description.strip()

    # Explicit scale prefix.
    m = re.search(r"\b([AD])\s?(\d{2})\b", text)
    if m:
        return float(m.group(2)), m.group(1)

    # Trailing bare grade, the common convention for TR compounds.
    m = re.search(r"(\d{2})\s*$", text)
    if m:
        value = float(m.group(1))
        if 30 <= value <= 99:
            return value, "A?"

    return None, None


def build_targets(
    xlsx_path: str | Path, labels: pd.DataFrame
) -> pd.DataFrame:
    """Assemble the regression targets, one row per sample of ``labels``.

    ``xlsx_path`` is either the characterisation workbook or the ``.npz``
    extract of step 0; the two give the same curves.
    """
    source = Path(xlsx_path)
    if source.suffix.lower() == ".npz":
        curves = read_tg_curves_native(source)
    else:
        curves = read_tg_curves(source)

    rows = []
    for _, meta in labels.iterrows():
        sheet = str(meta["sheet"])
        record: dict[str, object] = {"sheet": sheet, "family": meta["family"]}

        if sheet in curves:
            record.update(thermal_features(*curves[sheet]))
        else:
            record.update({k: float("nan") for k in TG_TARGETS})

        value, scale = parse_hardness(str(meta["description"]))
        record["hardness_raw"] = value if value is not None else float("nan")
        record["hardness_scale"] = scale
        record["shore_d"] = (
            shore_a_to_d(value) if (value is not None and scale != "D") else value
        )
        rows.append(record)

    return pd.DataFrame(rows)
