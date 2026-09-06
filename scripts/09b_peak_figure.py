#!/usr/bin/env python3
"""Figure for step 9 -- the controlled comparison against conventional peaks.

Four panels, one per question the comparison answers:

    (a) do the two feature sets differ at all?  Persistence against the
        prominence reported by the peak picker, feature by feature.
    (b) which conventional attribute carries the information?  Q^2 on the
        onset of degradation, and the adjusted Rand index, per descriptor.
    (c) is the robustness topological?  Self-retrieval under a wavenumber
        miscalibration, for the topological and the conventional descriptors
        and for the raw spectra.
    (d) does the peak picker's answer depend on its thresholds?

Writes:
    figures/09_peak_features.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

ORDER = ["peaks: position", "peaks: + intensity", "peaks: + width",
         "peaks: + area", "peaks: + prominence", "peaks: all five",
         "PI", "TFI", "raw spectra"]
SHORT = {"peaks: position": "position", "peaks: + intensity": "+ intensity",
         "peaks: + width": "+ width", "peaks: + area": "+ area",
         "peaks: + prominence": "+ prominence", "peaks: all five": "all five",
         "PI": "PI", "TFI": "TFI", "raw spectra": "raw spectra"}
COLOUR = {"TFI": "#0046a0", "PI": "#6b9bd2", "raw spectra": "#444444"}
PEAK_COLOUR = "#b05800"


def _colour(name: str) -> str:
    return COLOUR.get(name, PEAK_COLOUR)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))

    # -- (a) persistence against prominence ---------------------------------
    ax = axes[0, 0]
    agreement = pd.read_csv(RESULTS / "peak_agreement.csv")
    d = agreement[agreement["persistence"].notna() & agreement["matched"]].copy()
    d["edge"] = d["edge"].astype("boolean").fillna(False).astype(bool)
    interior = d[~d["essential"].astype(bool) & ~d["edge"]]
    special = d[d["essential"].astype(bool) | d["edge"]]
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1, zorder=1)
    ax.scatter(interior["prominence"], interior["persistence"], s=6,
               color=PEAK_COLOUR, alpha=0.45, zorder=2,
               label=f"interior bands ({len(interior)})")
    ax.scatter(special["prominence"], special["persistence"], s=14,
               facecolor="none", edgecolor="#0046a0", lw=0.8, zorder=3,
               label=f"essential class or edge ({len(special)})")
    equal = 100 * float((interior["persistence"] - interior["prominence"]).abs().lt(1e-9).mean())
    ax.set_xlabel("prominence reported by the peak picker")
    ax.set_ylabel("degree-0 persistence")
    ax.set_title(f"(a) the same quantity: {equal:.0f}% of interior bands agree exactly",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)

    # -- (b) which attribute carries the information? -----------------------
    ax = axes[0, 1]
    regression = pd.read_csv(RESULTS / "peak_regression.csv")
    q2 = (regression[regression.target == "T5"]
          .groupby("representation").Q2.max().reindex(ORDER))
    clusters = pd.read_csv(RESULTS / "peak_clustering.csv")
    ari = (clusters[clusters.subset.str.startswith("TPU")]
           .set_index("representation").ARI.reindex(ORDER))
    y = np.arange(len(ORDER))
    ax.barh(y - 0.2, q2.values, height=0.38, color=[_colour(n) for n in ORDER],
            label="$Q^2$, onset $T_5$")
    ax.barh(y + 0.2, ari.values, height=0.38, color=[_colour(n) for n in ORDER],
            alpha=0.45, label="ARI, families (n = 35)")
    ax.axvline(0, color="#888888", lw=0.8)
    ax.set_yticks(y, [SHORT[n] for n in ORDER], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("score")
    ax.set_title("(b) prominence is the attribute that carries the signal", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, frameon=False)

    # -- (c) robustness to a wavenumber miscalibration ----------------------
    ax = axes[1, 0]
    robust = pd.read_csv(RESULTS / "peak_robustness.csv")
    shift = robust[robust.artefact == "wavenumber shift"]
    for name in ORDER:
        sub = shift[shift.representation == name].sort_values("amplitude")
        if sub.empty:
            continue
        emphasis = name in ("TFI", "raw spectra", "peaks: + prominence")
        ax.plot(sub["amplitude"], sub["retrieval"], marker="o", ms=3,
                lw=2.0 if emphasis else 1.0,
                alpha=1.0 if emphasis else 0.35,
                color=_colour(name), label=SHORT[name] if emphasis else None)
    ax.set_xlabel(r"wavenumber miscalibration [cm$^{-1}$]")
    ax.set_ylabel("self-retrieval rate")
    ax.set_title("(c) the robustness comes from smoothing, not from topology",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    # -- (d) sensitivity of the peak picker ---------------------------------
    ax = axes[1, 1]
    sweep = pd.read_csv(RESULTS / "peak_sensitivity.csv")
    ax.scatter(sweep["n_peaks"], sweep["Q2_T5"], s=26, color=PEAK_COLOUR,
               label="peak picker, 20 threshold settings")
    tfi_q2 = float(q2["TFI"])
    ax.axhline(tfi_q2, color="#0046a0", lw=2,
               label=f"TFI, no threshold to set ($Q^2$ = {tfi_q2:.2f})")
    ax.set_xlabel("peaks detected per spectrum")
    ax.set_ylabel(r"$Q^2$, onset $T_5$")
    ax.set_ylim(0.6, 0.85)
    ax.set_title("(d) the conventional route is not very sensitive either",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(FIGURES / "09_peak_features.png", dpi=180)
    plt.close(fig)
    print(f"written to {FIGURES / '09_peak_features.png'}")


if __name__ == "__main__":
    main()
