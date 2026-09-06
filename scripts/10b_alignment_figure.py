#!/usr/bin/env python3
"""Figure for step 10 -- pre-processing and alignment against the descriptors.

Two panels. On the left the artefact the method was argued for, with the raw
spectra given the alignment step a spectroscopist would apply. On the right the
whole picture: what each representation retains at the largest amplitude of
each artefact, and its worst case over the four, which is the quantity that
matters when the artefact is not known in advance.

Writes:
    figures/10_alignment.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

ARTEFACTS = ["wavenumber shift", "baseline drift", "additive noise",
             "intensity envelope"]
ORDER = ["SNV", "raw, aligned (global)", "raw, aligned (intervals)",
         "SG 1st derivative", "SG 2nd derivative", "ALS baseline + SNV",
         "EMSC", "Fourier magnitude", "peaks: + prominence", "TFI"]
SHORT = {"SNV": "raw (SNV)", "raw, aligned (global)": "raw, aligned",
         "raw, aligned (intervals)": "raw, aligned by interval",
         "SG 1st derivative": "SG 1st derivative",
         "SG 2nd derivative": "SG 2nd derivative",
         "ALS baseline + SNV": "ALS baseline", "EMSC": "EMSC",
         "Fourier magnitude": "Fourier magnitude",
         "peaks: + prominence": "peaks + prominence", "TFI": "TFI"}


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(RESULTS / "alignment_robustness.csv")
    # The noise artefact is reported for the adaptive threshold, which is what
    # Sect. 5 prescribes for it; the adaptive rows carry a suffix.
    frame["representation"] = frame["representation"].str.replace(
        r"\s*\[adaptive\]$", "", regex=True)
    frame = frame.drop_duplicates(subset=["artefact", "level", "representation"],
                                  keep="last")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                             gridspec_kw={"width_ratios": [1.0, 1.35]})

    # -- (a) the wavenumber miscalibration ----------------------------------
    ax = axes[0]
    highlight = {"SNV": "#444444", "raw, aligned (global)": "#7a7a7a",
                 "Fourier magnitude": "#008060", "TFI": "#0046a0",
                 "SG 2nd derivative": "#b05800"}
    shift = frame[frame.artefact == "wavenumber shift"]
    for name, colour in highlight.items():
        sub = shift[shift.representation == name].sort_values("amplitude")
        if sub.empty:
            continue
        ax.plot(sub["amplitude"], sub["retrieval"], marker="o", ms=3.5, lw=2,
                color=colour, label=SHORT[name])
    ax.set_xlabel(r"wavenumber miscalibration [cm$^{-1}$]")
    ax.set_ylabel("self-retrieval rate")
    ax.set_title("(a) alignment restores the point-by-point comparison", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    # -- (b) every artefact at its largest amplitude ------------------------
    ax = axes[1]
    worst = (frame[frame.level == 1.0]
             .pivot(index="representation", columns="artefact", values="retrieval")
             .reindex(ORDER)[ARTEFACTS])
    grid = np.column_stack([worst.to_numpy(), worst.min(axis=1).to_numpy()])
    image = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="#222222")
    ax.set_xticks(range(5),
                  ["wavenumber\nshift", "baseline\ndrift", "additive\nnoise",
                   "intensity\nenvelope", "worst\ncase"], fontsize=8)
    ax.set_yticks(range(len(ORDER)), [SHORT[n] for n in ORDER], fontsize=8.5)
    ax.axvline(3.5, color="#222222", lw=1.5)
    ax.set_title("(b) self-retrieval at the largest amplitude of each artefact",
                 fontsize=10)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)

    fig.tight_layout()
    fig.savefig(FIGURES / "10_alignment.png", dpi=180)
    plt.close(fig)
    print(f"written to {FIGURES / '10_alignment.png'}")


if __name__ == "__main__":
    main()
