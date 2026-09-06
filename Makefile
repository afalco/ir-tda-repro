PYTHON ?= python3

.PHONY: all export extract persistence cluster robustness sensitivity thermal regression multimodal peaks alignment window conformal rips repeatedcv test clean

all: extract persistence cluster robustness sensitivity thermal regression multimodal peaks alignment window conformal rips repeatedcv

# The redistributable extract of the characterisation workbook. Only the holder
# of the workbook runs this; the extract it writes is included, and steps 1 and
# 6 fall back to it. Deliberately outside `all`.
export:
	$(PYTHON) scripts/00_export_native.py

extract:
	$(PYTHON) scripts/01_extract_spectra.py

persistence:
	$(PYTHON) scripts/02_compute_persistence.py

cluster:
	$(PYTHON) scripts/03_clustering.py

robustness:
	$(PYTHON) scripts/04_robustness.py

sensitivity:
	$(PYTHON) scripts/05_sensitivity.py

thermal:
	$(PYTHON) scripts/06_thermal_targets.py

regression:
	$(PYTHON) scripts/07_property_regression.py

multimodal:
	$(PYTHON) scripts/08_multimodal.py

# The controlled comparison against conventional peak descriptors. Parts B and
# D are the expensive ones; they can be run a target or an artefact at a time
# with --targets / --artefacts, and each invocation adds its rows to the table.
peaks:
	$(PYTHON) scripts/09_peak_features.py
	$(PYTHON) scripts/09b_peak_figure.py

# Pre-processing and alignment baselines. The noise artefact is run twice, with
# the fixed and the adaptive pruning threshold, since Sect. 5 prescribes the
# adaptive one for exactly that case.
alignment:
	$(PYTHON) scripts/10_alignment.py --parts B --targets T5
	$(PYTHON) scripts/10_alignment.py --parts D --artefacts "wavenumber shift"
	$(PYTHON) scripts/10_alignment.py --parts D --artefacts "baseline drift"
	$(PYTHON) scripts/10_alignment.py --parts D --artefacts "intensity envelope"
	$(PYTHON) scripts/10_alignment.py --parts D --artefacts "additive noise" \
	          --adaptive --suffix " [adaptive]"
	$(PYTHON) scripts/10c_confounders.py
	$(PYTHON) scripts/10b_alignment_figure.py

# The processing statement: the prediction expressed as a moulding decision.
window:
	$(PYTHON) scripts/11_moulding_window.py

# The margin with a finite-sample guarantee, plus the bootstrap on the gain and
# the permutation test. Run after `window`: the figure of step 11 reads the
# conformal margins written here.
conformal:
	$(PYTHON) scripts/12_conformal_margin.py
	$(PYTHON) scripts/11_moulding_window.py

# The Vietoris-Rips pipeline of the literature, on the same data. Needs ripser.
rips:
	$(PYTHON) scripts/13_rips_comparison.py

# How much of the reported Q2 is the partition: the seed of the inner selection
# loop, and a repeated outer k-fold against leave-one-out.
repeatedcv:
	$(PYTHON) scripts/14_repeated_cv.py --models ridge

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf data/processed/* results/* figures/*
