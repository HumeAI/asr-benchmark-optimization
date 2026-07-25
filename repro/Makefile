# Rebuild the paper's figures.
#
#   make figures    # everything the shipped data supports
#   make all        # also the figures needing BENCHMAXX_DATA
#
# Output goes to repro/out (override with BENCHMAXX_FIGURES).

PY ?= python

# Driven entirely by the derived data in repro/data — these work in a clean clone.
CELL_FIGS = \
	battery_panels.py \
	battery_ablation_fig.py \
	battery_masked_ablation_fig.py \
	isolate_fig.py \
	libnum_voice_gap_fig.py \
	libnum_voice_ladder_fig.py \
	nummask_lift_fig.py \
	orthohon_voice2x2_fig.py \
	patch_dissociation_fig.py \
	steer_causal_fig.py \
	wer_vs_badref_fig.py \
	regen_whitebox_figs.py \
	ortho_switch_fig.py

# Need a full results root in BENCHMAXX_DATA; see REPRODUCE.md.
DATA_FIGS = \
	masking_datasets_fig.py \
	masking_voice_fig.py \
	paired_grids.py

.PHONY: figures all cells clean
figures:
	@for f in $(CELL_FIGS); do echo "== $$f"; (cd figures && $(PY) $$f) || exit 1; done
	@echo "== consfull_fig.py robustness"
	@cd figures && $(PY) consfull_fig.py robustness
	@echo
	@echo "Wrote $$(ls $${BENCHMAXX_FIGURES:-out}/*.png | wc -l) PNGs to $${BENCHMAXX_FIGURES:-repro/out}"

all: figures
	@for f in $(DATA_FIGS); do echo "== $$f"; (cd figures && $(PY) $$f) || exit 1; done

# Re-derive the shipped cells from raw model outputs. Needs BENCHMAXX_DATA.
cells:
	$(PY) precompute_cells.py

clean:
	rm -rf out
