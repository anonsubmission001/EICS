SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -euo pipefail -c


PY ?= python3
ANALYSIS_DIR := analysis
RESULTS_DIR := results/analysis
LATEX_DIR := $(RESULTS_DIR)/latex
PLOTS_DIR := $(RESULTS_DIR)/plots

.DEFAULT_GOAL := all

.PHONY: all run prepare stats infer factorial plots latex pdf clean clean-logs clean-latex help

help:
	@echo "Targets:"
	@echo "  all        - run full pipeline (same as run)"
	@echo "  run        - execute ./run_analysis.sh"
	@echo "  prepare    - generate data_prepared.csv"
	@echo "  stats      - descriptive + inferential stats"
	@echo "  infer      - inferential_stats.py"
	@echo "  factorial  - inferential_factorial.py"
	@echo "  plots      - generate all plots"
	@echo "  latex      - generate LaTeX tables + report.tex"
	@echo "  pdf        - compile report.pdf (latexmk)"
	@echo "  clean      - remove generated outputs"
	@echo "  clean-logs - remove /tmp/*.log created by scripts"
	@echo "  clean-latex- latexmk -C in latex dir"

all: run

run:
	./run_analysis.sh

prepare:
	cd $(ANALYSIS_DIR)
	$(PY) prepare_data.py

stats: descriptive infer factorial

descriptive:
	cd $(ANALYSIS_DIR)
	$(PY) descriptive_stats.py

infer:
	cd $(ANALYSIS_DIR)
	$(PY) inferential_stats.py

factorial:
	cd $(ANALYSIS_DIR)
	$(PY) inferential_factorial.py

plots:
	cd $(ANALYSIS_DIR)
	$(PY) plots/calibration.py
	$(PY) plots/plot_data_loss.py
	$(PY) plots/plot_data_loss_points.py
	$(PY) plots/plot_data_loss_points_by_calib.py
	$(PY) plots/plot_accuracy.py
	$(PY) plots/plot_heatmap.py

latex:
	cd $(ANALYSIS_DIR)
	$(PY) generate_latex.py

pdf:
	cd $(LATEX_DIR)
	latexmk -pdf report.tex

clean:
	rm -rf $(RESULTS_DIR)

clean-logs:
	rm -f /tmp/prepare_data.log /tmp/descriptive_stats.log /tmp/inferential_stats.log \
	      /tmp/inferential_factorial.log /tmp/plot_calibration.log /tmp/plot_data_loss.log \
	      /tmp/plot_data_loss_points.log /tmp/plot_data_loss_points_by_calib.log \
	      /tmp/plot_accuracy.log /tmp/plot_heatmap.log /tmp/generate_latex.log /tmp/latexmk.log

clean-latex:
	@if command -v latexmk >/dev/null 2>&1; then \
	  cd $(LATEX_DIR) && latexmk -C; \
	else \
	  echo "[WARN] latexmk not found"; \
	fi
