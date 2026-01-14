#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Running eye tracker statistical analysis pipeline"

# Root directory of the project
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="${ROOT_DIR}/analysis"
LATEX_DIR="${ROOT_DIR}/results/analysis/latex"

cd "${ANALYSIS_DIR}"

# -----------------------------
# Data preparation
# -----------------------------
echo "[INFO] Preparing analysis dataset"
python prepare_data.py > /tmp/prepare_data.log 2>&1

# -----------------------------
# Descriptive statistics
# -----------------------------
echo "[INFO] Computing descriptive statistics"
python descriptive_stats.py > /tmp/descriptive_stats.log 2>&1

# -----------------------------
# Inferential statistics
# -----------------------------
echo "[INFO] Running inferential statistics"
python inferential_stats.py > /tmp/inferential_stats.log 2>&1

echo "[INFO] Running factorial inferential statistics"
python inferential_factorial.py > /tmp/inferential_factorial.log 2>&1

# -----------------------------
# Plots
# -----------------------------
echo "[INFO] Generating plots"

python plot_calibration.py > /tmp/plot_calibration.log 2>&1
python plot_data_loss.py > /tmp/plot_data_loss.log 2>&1
python plot_data_loss_points.py > /tmp/plot_data_loss_points.log 2>&1
python plot_data_loss_points_by_calib.py > /tmp/plot_data_loss_points_by_calib.log 2>&1
python plot_accuracy.py > /tmp/plot_accuracy.log 2>&1
python plot_heatmap.py > /tmp/plot_heatmap.log 2>&1

# -----------------------------
# LaTeX generation
# -----------------------------
echo "[INFO] Generating LaTeX tables and report"
python generate_latex.py > /tmp/generate_latex.log 2>&1

# -----------------------------
# Compile PDF report
# -----------------------------
if command -v latexmk >/dev/null 2>&1; then
    echo "[INFO] Compiling LaTeX report (latexmk)"
    cd "${LATEX_DIR}"
    latexmk -pdf report.tex > /tmp/latexmk.log 2>&1
    cd "${ANALYSIS_DIR}"
else
    echo "[WARN] latexmk not found — skipping PDF compilation"
fi

echo "[OK] Analysis pipeline completed successfully"
echo "[INFO] Logs available in /tmp/*.log"
