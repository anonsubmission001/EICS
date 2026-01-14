# Eye Trackers Comparison — Statistical Analysis

**Reproducible Statistical Analysis Pipeline**

This repository contains a **fully reproducible statistical analysis pipeline** for comparing eye-tracking setups (remote vs head-mounted) across participant postures (sitting vs standing) in a gaze-based manipulation task.

The pipeline produces:
* analysis-ready datasets,
* descriptive and inferential statistics,
* publication-ready figures,
* LaTeX tables,
* **a standalone compiled PDF report**.

---

## 📁 Repository Structure

```bash
eye_trackers_comparison/
├── analysis/
│   ├── prepare_data.py
│   ├── utils.py
│   ├── utils_stats.py
│   ├── heatmap.py
│   ├── descriptive_stats.py
│   ├── inferential_stats.py
│   ├── inferential_factorial.py
│   ├── generate_latex.py
│   ├── utils_stats.py
│   ├── plot_calibration.py
│   ├── plot_data_loss.py
│   ├── plot_data_loss_points.py
│   ├── plot_data_loss_points_by_calib.py
│   ├── plot_accuracy.py
│   └── plot_heatmap.py
│
├── results/
│   └── analysis/
│       ├── data_prepared.csv
│       ├── plots/
│       └── latex/
│           ├── tables.tex
│           ├── report.tex
│           └── report.pdf
│
├── run_analysis.sh
├── Makefile
└── README.md

```

---

## 📊 Data & Statistical Principles

* **Unit of analysis: participant**
* All gaze-level data are aggregated **within participant** before statistics
* No pseudo-replication from gazepoints or timestamps
* Explicit checks of:
  * normality,
  * homoscedasticity,
  * fallback to non-parametric tests when required
* Effect sizes reported systematically
* Multiple-testing correction (BH/FDR)

This pipeline is designed to be **reviewer-safe** and publication-ready.

---

## 🚀 Quick Start

**Run the full analysis pipeline**

```bash
make
# or
make all
```
This will:
1. Prepare the analysis dataset
2. Compute descriptive statistics
3. Run inferential and factorial analyses
4. Generate all figures
5. Generate LaTeX tables
6. Compile a standalone PDF report

Final output:
```bash
results/analysis/latex/report.pdf
```

---

## 🧩 Alternative Targets
Only data preparation

**1. Only data preparation**

```bash
make prepare
```

**2. Only statistics**

```bash
make stats
```

**3. Only plots**

```bash
make plots
```

**4. Generate LaTeX tables (no PDF)**

```bash
make latex
```

**5. Compile PDF report**

```bash
make pdf
```

---

## 📄 LaTeX Output

The script `generate_latex.py` produces:
```bash
results/analysis/latex/
├── descriptives_*.tex
├── tests_*.tex
├── factorial_*.tex
├── tables.tex     # master include
├── report.tex     # standalone compilable document
└── report.pdf
```

You can include all tables in a paper using:

```latex
\input{results/analysis/latex/tables.tex}
```

Or compile the standalone report directly:

```bash
cd results/analysis/latex
latexmk -pdf report.tex
```

---

## 🛠 Requirements

### Python

* Python ≥ 3.8
* numpy
* pandas
* scipy
* matplotlib
* seaborn
* statsmodels
* pylatex

```bash
pip install numpy pandas scipy matplotlib seaborn statsmodels pylatex
```

### LaTeX (optional, for PDF report)

* `latexmk`
* A standard LaTeX distribution (TeX Live / MikTeX)


---

## 🔁 Reproducibility

* All scripts are deterministic
* Figures and tables are generated programmatically
* No manual spreadsheet editing
* Suitable for:
  * supplementary materials
  * artifact evaluation
  * long-term maintenance

---

## 📄 Intended Usage

This repository supports:
* scientific publications,
* reviewer responses,
* replication studies,
* method comparisons in eye-tracking & HRI research.

---

## ✉️ Notes

* The Makefile is the recommended entry point
* Logs are written to /tmp/*.log during execution
* The pipeline is CI-friendly

---

## 👥 Authors

> Removed for blinded review

---

## 📖 How to Cite

> Removed for blinded review
