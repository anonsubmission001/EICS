#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_latex.py

Generate LaTeX tables from analysis CSV outputs using PyLaTeX.

Inputs (expected in ../results/analysis/):
- descriptives_overall.csv
- descriptives_by_position_setup.csv
- descriptives_by_position_setup_calib.csv
- descriptives_by_position_setup_calib_scene.csv
- tests_anova_or_kruskal.csv
- tests_pairwise.csv
- factorial_anova.csv
- factorial_posthoc.csv

Outputs (written to ../results/analysis/latex/):
- descriptives_overall.tex
- descriptives_by_position_setup.tex
- descriptives_by_pos_setup_calib_loss.tex
- descriptives_by_pos_setup_calib_scene.tex
- tests_omnibus.tex
- tests_pairwise.tex
- factorial_anova.tex
- factorial_posthoc.tex
- tables.tex  (master file that \\input{} all tables)

Notes:
- Python 3.8 compatible.
- Uses PyLaTeX to build tables (no pandas.to_latex).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pylatex import Document, Table, Tabular, NoEscape
from pylatex.utils import escape_latex
from pylatex.errors import TableRowSizeError

# -----------------------------
# Paths
# -----------------------------

INDIR = "../results/analysis"
OUTDIR = os.path.join(INDIR, "latex")

FILES = {
    "desc_overall": os.path.join(INDIR, "descriptives_overall.csv"),
    "desc_pos_setup": os.path.join(INDIR, "descriptives_by_position_setup.csv"),
    "desc_pos_setup_calib": os.path.join(INDIR, "descriptives_by_position_setup_calib.csv"),
    "desc_pos_setup_calib_scene": os.path.join(INDIR, "descriptives_by_position_setup_calib_scene.csv"),
    "tests_omnibus": os.path.join(INDIR, "tests_anova_or_kruskal.csv"),
    "tests_pairwise": os.path.join(INDIR, "tests_pairwise.csv"),
    "factorial_anova": os.path.join(INDIR, "factorial_anova.csv"),
    "factorial_posthoc": os.path.join(INDIR, "factorial_posthoc.csv"),
}

MASTER = os.path.join(OUTDIR, "tables.tex")


# -----------------------------
# Small helpers
# -----------------------------

def ensure_outdir() -> None:
    os.makedirs(OUTDIR, exist_ok=True)


def read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df


def is_nan(x: Any) -> bool:
    try:
        return bool(np.isnan(float(x)))
    except Exception:
        return True


def fmt_float(x: Any, nd: int = 2) -> str:
    try:
        v = float(x)
    except Exception:
        return ""
    if np.isnan(v):
        return ""
    return f"{v:.{nd}f}"


def fmt_p(x: Any) -> str:
    try:
        p = float(x)
    except Exception:
        return ""
    if np.isnan(p):
        return ""
    if p < 0.001:
        return r"$<.001$"
    return f"{p:.3f}"


def fmt_mean_ci(mean: Any, lo: Any, hi: Any, nd: int = 1) -> str:
    m = fmt_float(mean, nd)
    l = fmt_float(lo, nd)
    h = fmt_float(hi, nd)
    if m == "":
        return ""
    if l == "" or h == "":
        return m
    return rf"${m}\,[{l},{h}]$"


def fmt_median_iqr(med: Any, iqr: Any, nd: int = 1) -> str:
    m = fmt_float(med, nd)
    q = fmt_float(iqr, nd)
    if m == "":
        return ""
    if q == "":
        return m
    return rf"${m}\,({q})$"

def fmt_effect(x: Any, nd: int = 3) -> str:
    try:
        v = float(x)
    except Exception:
        return ""
    if np.isnan(v):
        return ""
    return f"{v:.{nd}f}"

def human_var(name: str) -> str:
    mapping = {
        "table_pct": r"Table (\%)",
        "screen_pct": r"Screen (\%)",
        "loss_pct": r"Unrelevant / Loss (\%)",
        "table_share_valid": r"Table among valid (\%)",
        "screen_share_valid": r"Screen among valid (\%)",
        "value": "Value",
    }
    return mapping.get(name, escape_latex(name))


def human_position(x: Any) -> str:
    s = str(x)
    return {"sitting": "Sitting", "standing": "Standing"}.get(s, escape_latex(s))


def human_setup(x: Any) -> str:
    s = str(x)
    return {"remote": "Remote", "head_mounted": "Head-mounted"}.get(s, escape_latex(s))


def human_scene(x: Any) -> str:
    s = str(x)
    return {"table": "Table", "screen": "Screen"}.get(s, escape_latex(s))


def human_calib(x: Any) -> str:
    s = str(x)
    mapping = {
        "no_issue": "No issue",
        "slight_issues": "Slight issues",
        "severe_issues": "Severe issues",
        "impossible": "Impossible",
    }
    return mapping.get(s, escape_latex(s))


def write_tex_snippet(doc: Document, out_path: str) -> None:
    """
    Write only the LaTeX body created by PyLaTeX (no full preamble),
    so it can be \\input{} into a paper.
    """
    tex = doc.dumps()  # full document including preamble by default
    # We want only content; easiest is: build Document with documentclass=None
    # But pylatex still includes minimal structure. We'll strip to content between \\begin{document} ... \\end{document}.
    begin = tex.find(r"\begin{document}")
    end = tex.rfind(r"\end{document}")
    if begin != -1 and end != -1:
        content = tex[begin + len(r"\begin{document}") : end].strip() + "\n"
    else:
        content = tex

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


def make_doc() -> Document:
    # Minimal doc: good for snippet extraction
    return Document(documentclass="article", document_options=["10pt"])

def _auto_colspec(df: pd.DataFrame) -> str:
    """
    Build a column spec matching df width.
    Heuristic:
      - numeric-looking columns -> 'r'
      - others -> 'l'
    """
    specs = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        # if at least half non-NaN -> numeric
        ratio_num = float(s.notna().mean()) if len(s) > 0 else 0.0
        specs.append("r" if ratio_num >= 0.5 else "l")
    return " ".join(specs)


def add_table(
    doc: Document,
    df: pd.DataFrame,
    caption: str,
    label: str,
    colspec: Optional[str] = None,
    headers: Optional[List[str]] = None,
) -> None:
    """
    Add a LaTeX table using PyLaTeX (snippet-friendly), with automatic
    fixes if colspec/headers do not match df width.
    """
    # Ensure headers match df width
    if headers is None or len(headers) != df.shape[1]:
        headers = [str(c) for c in df.columns]

    # Ensure colspec matches df width
    if colspec is None:
        colspec = _auto_colspec(df)
    else:
        # Count columns in provided colspec (works for "l l r", "llr", etc.)
        n_spec = len([ch for ch in colspec if ch in ("l", "c", "r", "p", "m", "b")])
        if n_spec != df.shape[1]:
            colspec = _auto_colspec(df)

    with doc.create(Table(position="t")) as table:
        table.append(NoEscape(r"\centering"))
        table.append(NoEscape(r"\small"))

        tab = Tabular(colspec)
        tab.add_hline()
        tab.add_row([NoEscape(h) for h in headers])
        tab.add_hline()

        for _, row in df.iterrows():
            cells = ["" if (isinstance(x, float) and np.isnan(x)) else str(x) for x in row.to_list()]
            try:
                tab.add_row([NoEscape(c) for c in cells])
            except TableRowSizeError:
                # Hard fallback: truncate or pad to match df width
                n = df.shape[1]
                cells = (cells[:n] + [""] * n)[:n]
                tab.add_row([NoEscape(c) for c in cells])

        tab.add_hline()
        table.append(tab)
        table.add_caption(NoEscape(caption))
        table.append(NoEscape(rf"\label{{{label}}}"))




# -----------------------------
# Builders
# -----------------------------

def build_descriptives(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Input schema from descriptive_stats.py:
      group cols..., variable, n, mean, std, median, iqr, min, max, ci_low, ci_high
    Output columns:
      group cols..., Variable, n, Mean [CI], SD, Median (IQR)
    """
    out_rows: List[Dict[str, str]] = []

    # stable ordering if present
    sort_cols = [c for c in ["position", "setup", "scene", "calibration", "variable"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    for _, r in df.iterrows():
        row: Dict[str, str] = {}

        if "position" in group_cols:
            row["Position"] = human_position(r["position"])
        if "setup" in group_cols:
            row["Setup"] = human_setup(r["setup"])
        if "scene" in group_cols:
            row["Scene"] = human_scene(r["scene"])
        if "calibration" in group_cols:
            row["Calibration"] = human_calib(r["calibration"])

        row["Variable"] = human_var(str(r["variable"]))
        row["n"] = str(int(float(r["n"]))) if not is_nan(r["n"]) else ""
        row["Mean [95\\% CI]"] = fmt_mean_ci(r["mean"], r["ci_low"], r["ci_high"], nd=1)
        row["SD"] = fmt_float(r["std"], nd=1)
        row["Median (IQR)"] = fmt_median_iqr(r["median"], r["iqr"], nd=1)

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def build_tests_omnibus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input schema from inferential_stats.py:
      family,dv,position,setup,factor,test,stat,p,k,n,eta2,omega2,epsilon2,p_adj_bh_global,...
    Output (paper-friendly):
      DV, Subset, Factor, Test, Stat, p, p(BH), Effect
    """
    cols = df.columns.tolist()

    # keep only meaningful rows
    df = df.copy()
    if "p" in cols:
        df = df[~df["p"].isna()].copy()

    rows: List[Dict[str, str]] = []
    for _, r in df.iterrows():
        dv = human_var(str(r.get("dv", "")))
        factor = str(r.get("factor", ""))
        test = str(r.get("test", ""))

        subset_parts = []
        if "position" in cols and not pd.isna(r.get("position", np.nan)):
            subset_parts.append(human_position(r["position"]))
        if "setup" in cols and not pd.isna(r.get("setup", np.nan)):
            subset_parts.append(human_setup(r["setup"]))

        subset = " / ".join(subset_parts) if subset_parts else "-"

        # effect selection
        eff = ""
        if "eta2" in cols and not is_nan(r.get("eta2", np.nan)):
            eff = rf"$\eta^2={fmt_effect(r['eta2'])}$"
        elif "omega2" in cols and not is_nan(r.get("omega2", np.nan)):
            eff = rf"$\omega^2={fmt_effect(r['omega2'])}$"
        elif "epsilon2" in cols and not is_nan(r.get("epsilon2", np.nan)):
            eff = rf"$\epsilon^2={fmt_effect(r['epsilon2'])}$"
        elif "hedges_g" in cols and not is_nan(r.get("hedges_g", np.nan)):
            eff = rf"$g={fmt_effect(r['hedges_g'])}$"

        rows.append(
            {
                "DV": dv,
                "Subset": subset,
                "Factor": escape_latex(factor),
                "Test": escape_latex(test),
                "Stat": fmt_float(r.get("stat", np.nan), nd=3),
                "p": fmt_p(r.get("p", np.nan)),
                "p(BH)": fmt_p(r.get("p_adj_bh_global", np.nan)) if "p_adj_bh_global" in cols else "",
                "Effect": eff,
            }
        )

    return pd.DataFrame(rows)


def build_tests_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input schema from inferential_stats.py:
      family,dv,position,setup,factor,test,group_a,group_b,stat,p,p_adj_bh,cohens_d,hedges_g,...
    Output:
      DV, Subset, Contrast, Test, Stat, p, p(BH), Effect
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["DV", "Subset", "Contrast", "Test", "Stat", "p", "p(BH)", "Effect"])

    cols = df.columns.tolist()
    rows: List[Dict[str, str]] = []

    for _, r in df.iterrows():
        dv = human_var(str(r.get("dv", "")))

        subset_parts = []
        if "position" in cols and not pd.isna(r.get("position", np.nan)):
            subset_parts.append(human_position(r["position"]))
        if "setup" in cols and not pd.isna(r.get("setup", np.nan)):
            subset_parts.append(human_setup(r["setup"]))
        subset = " / ".join(subset_parts) if subset_parts else "-"

        a = escape_latex(str(r.get("group_a", "")))
        b = escape_latex(str(r.get("group_b", "")))
        contrast = f"{a} vs {b}"

        eff = ""
        if "hedges_g" in cols and not is_nan(r.get("hedges_g", np.nan)):
            eff = rf"$g={fmt_effect(r['hedges_g'])}$"
        elif "cohens_d" in cols and not is_nan(r.get("cohens_d", np.nan)):
            eff = rf"$d={fmt_effect(r['cohens_d'])}$"

        rows.append(
            {
                "DV": dv,
                "Subset": subset,
                "Contrast": contrast,
                "Test": escape_latex(str(r.get("test", ""))),
                "Stat": fmt_float(r.get("stat", np.nan), nd=3),
                "p": fmt_p(r.get("p", np.nan)),
                "p(BH)": fmt_p(r.get("p_adj_bh", np.nan)) if "p_adj_bh" in cols else "",
                "Effect": eff,
            }
        )

    return pd.DataFrame(rows)


def build_factorial_anova(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input schema from inferential_factorial.py:
      dv,analysis_type,term,sum_sq,df,F,p,partial_eta2,p_adj_bh_global
    Output:
      DV, Analysis, Term, F, p, p(BH), partial η²
    """
    rows: List[Dict[str, str]] = []
    for _, r in df.iterrows():
        term = str(r.get("term", ""))
        if term == "Residual":
            continue

        rows.append(
            {
                "DV": human_var(str(r.get("dv", ""))),
                "Analysis": escape_latex(str(r.get("analysis_type", ""))),
                "Term": escape_latex(term),
                "F": fmt_float(r.get("F", np.nan), nd=3),
                "p": fmt_p(r.get("p", np.nan)),
                "p(BH)": fmt_p(r.get("p_adj_bh_global", np.nan)) if "p_adj_bh_global" in df.columns else "",
                r"$\eta_p^2$": fmt_effect(r.get("partial_eta2", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def build_factorial_posthoc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input schema from inferential_factorial.py:
      dv,comparison,test,group_a,group_b,stat,p,p_adj_bh,hedges_g/cohens_d,...
    Output:
      DV, Contrast, Test, Stat, p, p(BH), Effect
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["DV", "Contrast", "Test", "Stat", "p", "p(BH)", "Effect"])

    rows: List[Dict[str, str]] = []
    for _, r in df.iterrows():
        a = escape_latex(str(r.get("group_a", "")))
        b = escape_latex(str(r.get("group_b", "")))
        contrast = f"{a} vs {b}"

        eff = ""
        if "hedges_g" in df.columns and not is_nan(r.get("hedges_g", np.nan)):
            eff = rf"$g={fmt_effect(r['hedges_g'])}$"
        elif "cohens_d" in df.columns and not is_nan(r.get("cohens_d", np.nan)):
            eff = rf"$d={fmt_effect(r['cohens_d'])}$"

        rows.append(
            {
                "DV": human_var(str(r.get("dv", ""))),
                "Contrast": contrast,
                "Test": escape_latex(str(r.get("test", ""))),
                "Stat": fmt_float(r.get("stat", np.nan), nd=3),
                "p": fmt_p(r.get("p", np.nan)),
                "p(BH)": fmt_p(r.get("p_adj_bh", np.nan)) if "p_adj_bh" in df.columns else "",
                "Effect": eff,
            }
        )
    return pd.DataFrame(rows)

def write_compilable_report(
    out_path: str,
    title: str = "Eye Trackers Comparison — Statistical Report",
    author: str = "",
    include_figures: bool = True,
) -> None:
    """
    Create a standalone LaTeX report that compiles directly.

    The report will:
    - include a minimal preamble (article)
    - load booktabs + geometry + graphicx + hyperref
    - input tables.tex from the same folder
    - optionally include key figures from ../plots (relative to OUTDIR)
    """
    # report.tex will be placed in OUTDIR (../results/analysis/latex)
    # Figures are in ../results/analysis/plots, relative path from latex/ is: ../plots/
    figures_dir = "../plots"

    fig_paths = [
        # calibration
        f"{figures_dir}/calibration_score_sitting.png",
        f"{figures_dir}/calibration_score_standing.png",
        # data loss
        f"{figures_dir}/data_loss_sitting.png",
        f"{figures_dir}/data_loss_standing.png",
        f"{figures_dir}/data_loss_points_sitting.png",
        f"{figures_dir}/data_loss_points_standing.png",
        f"{figures_dir}/data_loss_points_by_calib_sitting.png",
        f"{figures_dir}/data_loss_points_by_calib_standing.png",
        # heatmaps
        f"{figures_dir}/heatmap_table.png",
        f"{figures_dir}/heatmap_screen.png",
        # accuracy
        f"{figures_dir}/accuracy_time_sitting_table_grasp.png",
        f"{figures_dir}/accuracy_time_standing_table_grasp.png",
        f"{figures_dir}/accuracy_time_sitting_table_release.png",
        f"{figures_dir}/accuracy_time_standing_table_release.png",
    ]

    lines: List[str] = []
    lines += [
        r"\documentclass[10pt]{article}",
        r"\usepackage[a4paper,margin=2.2cm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\usepackage{hyperref}",
        r"\usepackage{caption}",
        r"\usepackage{subcaption}",
        r"\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,citecolor=blue}",
        r"\title{" + title + r"}",
        r"\author{" + (author if author else r"\vspace{-1em}") + r"}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\tableofcontents",
        r"\clearpage",
        "",
        r"\section{Tables}",
        r"All tables below are auto-generated from the analysis CSV outputs.",
        r"\input{tables.tex}",
        "",
    ]

    if include_figures:
        lines += [
            r"\clearpage",
            r"\section{Figures}",
            r"Figures below are generated by the plotting scripts in \texttt{analysis/plots/}.",
            "",
        ]

        # Include figures only if files exist (relative to report.tex folder)
        for p in fig_paths:
            # We check existence on disk using OUTDIR anchor
            abs_path = os.path.normpath(os.path.join(OUTDIR, p))
            if os.path.exists(abs_path):
                lines += [
                    r"\begin{figure}[H]",
                    r"\centering",
                    r"\includegraphics[width=0.95\linewidth]{" + p.replace("\\", "/") + r"}",
                    r"\caption{" + escape_latex(os.path.basename(p)) + r"}",
                    r"\end{figure}",
                    "",
                ]

    lines += [r"\end{document}", ""]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# -----------------------------
# Main generation
# -----------------------------

def main() -> None:
    ensure_outdir()

    generated: List[str] = []

    # --- Descriptives: overall
    df = read_csv(FILES["desc_overall"])
    if df is not None and not df.empty:
        tdf = build_descriptives(df, group_cols=[])
        doc = make_doc()
        add_table(
            doc,
            tdf,
            caption="Descriptive statistics (overall). Values are mean with 95\\% CI, SD, and median (IQR).",
            label="tab:desc_overall",
            colspec="l r l l l",
            headers=["Variable", "n", "Mean [95\\% CI]", "SD", "Median (IQR)"],
        )
        out = os.path.join(OUTDIR, "descriptives_overall.tex")
        write_tex_snippet(doc, out)
        generated.append("descriptives_overall.tex")

    # --- Descriptives: by position × setup
    df = read_csv(FILES["desc_pos_setup"])
    if df is not None and not df.empty:
        tdf = build_descriptives(df, group_cols=["position", "setup"])
        doc = make_doc()
        add_table(
            doc,
            tdf,
            caption="Descriptive statistics by position and setup.",
            label="tab:desc_pos_setup",
            colspec="l l l r l l l",
            headers=["Position", "Setup", "Variable", "n", "Mean [95\\% CI]", "SD", "Median (IQR)"],
        )
        out = os.path.join(OUTDIR, "descriptives_by_position_setup.tex")
        write_tex_snippet(doc, out)
        generated.append("descriptives_by_position_setup.tex")

    # --- Descriptives: loss by position × setup × calibration (loss)
    df = read_csv(FILES["desc_pos_setup_calib"])
    if df is not None and not df.empty:
        tdf = build_descriptives(df, group_cols=["position", "setup", "calibration"])
        doc = make_doc()
        add_table(
            doc,
            tdf,
            caption="Descriptive statistics for data loss by position, setup, and calibration (loss label).",
            label="tab:desc_pos_setup_calib_loss",
            colspec="l l l l r l l",
            headers=["Position", "Setup", "Calibration", "Variable", "n", "Mean [95\\% CI]", "SD"],
        )
        out = os.path.join(OUTDIR, "descriptives_by_pos_setup_calib_loss.tex")
        write_tex_snippet(doc, out)
        generated.append("descriptives_by_pos_setup_calib_loss.tex")

    # --- Descriptives: scene-specific calibration (table/screen)
    df = read_csv(FILES["desc_pos_setup_calib_scene"])
    if df is not None and not df.empty:
        # In this table, "variable" is "value", we still format
        tdf = df.copy()
        # expected columns: position, setup, scene, calibration, variable, ...
        tdf = build_descriptives(tdf, group_cols=["position", "setup", "scene", "calibration"])
        doc = make_doc()
        add_table(
            doc,
            tdf,
            caption="Descriptive statistics by position, setup, scene, and scene-specific calibration.",
            label="tab:desc_pos_setup_calib_scene",
            colspec="l l l l l r l l",
            headers=["Position", "Setup", "Scene", "Calibration", "Variable", "n", "Mean [95\\% CI]", "SD"],
        )
        out = os.path.join(OUTDIR, "descriptives_by_pos_setup_calib_scene.tex")
        write_tex_snippet(doc, out)
        generated.append("descriptives_by_pos_setup_calib_scene.tex")

    # --- Omnibus tests
    df = read_csv(FILES["tests_omnibus"])
    if df is not None and not df.empty:
        tdf = build_tests_omnibus(df)
        doc = make_doc()
        add_table(
            doc,
            tdf,
            caption="Omnibus inferential tests. BH indicates Benjamini--Hochberg correction across omnibus tests.",
            label="tab:tests_omnibus",
            colspec="l l l l r r r l",
            headers=["DV", "Subset", "Factor", "Test", "Stat", "p", "p(BH)", "Effect"],
        )
        out = os.path.join(OUTDIR, "tests_omnibus.tex")
        write_tex_snippet(doc, out)
        generated.append("tests_omnibus.tex")

    # --- Pairwise tests
    df = read_csv(FILES["tests_pairwise"])
    if df is not None and not df.empty:
        tdf = build_tests_pairwise(df)
        doc = make_doc()
        add_table(
            doc,
            tdf,
            caption="Pairwise comparisons. p(BH) is corrected within each comparison family.",
            label="tab:tests_pairwise",
            colspec="l l l l r r r l",
            headers=["DV", "Subset", "Contrast", "Test", "Stat", "p", "p(BH)", "Effect"],
        )
        out = os.path.join(OUTDIR, "tests_pairwise.tex")
        write_tex_snippet(doc, out)
        generated.append("tests_pairwise.tex")

    # --- Factorial ANOVA
    df = read_csv(FILES["factorial_anova"])
    if df is not None and not df.empty:
        tdf = build_factorial_anova(df)
        doc = make_doc()
        add_table(
            doc,
            tdf,
            caption="Factorial analysis (setup $\\times$ position). Partial $\\eta_p^2$ reported.",
            label="tab:factorial_anova",
            colspec="l l l r r r r",
            headers=["DV", "Analysis", "Term", "F", "p", "p(BH)", r"$\eta_p^2$"],
        )
        out = os.path.join(OUTDIR, "factorial_anova.tex")
        write_tex_snippet(doc, out)
        generated.append("factorial_anova.tex")

    # --- Factorial posthoc
    df = read_csv(FILES["factorial_posthoc"])
    if df is not None and not df.empty:
        tdf = build_factorial_posthoc(df)
        doc = make_doc()
        add_table(
            doc,
            tdf,
            caption="Post-hoc comparisons between position $\\times$ setup cells (BH corrected).",
            label="tab:factorial_posthoc",
            colspec="l l l r r r l",
            headers=["DV", "Contrast", "Test", "Stat", "p", "p(BH)", "Effect"],
        )
        out = os.path.join(OUTDIR, "factorial_posthoc.tex")
        write_tex_snippet(doc, out)
        generated.append("factorial_posthoc.tex")

    # --- Master file
    with open(MASTER, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by generate_latex.py\n")
        f.write("% Include this file with: \\input{results/analysis/latex/tables.tex}\n\n")
        for name in generated:
            f.write(rf"\input{{{name}}}" + "\n\n")

    print("[OK] Generated LaTeX snippets in:", OUTDIR)
    print("[OK] Master include file:", MASTER)

    # --- Standalone compilable report
    report_path = os.path.join(OUTDIR, "report.tex")
    write_compilable_report(
        report_path,
        title="Eye Trackers Comparison — Statistical Report",
        author="",
        include_figures=True,
    )
    print("[OK] Standalone report:", report_path)



if __name__ == "__main__":
    main()
