#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inferential_factorial.py

Factorial (2x2) inferential stats on participant-level reliability metrics.

Design:
- Factors: position (sitting/standing) × setup (remote/head_mounted)
- DVs: loss_pct, table_pct, screen_pct, table_share_valid, screen_share_valid

Input:
- ../results/analysis/data_prepared.csv   (from prepare_data.py)

Outputs:
- ../results/analysis/factorial_assumptions.csv
- ../results/analysis/factorial_anova.csv
- ../results/analysis/factorial_posthoc.csv

Method:
- Primary: two-way ANOVA via statsmodels OLS with interaction:
    DV ~ C(position) * C(setup)
- Assumptions:
    * Shapiro on model residuals
    * Levene across the 4 cells (pos×setup)
  If violated, we also run a *rank-transformed ANOVA* (approx nonparam alternative):
    rank(DV) ~ C(position) * C(setup)
  and mark `analysis_type = "rank_anova"`.

Post-hoc:
- For interaction cell comparisons (4 groups), we do pairwise tests:
    * Welch t-test if parametric_ok else Mann-Whitney
  with BH correction within each DV.

Python:
- Compatible with Python 3.8.
"""

from __future__ import annotations

import os
from itertools import combinations
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import utils
import utils_stats as ustats  # make sure you've applied the Python 3.8 fix (no "|" unions)


# -----------------------------
# Config
# -----------------------------

INDIR = "../results/analysis"
INCSV = os.path.join(INDIR, "data_prepared.csv")
OUTDIR = INDIR

DVS = [
    "loss_pct",
    "table_pct",
    "screen_pct",
    "table_share_valid",
    "screen_share_valid",
]

FACTORS = ["position", "setup"]


# -----------------------------
# Statsmodels import (required)
# -----------------------------

def _import_statsmodels():
    try:
        import statsmodels.api as sm  # noqa: F401
        import statsmodels.formula.api as smf  # noqa: F401
        from statsmodels.stats.anova import anova_lm  # noqa: F401
        return True
    except Exception as e:
        print("[ERROR] statsmodels is required for factorial ANOVA.")
        print("Install it with: pip install statsmodels")
        raise


# -----------------------------
# Helpers
# -----------------------------

def _ensure_outdir() -> None:
    os.makedirs(OUTDIR, exist_ok=True)


def _clean_df(df: pd.DataFrame, dv: str) -> pd.DataFrame:
    d = df.copy()
    d[dv] = pd.to_numeric(d[dv], errors="coerce")
    d = d[~d[dv].isna()].copy()
    # keep only participants with data
    d = d[d["n_timestamps_total"].fillna(0).astype(int) > 0].copy()
    # ensure factor categories
    d["position"] = pd.Categorical(d["position"], categories=utils.POSITIONS, ordered=True)
    d["setup"] = pd.Categorical(d["setup"], categories=utils.SETUPS, ordered=True)
    return d


def _cell_labels(df: pd.DataFrame) -> List[str]:
    return [
        f"{pos}__{setup}"
        for pos in utils.POSITIONS
        for setup in utils.SETUPS
        if ((df["position"] == pos) & (df["setup"] == setup)).any()
    ]


def _groups_by_cell(df: pd.DataFrame, dv: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for pos in utils.POSITIONS:
        for setup in utils.SETUPS:
            label = f"{pos}__{setup}"
            arr = pd.to_numeric(df.loc[(df["position"] == pos) & (df["setup"] == setup), dv], errors="coerce").to_numpy(dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size > 0:
                out[label] = arr
    return out


def _residual_shapiro(residuals: np.ndarray) -> dict[str, Any]:
    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    residuals = residuals[~np.isnan(residuals)]
    # conservative: if too small, fail assumptions
    if residuals.size < 3:
        return {"test": "shapiro_resid", "n": int(residuals.size), "W": np.nan, "p": np.nan, "ok": False}
    W, p = stats.shapiro(residuals)
    return {"test": "shapiro_resid", "n": int(residuals.size), "W": float(W), "p": float(p), "ok": bool(p >= 0.05)}


def _levene_cells(groups: Dict[str, np.ndarray]) -> dict[str, Any]:
    if len(groups) < 2 or any(v.size < 2 for v in groups.values()):
        return {"test": "levene_cells", "k": len(groups), "stat": np.nan, "p": np.nan, "ok": False}
    stat, p = stats.levene(*groups.values(), center="median")
    return {"test": "levene_cells", "k": len(groups), "stat": float(stat), "p": float(p), "ok": bool(p >= 0.05)}


def _partial_eta_squared(ss_effect: float, ss_error: float) -> float:
    denom = ss_effect + ss_error
    if denom <= 0:
        return float("nan")
    return float(ss_effect / denom)


# -----------------------------
# Core: factorial ANOVA
# -----------------------------

def run_factorial_anova(df: pd.DataFrame, dv: str, rank_transform: bool = False) -> Tuple[pd.DataFrame, dict[str, Any]]:
    """
    Returns:
      - ANOVA table with partial eta^2
      - model diagnostics (residual Shapiro + Levene on cells)
    """
    _import_statsmodels()
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    d = _clean_df(df, dv)

    if rank_transform:
        d["_dv_used"] = stats.rankdata(d[dv].to_numpy(dtype=float), method="average")
        analysis_type = "rank_anova"
    else:
        d["_dv_used"] = d[dv].astype(float)
        analysis_type = "anova"

    # Fit OLS with interaction
    formula = "_dv_used ~ C(position) * C(setup)"
    model = smf.ols(formula, data=d).fit()

    # Assumptions
    sh = _residual_shapiro(model.resid)
    groups = _groups_by_cell(d, "_dv_used")
    lev = _levene_cells(groups)

    # Type II ANOVA (reasonable default when balanced-ish; still ok for 2x2)
    aov = anova_lm(model, typ=2)

    # aov index: C(position), C(setup), C(position):C(setup), Residual
    ss_error = float(aov.loc["Residual", "sum_sq"]) if "Residual" in aov.index else float("nan")

    rows = []
    for term in aov.index:
        row = {
            "dv": dv,
            "analysis_type": analysis_type,
            "term": str(term),
            "sum_sq": float(aov.loc[term, "sum_sq"]),
            "df": float(aov.loc[term, "df"]),
            "F": float(aov.loc[term, "F"]) if "F" in aov.columns else float("nan"),
            "p": float(aov.loc[term, "PR(>F)"]) if "PR(>F)" in aov.columns else float("nan"),
        }
        if term != "Residual" and not np.isnan(ss_error):
            row["partial_eta2"] = _partial_eta_squared(row["sum_sq"], ss_error)
        else:
            row["partial_eta2"] = float("nan")
        rows.append(row)

    diag = {
        "dv": dv,
        "analysis_type": analysis_type,
        "n": int(len(d)),
        "shapiro_resid_p": sh["p"],
        "shapiro_resid_ok": sh["ok"],
        "levene_cells_p": lev["p"],
        "levene_cells_ok": lev["ok"],
        "parametric_ok": bool(sh["ok"] and lev["ok"]),
    }

    return pd.DataFrame(rows), diag


# -----------------------------
# Post-hoc: pairwise cell comparisons (4 groups)
# -----------------------------

def posthoc_cells(df: pd.DataFrame, dv: str, parametric_ok: bool) -> pd.DataFrame:
    d = _clean_df(df, dv)
    # define 4 cells
    d["cell"] = d["position"].astype(str) + "__" + d["setup"].astype(str)

    groups = {}
    for cell, gdf in d.groupby("cell", observed=True):
        arr = pd.to_numeric(gdf[dv], errors="coerce").to_numpy(dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size > 0:
            groups[str(cell)] = arr

    cells = list(groups.keys())
    pairs = list(combinations(cells, 2))

    rows: List[Dict[str, Any]] = []
    for a, b in pairs:
        x = groups[a]
        y = groups[b]
        if parametric_ok:
            res = ustats.ttest_independent(x, y, equal_var=False, alternative="two-sided")
            rows.append(
                {
                    "dv": dv,
                    "comparison": "cell_pair",
                    "test": "welch_t",
                    "group_a": a,
                    "group_b": b,
                    "stat": res["t"],
                    "p": res["p"],
                    "cohens_d": res.get("cohens_d", np.nan),
                    "hedges_g": res.get("hedges_g", np.nan),
                    "n_a": res["n_x"],
                    "n_b": res["n_y"],
                }
            )
        else:
            res = ustats.mannwhitney_u(x, y, alternative="two-sided")
            rows.append(
                {
                    "dv": dv,
                    "comparison": "cell_pair",
                    "test": "mannwhitneyu",
                    "group_a": a,
                    "group_b": b,
                    "stat": res["U"],
                    "p": res["p"],
                    "n_a": res["n_x"],
                    "n_b": res["n_y"],
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["p_adj_bh"] = ustats.p_adjust(out["p"].astype(float).to_list(), method="bh")
    return out.sort_values(["p_adj_bh", "p"]).reset_index(drop=True)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    _ensure_outdir()

    df = pd.read_csv(INCSV)

    diag_rows: List[Dict[str, Any]] = []
    anova_frames: List[pd.DataFrame] = []
    posthoc_frames: List[pd.DataFrame] = []

    for dv in DVS:
        d = _clean_df(df, dv)
        if d.empty:
            continue

        # Run standard ANOVA
        aov, diag = run_factorial_anova(d, dv, rank_transform=False)
        diag_rows.append(diag)
        anova_frames.append(aov)

        # If assumptions fail, also run rank-ANOVA
        if not diag["parametric_ok"]:
            aov_r, diag_r = run_factorial_anova(d, dv, rank_transform=True)
            diag_rows.append(diag_r)
            anova_frames.append(aov_r)

        # Post-hoc on cells using decision from standard ANOVA diagnostics
        post = posthoc_cells(d, dv, parametric_ok=diag["parametric_ok"])
        if not post.empty:
            post.insert(1, "parametric_ok", bool(diag["parametric_ok"]))
            posthoc_frames.append(post)

    # Export
    df_diag = pd.DataFrame(diag_rows)
    df_anova = pd.concat(anova_frames, ignore_index=True) if anova_frames else pd.DataFrame()
    df_post = pd.concat(posthoc_frames, ignore_index=True) if posthoc_frames else pd.DataFrame()

    # Global BH correction across ANOVA terms (excluding Residual rows)
    if not df_anova.empty:
        mask = df_anova["term"] != "Residual"
        df_anova.loc[mask, "p_adj_bh_global"] = ustats.p_adjust(df_anova.loc[mask, "p"].astype(float).to_list(), method="bh")
        df_anova.loc[~mask, "p_adj_bh_global"] = np.nan

    df_diag.to_csv(os.path.join(OUTDIR, "factorial_assumptions.csv"), index=False)
    df_anova.to_csv(os.path.join(OUTDIR, "factorial_anova.csv"), index=False)
    df_post.to_csv(os.path.join(OUTDIR, "factorial_posthoc.csv"), index=False)

    print("[OK] Wrote:")
    print(" - factorial_assumptions.csv")
    print(" - factorial_anova.csv")
    print(" - factorial_posthoc.csv")


if __name__ == "__main__":
    main()
