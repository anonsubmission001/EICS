#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inferential_stats.py

Run clean inferential statistics on participant-level reliability metrics
prepared by prepare_data.py.

Input:
- ../results/analysis/data_prepared.csv

Outputs (CSV):
- ../results/analysis/tests_assumptions.csv
- ../results/analysis/tests_anova_or_kruskal.csv
- ../results/analysis/tests_pairwise.csv

What it does:
1) For each dependent variable (DV), and for each stratum:
   - position in {sitting, standing}
   - optionally position×scene (for table_pct/screen_pct)
   we compare groups defined by:
   - setup (remote vs head_mounted)  [2 groups]
   - calibration level              [>=2 groups depending on filter]
2) Chooses parametric vs non-parametric:
   - If all groups pass Shapiro AND Levene passes => ANOVA / t-test
   - else => Kruskal (k>2) / Mann-Whitney (2)
3) Computes effect sizes:
   - t-test: Cohen's d + Hedges' g
   - ANOVA: eta^2 + omega^2
   - Kruskal: epsilon^2 (approx)
4) Pairwise comparisons for k>2:
   - t-test (Welch) or Mann-Whitney
   - p-value correction with Benjamini-Hochberg (BH/FDR) within each family

Important note:
- This script treats participants as independent units.
- If you later want random effects (LMM), do that in a separate script.
"""

from __future__ import annotations

import os
from itertools import combinations
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import utils

# Use whichever module you have fixed for Python 3.8
# - If you kept utils_states.py: import utils_states as ustats
# - If you fixed utils_stats.py: import utils_stats as ustats
import utils_stats as ustats


INDIR = "../results/analysis"
INCSV = os.path.join(INDIR, "data_prepared.csv")
OUTDIR = INDIR

DV_BLOCKS = [
    # (name, dataframe column, grouping factors)
    # Setup effect on loss
    ("loss_by_position_setup", "loss_pct", ["position", "setup"]),
    # Setup effect on table/screen proportions (scene-specific DV)
    ("table_by_position_setup", "table_pct", ["position", "setup"]),
    ("screen_by_position_setup", "screen_pct", ["position", "setup"]),
]

CALIB_ORDER = ["no_issue", "slight_issues", "severe_issues"]
CALIB_ALL = CALIB_ORDER + ["impossible"]


def _ensure_outdir() -> None:
    os.makedirs(OUTDIR, exist_ok=True)


def _as_clean_numeric(x: pd.Series) -> np.ndarray:
    a = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    return a[~np.isnan(a)]


def epsilon_squared_kruskal(H: float, n: int, k: int) -> float:
    """
    Epsilon-squared effect size for Kruskal-Wallis (approx).
    """
    if n <= 1 or k <= 1:
        return float("nan")
    return float((H - (k - 1)) / (n - 1))


def assumptions_for_groups(groups: Dict[str, np.ndarray]) -> dict[str, Any]:
    """
    Compute Shapiro per group and Levene across groups.
    Returns a flat dict with:
      - shapiro_ok_all
      - levene_ok
      - per-group shapiro stats in nested form
    """
    shapiro = {}
    shapiro_ok_all = True
    for name, arr in groups.items():
        res = ustats.shapiro_test(arr)
        shapiro[name] = res
        # If too small n, we mark as not ok for parametric (conservative)
        if not res["ok"]:
            shapiro_ok_all = False

    lev = ustats.levene_test(*groups.values(), center="median")
    levene_ok = bool(lev["ok"])

    return {
        "shapiro": shapiro,
        "shapiro_ok_all": bool(shapiro_ok_all),
        "levene": lev,
        "levene_ok": bool(levene_ok),
        "parametric_ok": bool(shapiro_ok_all and levene_ok),
    }


def run_k_group_test(
    y_by_group: Dict[str, np.ndarray],
) -> dict[str, Any]:
    """
    Choose ANOVA vs Kruskal depending on assumptions.
    Returns a dict describing the omnibus test.
    """
    k = len(y_by_group)
    n = int(sum(len(v) for v in y_by_group.values()))
    if k < 2:
        return {"test": "none", "k": k, "n": n, "stat": np.nan, "p": np.nan}

    assump = assumptions_for_groups(y_by_group)
    groups_list = list(y_by_group.values())

    if assump["parametric_ok"]:
        # One-way ANOVA
        ares = ustats.anova_oneway(*groups_list)
        ares.update(
            {
                "family": "parametric",
                "stat": ares.get("F", np.nan),
                "p": ares.get("p", np.nan),
                "effect_eta2": ares.get("eta2", np.nan),
                "effect_omega2": ares.get("omega2", np.nan),
            }
        )
        return {**assump, **ares}

    # Kruskal-Wallis
    H, p = stats.kruskal(*groups_list)
    return {
        **assump,
        "test": "kruskal",
        "family": "nonparametric",
        "k": k,
        "n": n,
        "stat": float(H),
        "p": float(p),
        "effect_epsilon2": epsilon_squared_kruskal(float(H), n=n, k=k),
    }


def pairwise_tests(
    y_by_group: Dict[str, np.ndarray],
    family_name: str,
) -> pd.DataFrame:
    """
    Pairwise comparisons within a family of groups.
    Uses Welch t-test if parametric_ok, else Mann-Whitney.
    BH correction applied within this family.
    """
    labels = list(y_by_group.keys())
    pairs = list(combinations(labels, 2))
    rows: List[Dict[str, Any]] = []

    # Decide parametric based on assumptions across all groups
    assump = assumptions_for_groups(y_by_group)
    use_param = assump["parametric_ok"]

    for a, b in pairs:
        x = y_by_group[a]
        y = y_by_group[b]

        if use_param:
            res = ustats.ttest_independent(x, y, equal_var=False, alternative="two-sided")
            rows.append(
                {
                    "family": family_name,
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
                    "family": family_name,
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
    if len(out) == 0:
        return out

    out["p_adj_bh"] = ustats.p_adjust(out["p"].astype(float).to_list(), method="bh")
    out["parametric_ok"] = bool(use_param)
    out["shapiro_ok_all"] = bool(assump["shapiro_ok_all"])
    out["levene_ok"] = bool(assump["levene_ok"])
    return out.sort_values(["p_adj_bh", "p"]).reset_index(drop=True)


def main() -> None:
    _ensure_outdir()

    df = pd.read_csv(INCSV)

    # Keep only participants with data
    df = df[df["n_timestamps_total"].fillna(0).astype(int) > 0].copy()

    # Categories
    df["position"] = pd.Categorical(df["position"], categories=utils.POSITIONS, ordered=True)
    df["setup"] = pd.Categorical(df["setup"], categories=utils.SETUPS, ordered=True)

    assumptions_rows: List[Dict[str, Any]] = []
    omnibus_rows: List[Dict[str, Any]] = []
    pairwise_frames: List[pd.DataFrame] = []

    # -----------------------------
    # A) Setup effect within each position (2-group comparisons)
    # -----------------------------
    for dv_name, dv_col, grp in DV_BLOCKS:
        for pos in utils.POSITIONS:
            sdf = df[df["position"] == pos].copy()
            sdf = sdf[~sdf[dv_col].isna()]
            if sdf.empty:
                continue

            # groups: setup
            y_by = {}
            for setup in utils.SETUPS:
                arr = _as_clean_numeric(sdf.loc[sdf["setup"] == setup, dv_col])
                if arr.size > 0:
                    y_by[setup] = arr

            if len(y_by) < 2:
                continue

            fam = f"{dv_name}__position={pos}__factor=setup"
            assump = assumptions_for_groups(y_by)
            assumptions_rows.append(
                {
                    "family": fam,
                    "dv": dv_col,
                    "position": pos,
                    "factor": "setup",
                    "parametric_ok": assump["parametric_ok"],
                    "shapiro_ok_all": assump["shapiro_ok_all"],
                    "levene_ok": assump["levene_ok"],
                    "levene_p": assump["levene"]["p"],
                }
            )

            # For 2 groups, do t-test or Mann-Whitney directly
            if assump["parametric_ok"]:
                t = ustats.ttest_independent(y_by[utils.SETUPS[0]], y_by[utils.SETUPS[1]], equal_var=False)
                omnibus_rows.append(
                    {
                        "family": fam,
                        "dv": dv_col,
                        "position": pos,
                        "factor": "setup",
                        "test": "welch_t",
                        "stat": t["t"],
                        "p": t["p"],
                        "cohens_d": t.get("cohens_d", np.nan),
                        "hedges_g": t.get("hedges_g", np.nan),
                        "n_a": t["n_x"],
                        "n_b": t["n_y"],
                    }
                )
            else:
                u = ustats.mannwhitney_u(y_by[utils.SETUPS[0]], y_by[utils.SETUPS[1]])
                omnibus_rows.append(
                    {
                        "family": fam,
                        "dv": dv_col,
                        "position": pos,
                        "factor": "setup",
                        "test": "mannwhitneyu",
                        "stat": u["U"],
                        "p": u["p"],
                        "n_a": u["n_x"],
                        "n_b": u["n_y"],
                    }
                )

    # -----------------------------
    # B) Calibration effect within each position × setup
    #    Use:
    #      - loss_pct with calibration_loss
    #      - table_pct with calibration_table
    #      - screen_pct with calibration_screen
    # Filter out "impossible" by default (can change).
    # -----------------------------
    calib_specs = [
        ("loss_pct", "calibration_loss"),
        ("table_pct", "calibration_table"),
        ("screen_pct", "calibration_screen"),
    ]

    for dv_col, calib_col in calib_specs:
        for pos in utils.POSITIONS:
            for setup in utils.SETUPS:
                sdf = df[(df["position"] == pos) & (df["setup"] == setup)].copy()
                sdf = sdf[~sdf[dv_col].isna()]
                if sdf.empty:
                    continue

                # keep only the three ordered levels by default
                sdf = sdf[sdf[calib_col].isin(CALIB_ORDER)].copy()
                if sdf.empty:
                    continue

                y_by = {}
                for level in CALIB_ORDER:
                    arr = _as_clean_numeric(sdf.loc[sdf[calib_col] == level, dv_col])
                    if arr.size > 0:
                        y_by[level] = arr

                if len(y_by) < 2:
                    continue

                fam = f"{dv_col}__position={pos}__setup={setup}__factor=calibration({calib_col})"

                omnibus = run_k_group_test(y_by)
                assumptions_rows.append(
                    {
                        "family": fam,
                        "dv": dv_col,
                        "position": pos,
                        "setup": setup,
                        "factor": calib_col,
                        "parametric_ok": omnibus.get("parametric_ok", False),
                        "shapiro_ok_all": omnibus.get("shapiro_ok_all", False),
                        "levene_ok": omnibus.get("levene_ok", False),
                        "levene_p": omnibus.get("levene", {}).get("p", np.nan),
                    }
                )

                omnibus_rows.append(
                    {
                        "family": fam,
                        "dv": dv_col,
                        "position": pos,
                        "setup": setup,
                        "factor": calib_col,
                        "test": omnibus["test"],
                        "stat": omnibus["stat"],
                        "p": omnibus["p"],
                        "k": omnibus.get("k", np.nan),
                        "n": omnibus.get("n", np.nan),
                        "eta2": omnibus.get("effect_eta2", np.nan),
                        "omega2": omnibus.get("effect_omega2", np.nan),
                        "epsilon2": omnibus.get("effect_epsilon2", np.nan),
                    }
                )

                # Pairwise only if k>2
                if len(y_by) > 2:
                    pw = pairwise_tests(y_by, family_name=fam)
                    if not pw.empty:
                        # enrich
                        pw.insert(1, "dv", dv_col)
                        pw.insert(2, "position", pos)
                        pw.insert(3, "setup", setup)
                        pw.insert(4, "factor", calib_col)
                        pairwise_frames.append(pw)

    # -----------------------------
    # Export
    # -----------------------------
    df_ass = pd.DataFrame(assumptions_rows)
    df_omni = pd.DataFrame(omnibus_rows)

    # BH correction across omnibus tests (global)
    if not df_omni.empty:
        df_omni["p_adj_bh_global"] = ustats.p_adjust(df_omni["p"].astype(float).to_list(), method="bh")

    df_ass.to_csv(os.path.join(OUTDIR, "tests_assumptions.csv"), index=False)
    df_omni.to_csv(os.path.join(OUTDIR, "tests_anova_or_kruskal.csv"), index=False)

    if pairwise_frames:
        df_pw = pd.concat(pairwise_frames, ignore_index=True)
        df_pw.to_csv(os.path.join(OUTDIR, "tests_pairwise.csv"), index=False)
    else:
        # still write an empty file for reproducibility
        pd.DataFrame().to_csv(os.path.join(OUTDIR, "tests_pairwise.csv"), index=False)

    print("[OK] Wrote:")
    print(" - tests_assumptions.csv")
    print(" - tests_anova_or_kruskal.csv")
    print(" - tests_pairwise.csv")


if __name__ == "__main__":
    main()
