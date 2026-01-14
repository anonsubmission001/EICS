#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
descriptive_stats.py

Generate clean descriptive-statistics tables from the analysis-ready dataset
produced by prepare_data.py.

Input:
- ../results/analysis/data_prepared.csv

Outputs:
- ../results/analysis/descriptives_overall.csv
- ../results/analysis/descriptives_by_position_setup.csv
- ../results/analysis/descriptives_by_position_setup_calib.csv
- ../results/analysis/descriptives_by_position_setup_calib_scene.csv

Notes:
- Stats are computed per group on participant-level variables:
  table_pct, screen_pct, loss_pct, table_share_valid, screen_share_valid
- Confidence intervals are Student-t CIs on the mean (when n>=2).
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd

import utils

# If you named your module utils_states.py, use that:
import utils_stats as ustats
# If you prefer the uploaded utils_stats.py, replace the import above by:
# import utils_stats as ustats


INDIR = "../results/analysis"
INCSV = os.path.join(INDIR, "data_prepared.csv")
OUTDIR = os.path.join(INDIR)

NUM_VARS = [
    "table_pct",
    "screen_pct",
    "loss_pct",
    "table_share_valid",
    "screen_share_valid",
]

# For "calibration", we’ll summarize using the scene-specific label
# (calibration can differ between table and screen in remote setup).
CALIB_COL_BY_SCENE = {
    "table": "calibration_table",
    "screen": "calibration_screen",
    "loss": "calibration_loss",
}

CALIB_ORDER = ["no_issue", "slight_issues", "severe_issues", "impossible"]


def _ensure_outdir() -> None:
    os.makedirs(OUTDIR, exist_ok=True)


def _safe_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _summarize_vector(x: pd.Series, ci: float = 0.95) -> dict[str, float]:
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    arr = arr[~np.isnan(arr)]
    desc = ustats.describe(arr)

    mean_, ci_low, ci_high = ustats.confidence_interval_mean(arr, confidence=ci)
    return {
        "n": int(desc["n"]),
        "mean": _safe_float(desc["mean"]),
        "std": _safe_float(desc["std"]),
        "median": _safe_float(desc["median"]),
        "iqr": _safe_float(desc["iqr"]),
        "min": _safe_float(desc["min"]),
        "max": _safe_float(desc["max"]),
        "ci_low": _safe_float(ci_low),
        "ci_high": _safe_float(ci_high),
    }


def summarize_df(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
    ci: float = 0.95,
) -> pd.DataFrame:
    """
    Return a tidy table with one row per (group, variable).
    Columns:
      group_cols..., variable, n, mean, std, median, iqr, min, max, ci_low, ci_high
    """
    if len(group_cols) == 0:
        # One global group
        grouped = [((), df)]
    else:
        grouped = list(df.groupby(group_cols, dropna=False, observed=True))

    rows: list[dict[str, object]] = []
    for gkey, gdf in grouped:
        # Normalize gkey into tuple
        if len(group_cols) == 0:
            gkey_t = ()
        elif isinstance(gkey, tuple):
            gkey_t = gkey
        else:
            gkey_t = (gkey,)

        g_info = {col: gkey_t[i] for i, col in enumerate(group_cols)}
        for v in value_cols:
            s = _summarize_vector(gdf[v], ci=ci)
            rows.append(
                {
                    **g_info,
                    "variable": v,
                    **s,
                }
            )

    out = pd.DataFrame(rows)

    # nice ordering
    if "position" in out.columns:
        out["position"] = pd.Categorical(out["position"], categories=utils.POSITIONS, ordered=True)
    if "setup" in out.columns:
        out["setup"] = pd.Categorical(out["setup"], categories=utils.SETUPS, ordered=True)
    if "calibration" in out.columns:
        out["calibration"] = pd.Categorical(out["calibration"], categories=CALIB_ORDER, ordered=True)

    sort_cols = [c for c in ["position", "setup", "scene", "calibration", "variable"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def main() -> None:
    _ensure_outdir()

    df = pd.read_csv(INCSV)

    # Optional: keep only participants with at least some timestamps
    df = df[df["n_timestamps_total"].fillna(0).astype(int) > 0].copy()

    # Harmonize categories
    df["position"] = pd.Categorical(df["position"], categories=utils.POSITIONS, ordered=True)
    df["setup"] = pd.Categorical(df["setup"], categories=utils.SETUPS, ordered=True)

    # -----------------------------------------
    # 1) Overall descriptives
    # -----------------------------------------
    overall = summarize_df(df, group_cols=[], value_cols=NUM_VARS, ci=0.95)
    overall.to_csv(os.path.join(OUTDIR, "descriptives_overall.csv"), index=False)

    # -----------------------------------------
    # 2) By position × setup
    # -----------------------------------------
    by_pos_setup = summarize_df(df, group_cols=["position", "setup"], value_cols=NUM_VARS, ci=0.95)
    by_pos_setup.to_csv(os.path.join(OUTDIR, "descriptives_by_position_setup.csv"), index=False)

    # -----------------------------------------
    # 3) By position × setup × calibration (using calibration_loss as a generic label)
    #    Useful for "data loss" analyses.
    # -----------------------------------------
    df_loss = df.copy()
    df_loss["calibration"] = df_loss["calibration_loss"]
    df_loss["calibration"] = pd.Categorical(df_loss["calibration"], categories=CALIB_ORDER, ordered=True)

    by_pos_setup_calib = summarize_df(
        df_loss,
        group_cols=["position", "setup", "calibration"],
        value_cols=["loss_pct"],
        ci=0.95,
    )
    by_pos_setup_calib.to_csv(os.path.join(OUTDIR, "descriptives_by_position_setup_calib.csv"), index=False)

    # -----------------------------------------
    # 4) Scene-specific calibration descriptives:
    #    For table_pct and screen_pct, it’s cleaner to use scene-specific calibration labels.
    # -----------------------------------------
    scene_rows = []
    for scene in ["table", "screen"]:
        calib_col = CALIB_COL_BY_SCENE[scene]
        tmp = df.copy()
        tmp["scene"] = scene
        tmp["calibration"] = pd.Categorical(tmp[calib_col], categories=CALIB_ORDER, ordered=True)
        tmp["value"] = tmp[f"{scene}_pct"]
        scene_rows.append(tmp[["participant", "position", "setup", "scene", "calibration", "value"]])

    df_scene = pd.concat(scene_rows, ignore_index=True)

    by_pos_setup_calib_scene = summarize_df(
        df_scene,
        group_cols=["position", "setup", "scene", "calibration"],
        value_cols=["value"],
        ci=0.95,
    )
    by_pos_setup_calib_scene.to_csv(
        os.path.join(OUTDIR, "descriptives_by_position_setup_calib_scene.csv"),
        index=False,
    )

    print("[OK] Wrote:")
    print(" - descriptives_overall.csv")
    print(" - descriptives_by_position_setup.csv")
    print(" - descriptives_by_position_setup_calib.csv")
    print(" - descriptives_by_position_setup_calib_scene.csv")


if __name__ == "__main__":
    main()
