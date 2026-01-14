#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_data_loss_points_by_calib.py

Bar + participant points (jitter) for sample repartition:
Table / Screen / Unrelevant(loss), split by position and setup,
with points *colored by calibration level*.

Input:
- ../results/analysis/data_prepared.csv

Outputs:
- ../results/analysis/plots/data_loss_points_by_calib_{position}.png

Plot:
- Bars: mean % across participants (per setup)
- Error bars: 95% CI (Student-t)
- Points: each participant value (jittered), colored by calibration level

Calibration used:
- loss_pct points colored by calibration_loss
- table_pct points colored by calibration_table
- screen_pct points colored by calibration_screen
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
import utils_stats as ustats  # ensure Python 3.8 fix applied


INDIR = "../results/analysis"
INCSV = os.path.join(INDIR, "data_prepared.csv")
OUTDIR = os.path.join(INDIR, "plots")
OUTPNG = os.path.join(OUTDIR, "data_loss_points_by_calib_{position}.png")

# Categories: (value_col, label, calib_col)
CATS = [
    ("table_pct", "Table", "calibration_table"),
    ("screen_pct", "Screen", "calibration_screen"),
    ("loss_pct", "Unrelevant", "calibration_loss"),
]

CALIB_ORDER = ["no_issue", "slight_issues", "severe_issues", "impossible"]
CALIB_LABEL = {k: utils.SCORES_LEGEND.get(k, k) for k in CALIB_ORDER}

JITTER_STD = 0.06
POINT_SIZE = 22
POINT_ALPHA = 0.55


def _ensure_outdir() -> None:
    os.makedirs(OUTDIR, exist_ok=True)


def _ci95(series: pd.Series) -> tuple[np.ndarray, float, float, float]:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    arr = arr[~np.isnan(arr)]
    m, lo, hi = ustats.confidence_interval_mean(arr, confidence=0.95)
    return arr, m, lo, hi


def _calib_color_map(levels: list[str]) -> dict[str, float]:
    """
    Map calibration levels to a numeric scalar in [0,1] so we can use a colormap
    without hardcoding colors.
    """
    levels = [l for l in levels if l in CALIB_ORDER]
    out = {}
    for lvl in levels:
        out[lvl] = CALIB_ORDER.index(lvl) / max(1, (len(CALIB_ORDER) - 1))
    return out


def plot_position(df: pd.DataFrame, position: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)

    dpos = df[df["position"] == position].copy()
    if dpos.empty:
        return

    x = np.arange(len(CATS))
    width = 0.8 / len(utils.SETUPS)

    fig, ax = plt.subplots(figsize=(12, 5))

    # For a consistent mapping, consider all calibration levels present in this position
    calib_levels_present = set()
    for _vcol, _lab, ccol in CATS:
        calib_levels_present |= set(dpos[ccol].dropna().astype(str).unique().tolist())
    calib_levels_present = sorted(list(calib_levels_present), key=lambda z: CALIB_ORDER.index(z) if z in CALIB_ORDER else 999)

    cmap = plt.get_cmap("viridis")
    calib_to_scalar = _calib_color_map([lvl for lvl in calib_levels_present if lvl in CALIB_ORDER])

    for i, setup in enumerate(utils.SETUPS):
        d = dpos[dpos["setup"] == setup]
        if d.empty:
            continue

        means = []
        yerr_low = []
        yerr_high = []

        offset = (i - (len(utils.SETUPS) - 1) / 2) * width
        centers = x + offset

        for j, (val_col, _label, calib_col) in enumerate(CATS):
            arr, m, lo, hi = _ci95(d[val_col])
            means.append(m)
            yerr_low.append(m - lo if not np.isnan(lo) else np.nan)
            yerr_high.append(hi - m if not np.isnan(hi) else np.nan)

            # participant points
            dd = d[[val_col, calib_col]].copy()
            dd[val_col] = pd.to_numeric(dd[val_col], errors="coerce")
            dd = dd[~dd[val_col].isna()]

            if dd.empty:
                continue

            jitter = rng.normal(loc=0.0, scale=JITTER_STD, size=len(dd))
            xs = np.full((len(dd),), centers[j]) + jitter
            ys = dd[val_col].to_numpy(dtype=float)
            cl = dd[calib_col].astype(str).to_list()

            # Convert calibration level -> scalar for colormap
            cs = []
            for lvl in cl:
                if lvl in calib_to_scalar:
                    cs.append(calib_to_scalar[lvl])
                else:
                    cs.append(np.nan)

            sc = ax.scatter(
                xs,
                ys,
                s=POINT_SIZE,
                alpha=POINT_ALPHA,
                c=cs,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
            )

        ax.bar(
            centers,
            means,
            width=width,
            alpha=0.55,
            yerr=[yerr_low, yerr_high],
            capsize=5,
            label=utils.SETUPS_LEGEND.get(setup, setup),
        )

    ax.set_ylabel("Samples (%)")
    ax.set_title(f"Sample repartition by setup with calibration-colored points ({position})")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _v, lab, _c in CATS])
    ax.set_ylim(0, 100)
    ax.legend(loc="best")

    # Add a colorbar with calibration ticks (only for known levels present)
    known_levels = [lvl for lvl in calib_levels_present if lvl in calib_to_scalar]
    if known_levels:
        ticks = [calib_to_scalar[lvl] for lvl in known_levels]
        ticklabels = [CALIB_LABEL.get(lvl, lvl) for lvl in known_levels]
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticklabels)
        cbar.set_label("Calibration level")

    fig.tight_layout()
    fig.savefig(OUTPNG.format(position=position))
    plt.close(fig)


def main() -> None:
    _ensure_outdir()

    df = pd.read_csv(INCSV)
    df = df[df["n_timestamps_total"].fillna(0).astype(int) > 0].copy()

    df["position"] = pd.Categorical(df["position"], categories=utils.POSITIONS, ordered=True)
    df["setup"] = pd.Categorical(df["setup"], categories=utils.SETUPS, ordered=True)

    for pos in utils.POSITIONS:
        plot_position(df, pos, seed=42)

    print("[OK] Wrote plots to:", OUTDIR)


if __name__ == "__main__":
    main()
