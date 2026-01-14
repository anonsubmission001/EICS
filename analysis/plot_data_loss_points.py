#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_data_loss_points.py

Bar + participant-level points (jitter) for sample repartition:
Table / Screen / Unrelevant(loss), split by position and setup.

Input:
- ../results/analysis/data_prepared.csv

Outputs:
- ../results/analysis/plots/data_loss_points_{position}.png

Plot:
- Bars: mean % across participants
- Error bars: 95% CI (Student-t)
- Points: each participant value (jittered), semi-transparent

This is useful to show dispersion + potential outliers.
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
OUTPNG = os.path.join(OUTDIR, "data_loss_points_{position}.png")

CATS = [
    ("table_pct", "Table"),
    ("screen_pct", "Screen"),
    ("loss_pct", "Unrelevant"),
]

JITTER_STD = 0.06   # horizontal jitter strength
POINT_SIZE = 18     # scatter size
POINT_ALPHA = 0.35  # scatter transparency


def _ensure_outdir() -> None:
    os.makedirs(OUTDIR, exist_ok=True)


def _ci95(series: pd.Series) -> tuple[np.ndarray, float, float, float]:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    arr = arr[~np.isnan(arr)]
    m, lo, hi = ustats.confidence_interval_mean(arr, confidence=0.95)
    return arr, m, lo, hi


def plot_position(df: pd.DataFrame, position: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)

    dpos = df[df["position"] == position].copy()
    if dpos.empty:
        return

    x = np.arange(len(CATS))
    width = 0.8 / len(utils.SETUPS)

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, setup in enumerate(utils.SETUPS):
        d = dpos[dpos["setup"] == setup]
        if d.empty:
            continue

        means = []
        yerr_low = []
        yerr_high = []

        # bar centers for this setup
        offset = (i - (len(utils.SETUPS) - 1) / 2) * width
        centers = x + offset

        for j, (col, _label) in enumerate(CATS):
            arr, m, lo, hi = _ci95(d[col])
            means.append(m)
            yerr_low.append(m - lo if not np.isnan(lo) else np.nan)
            yerr_high.append(hi - m if not np.isnan(hi) else np.nan)

            # participant points with jitter
            if arr.size > 0:
                jitter = rng.normal(loc=0.0, scale=JITTER_STD, size=arr.size)
                ax.scatter(
                    np.full(arr.shape, centers[j]) + jitter,
                    arr,
                    s=POINT_SIZE,
                    alpha=POINT_ALPHA,
                )

        ax.bar(
            centers,
            means,
            width=width,
            alpha=0.65,
            yerr=[yerr_low, yerr_high],
            capsize=5,
            label=utils.SETUPS_LEGEND.get(setup, setup),
        )

    ax.set_ylabel("Samples (%)")
    ax.set_title(f"Sample repartition by setup with participant points ({position})")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _col, lab in CATS])
    ax.set_ylim(0, 100)
    ax.legend(loc="best")
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
