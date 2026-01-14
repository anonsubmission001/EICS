#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_data_loss.py

Plot participant-level sample repartition (Table / Screen / Unrelevant(loss)),
split by position and setup, using the analysis-ready dataset produced by
prepare_data.py.

Input:
- ../results/analysis/data_prepared.csv

Outputs (PNG):
- ../results/analysis/plots/data_loss_{position}.png

Figure:
- Bars = mean percentage across participants
- Error bars = 95% CI on the mean (Student-t)
- Groups (colors) = setup (Remote vs Head Mounted)
- Categories = Table, Screen, Unrelevant

This replaces the older reliability.py::data_loss logic with a clean,
participant-level aggregation (no per-gazepoint inflation).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
import utils_stats as ustats  # ensure Python 3.8 fix applied (no "|" unions)


INDIR = "../results/analysis"
INCSV = os.path.join(INDIR, "data_prepared.csv")
OUTDIR = os.path.join(INDIR, "plots")
OUTPNG = os.path.join(OUTDIR, "data_loss_{position}.png")

CATS = [
    ("table_pct", "Table"),
    ("screen_pct", "Screen"),
    ("loss_pct", "Unrelevant"),
]


def _ensure_outdir() -> None:
    os.makedirs(OUTDIR, exist_ok=True)


def _ci95(series: pd.Series) -> tuple[float, float, float]:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    arr = arr[~np.isnan(arr)]
    return ustats.confidence_interval_mean(arr, confidence=0.95)


def plot_position(df: pd.DataFrame, position: str) -> None:
    dpos = df[df["position"] == position].copy()
    if dpos.empty:
        return

    x = np.arange(len(CATS))
    width = 0.8 / len(utils.SETUPS)

    fig, ax = plt.subplots(figsize=(5, 5))

    for i, setup in enumerate(utils.SETUPS):
        d = dpos[dpos["setup"] == setup]
        if d.empty:
            continue

        means = []
        yerr_low = []
        yerr_high = []

        for col, _label in CATS:
            m, lo, hi = _ci95(d[col])
            means.append(m)
            # yerr expects distances from mean
            yerr_low.append(m - lo if not np.isnan(lo) else np.nan)
            yerr_high.append(hi - m if not np.isnan(hi) else np.nan)

        offset = (i - (len(utils.SETUPS) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            alpha=0.75,
            yerr=[yerr_low, yerr_high],
            capsize=5,
            label=utils.SETUPS_LEGEND.get(setup, setup),
        )

    ax.set_ylabel("Samples (%)")
    ax.set_title(f"Sample repartition by setup ({position})")
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

    # Basic filtering
    df = df[df["n_timestamps_total"].fillna(0).astype(int) > 0].copy()
    df["position"] = pd.Categorical(df["position"], categories=utils.POSITIONS, ordered=True)
    df["setup"] = pd.Categorical(df["setup"], categories=utils.SETUPS, ordered=True)

    for pos in utils.POSITIONS:
        plot_position(df, pos)

    print("[OK] Wrote plots to:", OUTDIR)


if __name__ == "__main__":
    main()
