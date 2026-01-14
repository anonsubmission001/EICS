#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plots/calibration.py

Calibration score distribution plots (counts of participants per calibration level),
split by position and device/setup, matching the intent of reliability.py::calibration_score
but with a clean implementation and outputs.

Input:
- {utils.DATASET}/setup/participants.csv

Outputs (PNG):
- ../results/analysis/plots/calibration_score_{position}.png

Notes:
- For head_mounted: use 'pupil'
- For remote: use 'tobii' and 'fovio'
- Categories ordered as in utils.SCORES (impossible -> ... -> no_issue)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils


OUTDIR = "../results/analysis/plots"
OUTPNG = os.path.join(OUTDIR, "calibration_score_{position}.png")


# Keep a stable order for bars
SCORE_ORDER = list(utils.SCORES)  # ['impossible','severe_issues','slight_issues','no_issue']
SCORE_LABELS = [utils.SCORES_LEGEND[s] for s in SCORE_ORDER]


def _ensure_outdir() -> None:
    os.makedirs(OUTDIR, exist_ok=True)


def _device_columns_for_setup(setup: str) -> list[str]:
    if setup == "head_mounted":
        return ["pupil"]
    return ["tobii", "fovio"]


def plot_calibration_for_position(df: pd.DataFrame, position: str) -> None:
    dpos = df[df["position"] == position].copy()
    if dpos.empty:
        return

    # We will plot counts per score level for each device (pupil/tobii/fovio)
    devices = []
    counts = {}

    for setup in utils.SETUPS:
        dsetup = dpos[dpos["setup"] == setup]
        if dsetup.empty:
            continue

        for dev in _device_columns_for_setup(setup):
            devices.append(dev)
            # count each score level
            c = []
            for score in SCORE_ORDER:
                c.append(int((dsetup[dev] == score).sum()))
            counts[dev] = c

    if not devices:
        return

    x = np.arange(len(SCORE_ORDER))
    width = 0.8 / max(1, len(devices))  # keep total bar span reasonable

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, dev in enumerate(devices):
        offset = (i - (len(devices) - 1) / 2) * width
        ax.bar(x + offset, counts[dev], width=width, alpha=0.75, label=utils.DEVICES_LEGEND.get(dev, dev))

    ax.set_ylabel("#Participants")
    ax.set_xlabel("Calibration quality")
    ax.set_title(f"Calibration score distribution ({position})")
    ax.set_xticks(x)
    ax.set_xticklabels(SCORE_LABELS)
    ax.set_ylim(0, 20)

    ax.legend(loc="upper left")
    fig.tight_layout()

    fig.savefig(OUTPNG.format(position=position))
    plt.close(fig)


def main() -> None:
    _ensure_outdir()

    participants_path = f"{utils.DATASET}/setup/participants.csv"
    df = pd.read_csv(participants_path)

    # Ensure expected columns exist
    for col in ["id", "position", "setup"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {participants_path}")

    for pos in utils.POSITIONS:
        plot_calibration_for_position(df, pos)

    print("[OK] Calibration plots saved to:", OUTDIR)


if __name__ == "__main__":
    main()
