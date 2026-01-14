#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_heatmap.py

Compute and plot mean gaze heatmaps per:
- scene (table / screen)
- position (sitting / standing)
- setup (remote / head_mounted)

This is a clean version of reliability.py::scene_cover.

Input:
- {utils.DATASET}/setup/participants.csv
- {utils.DATASET}/participants/{id}/{figure}/{scene}/gazepoints.csv

Output:
- ../results/analysis/plots/heatmap_{scene}.png

Method:
- For each participant, build one Heatmap per (scene, figure)
- Merge figures → participant-level heatmap
- Average heatmaps across participants (same position × setup)
"""

from __future__ import annotations

import os
import math
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt

import utils
from heatmap import Heatmap, map_mean, map_viz


OUTDIR = "../results/analysis/plots"
OUTPNG = os.path.join(OUTDIR, "heatmap_{scene}.png")


def _ensure_outdir() -> None:
    os.makedirs(OUTDIR, exist_ok=True)


def participant_heatmaps(participant_id: str, figures: List[str]) -> Dict[str, Heatmap]:
    """
    Build one Heatmap per scene for a single participant,
    aggregating across all available figures.
    """
    maps = {sc: Heatmap() for sc in utils.SCENES}

    any_data = False
    for f in figures:
        base = f"{utils.DATASET}/participants/{participant_id}/{f}"
        for sc in utils.SCENES:
            path = f"{base}/{sc}/gazepoints.csv"
            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)
            for _, r in df.iterrows():
                x = float(r["x"])
                y = float(r["y"])
                if math.isnan(x) or math.isnan(y):
                    continue
                maps[sc].add_point(x, y)
                any_data = True

    if not any_data:
        return {}

    return maps


def collect_maps(df_participants: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, List]]]:
    """
    maps[scene][position][setup] = list of participant heatmaps (percent arrays)
    """
    maps = {
        sc: {p: {s: [] for s in utils.SETUPS} for p in utils.POSITIONS}
        for sc in utils.SCENES
    }

    for _, row in df_participants.iterrows():
        pid = str(row["id"])
        position = str(row["position"])
        setup = str(row["setup"])
        figures = [f for f in utils.FIGURES if float(row.get(f, 0)) > 0]

        if not figures:
            continue

        ph = participant_heatmaps(pid, figures)
        if not ph:
            continue

        for sc, hm in ph.items():
            maps[sc][position][setup].append(hm.get_percent())

    return maps


def plot_scene(scene: str, maps_scene: Dict[str, Dict[str, List]]) -> None:
    """
    One figure per scene:
    columns = position
    rows = setup
    """
    fig, axes = plt.subplots(
        nrows=len(utils.SETUPS),
        ncols=len(utils.POSITIONS),
        figsize=(6 * len(utils.POSITIONS), 4 * len(utils.SETUPS)),
        squeeze=False,
    )

    for i, setup in enumerate(utils.SETUPS):
        for j, position in enumerate(utils.POSITIONS):
            ax = axes[i][j]
            data = maps_scene[position][setup]

            if not data:
                ax.axis("off")
                continue

            mean_map = map_mean(data)
            map_viz(mean_map, ax=ax)

            ax.set_title(f"{utils.SETUPS_LEGEND[setup]} – {position.capitalize()}")

    fig.suptitle(f"Gaze coverage heatmap ({scene})", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPNG.format(scene=scene))
    plt.close(fig)


def main() -> None:
    _ensure_outdir()

    participants_path = f"{utils.DATASET}/setup/participants.csv"
    df = pd.read_csv(participants_path)

    # keep only participants with at least one figure
    has_any = False
    mask = []
    for _, row in df.iterrows():
        ok = any(float(row.get(f, 0)) > 0 for f in utils.FIGURES)
        mask.append(ok)
        has_any |= ok
    df = df[pd.Series(mask)].copy()

    if df.empty:
        print("[WARN] No participant with gaze data.")
        return

    maps = collect_maps(df)

    for sc in utils.SCENES:
        plot_scene(sc, maps[sc])

    print("[OK] Heatmaps written to:", OUTDIR)


if __name__ == "__main__":
    main()
