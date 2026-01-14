#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_accuracy.py

Compute + plot gaze-to-block distance ("accuracy") as a function of time before an event
(grasp / release), split by position and setup.

This is a cleaned, reproducible version of the logic in acc.py / table_accuracy.py:
- Unit of analysis for the curve: participant-level mean distance per time bin
  (so we don't artificially inflate N with gazepoints).
- Then we aggregate across participants (mean + 95% CI) per bin.

Inputs:
- {utils.DATASET}/setup/participants.csv
- {utils.DATASET}/participants/{id}/{figure}/table/events.csv
- {utils.DATASET}/participants/{id}/{figure}/table/gazepoints.csv
- {utils.DATASET}/participants/{id}/{figure}/table/states.csv

Outputs (PNG):
- ../results/analysis/plots/accuracy_time_{position}_table_{event}.png

Requires:
- position.py (Point, Position)
- utils.py (constants)
- utils_stats.py (confidence_interval_mean)  # ensure Python 3.8 fix applied

Notes:
- MAX_TIME = 2000ms window before event.
- time bins default 10ms (same as your earlier scripts).
"""

from __future__ import annotations

import os
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from position import Point, Position
import utils
import utils_stats as ustats  # ensure Python 3.8 fix applied (no "|" unions)


# -----------------------------
# Config
# -----------------------------

MAX_TIME = 2000          # ms before event
BIN_MS = 10              # time binning (ms)
WIDTH_MM = 760
HEIGHT_MM = 380

OUTDIR = "../results/analysis/plots"
OUTPNG = os.path.join(OUTDIR, "accuracy_time_{position}_table_{event}.png")

EVENTS = ["grasp", "release"]


# -----------------------------
# Helpers
# -----------------------------

def _ensure_outdir() -> None:
    os.makedirs(OUTDIR, exist_ok=True)


def round_to(n: int, precision: int = BIN_MS) -> int:
    return int(round(n / precision) * precision)


def get_grasped_block_position(df_states: pd.DataFrame, ts_event: int, block_id: int,
                               width: int = WIDTH_MM, height: int = HEIGHT_MM) -> Position | None:
    p = None
    for i in df_states.index:
        ts = int(df_states.at[i, "timestamp"])
        if ts < ts_event:
            p = Position(
                Point(float(df_states.at[i, f"{block_id}_x0"]) * width, float(df_states.at[i, f"{block_id}_y0"]) * height),
                Point(float(df_states.at[i, f"{block_id}_x1"]) * width, float(df_states.at[i, f"{block_id}_y1"]) * height),
                Point(float(df_states.at[i, f"{block_id}_x2"]) * width, float(df_states.at[i, f"{block_id}_y2"]) * height),
                Point(float(df_states.at[i, f"{block_id}_x3"]) * width, float(df_states.at[i, f"{block_id}_y3"]) * height),
            )
        else:
            break
    return p


def get_released_block_position(df_states: pd.DataFrame, ts_event: int, block_id: int,
                                width: int = WIDTH_MM, height: int = HEIGHT_MM) -> Position | None:
    p = None
    for i in df_states.index:
        ts = int(df_states.at[i, "timestamp"])
        if ts <= ts_event:
            p = Position(
                Point(float(df_states.at[i, f"{block_id}_x0"]) * width, float(df_states.at[i, f"{block_id}_y0"]) * height),
                Point(float(df_states.at[i, f"{block_id}_x1"]) * width, float(df_states.at[i, f"{block_id}_y1"]) * height),
                Point(float(df_states.at[i, f"{block_id}_x2"]) * width, float(df_states.at[i, f"{block_id}_y2"]) * height),
                Point(float(df_states.at[i, f"{block_id}_x3"]) * width, float(df_states.at[i, f"{block_id}_y3"]) * height),
            )
        else:
            break
    return p


def iter_distances_in_window(df_gz: pd.DataFrame, start_ts: int, event_ts: int, position: Position,
                             width: int = WIDTH_MM, height: int = HEIGHT_MM) -> Dict[int, float]:
    """
    Returns distances indexed by dt = (event_ts - gaze_ts) in ms, for gaze points within:
      max(start_ts, event_ts - MAX_TIME) <= gaze_ts <= event_ts
    """
    out: Dict[int, float] = {}

    lower = max(start_ts, event_ts - MAX_TIME)

    for i in df_gz.index:
        ts_gz = int(df_gz.at[i, "timestamp"])
        if ts_gz < lower:
            continue
        if ts_gz > event_ts:
            break

        x = float(df_gz.at[i, "x"])
        y = float(df_gz.at[i, "y"])
        if math.isnan(x) or math.isnan(y):
            continue

        gaze = Point(x * width, y * height)
        dt = event_ts - ts_gz
        out[dt] = position.distance(gaze)

    return out


def process_participant(participant_id: str, position_p: str, setup: str, figures: List[str]) -> Dict[str, Dict[int, float]]:
    """
    Returns participant-level mean distance per time bin, separately for each event type.
    Output:
      { "grasp": {bin_ms: mean_distance}, "release": {...} }
    """
    # collect per-event per-bin list of distances
    bins: Dict[str, Dict[int, List[float]]] = {e: {} for e in EVENTS}

    for f in figures:
        base = f"{utils.DATASET}/participants/{participant_id}/{f}/table"
        events_path = os.path.join(base, "events.csv")
        gz_path = os.path.join(base, "gazepoints.csv")
        states_path = os.path.join(base, "states.csv")

        if not (os.path.exists(events_path) and os.path.exists(gz_path) and os.path.exists(states_path)):
            continue

        df_event = pd.read_csv(events_path)
        df_gz = pd.read_csv(gz_path)
        df_states = pd.read_csv(states_path)

        if df_event.empty or df_gz.empty or df_states.empty:
            continue

        prev_ts = 0
        for _, er in df_event.iterrows():
            ev = str(er["event"])
            ts = int(er["timestamp"])

            if ev == "start":
                prev_ts = ts
                continue
            if ev not in EVENTS:
                continue

            block_id = int(er["block_id"])
            pos_obj = get_grasped_block_position(df_states, ts, block_id) if ev == "grasp" else get_released_block_position(df_states, ts, block_id)
            if pos_obj is None:
                prev_ts = ts
                continue

            distances = iter_distances_in_window(df_gz, prev_ts, ts, pos_obj)
            for dt, dist in distances.items():
                b = round_to(dt, BIN_MS)
                bins[ev].setdefault(b, []).append(float(dist))

            prev_ts = ts

    # participant-level mean per bin
    out: Dict[str, Dict[int, float]] = {e: {} for e in EVENTS}
    for ev in EVENTS:
        for b, values in bins[ev].items():
            if values:
                out[ev][b] = float(np.mean(np.asarray(values, dtype=float)))
    return out


def aggregate_across_participants(participant_curves: List[Dict[int, float]]) -> Dict[int, Tuple[float, float, float, int]]:
    """
    Aggregate participant curves into (mean, ci_low, ci_high, n_participants_with_value) per bin.
    """
    # bin -> list of participant means
    per_bin: Dict[int, List[float]] = {}
    for curve in participant_curves:
        for b, val in curve.items():
            per_bin.setdefault(b, []).append(float(val))

    agg: Dict[int, Tuple[float, float, float, int]] = {}
    for b, vals in per_bin.items():
        arr = np.asarray(vals, dtype=float)
        m, lo, hi = ustats.confidence_interval_mean(arr, confidence=0.95)
        agg[b] = (float(m), float(lo), float(hi), int(arr.size))
    return agg


# -----------------------------
# Plotting
# -----------------------------

def plot_accuracy(df_participants: pd.DataFrame, event: str, position: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))

    # reference block sizes (mm)
    ax.plot([0, 1000, 2000], [16, 16, 16], color="black", linestyle="solid", label="1x1 block")
    ax.plot([0, 1000, 2000], [32, 32, 32], color="black", linestyle="dashed", label="2x2 block")
    ax.plot([0, 1000, 2000], [64, 64, 64], color="black", linestyle="dotted", label="4x4 block")

    for setup in utils.SETUPS:
        sdf = df_participants[(df_participants["position"] == position) & (df_participants["setup"] == setup)].copy()
        if sdf.empty:
            continue

        curves: List[Dict[int, float]] = []
        for _, row in sdf.iterrows():
            pid = str(row["id"])
            figures = [f for f in utils.FIGURES if float(row.get(f, 0)) > 0]
            pcurve = process_participant(pid, position, setup, figures)[event]
            if pcurve:
                curves.append(pcurve)

        if not curves:
            continue

        agg = aggregate_across_participants(curves)

        xs = sorted(agg.keys())
        mean = np.array([agg[x][0] for x in xs], dtype=float)
        lo = np.array([agg[x][1] for x in xs], dtype=float)
        hi = np.array([agg[x][2] for x in xs], dtype=float)

        # Error bars as asymmetric distances from mean
        yerr = np.vstack([mean - lo, hi - mean])
        ax.errorbar(xs, mean, yerr=yerr, fmt="o", label=utils.SETUPS_LEGEND.get(setup, setup))

    ax.set_xlim(MAX_TIME, 0)
    ax.set_ylim(0, 350)
    ax.set_xlabel(f"Time before {event} event (ms)")
    ax.set_ylabel("Distance (mm)")
    ax.set_title(f"Accuracy over time ({position}, table scene, {event})")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPNG.format(position=position, event=event))
    plt.close(fig)


def main() -> None:
    _ensure_outdir()

    participants_path = f"{utils.DATASET}/setup/participants.csv"
    df = pd.read_csv(participants_path)

    # basic filtering (only participants with any figure)
    has_any = np.zeros(len(df), dtype=bool)
    for f in utils.FIGURES:
        if f in df.columns:
            has_any |= (pd.to_numeric(df[f], errors="coerce").fillna(0) > 0).to_numpy()
    df = df[has_any].copy()

    for event in EVENTS:
        for position in utils.POSITIONS:
            plot_accuracy(df, event=event, position=position)

    print("[OK] Wrote accuracy plots to:", OUTDIR)


if __name__ == "__main__":
    main()
