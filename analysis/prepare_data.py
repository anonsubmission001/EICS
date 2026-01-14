#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_data.py

Build a clean, analysis-ready dataset (one row per participant) for reliability analyses.

Inputs (expected):
- {utils.DATASET}/setup/participants.csv
- {utils.DATASET}/participants/{id}/{figure}/{scene}/gazepoints.csv
  with columns: timestamp, x, y

Outputs:
- ../results/analysis/data_prepared.csv

What we compute per participant:
- position, setup
- calibration labels (table/screen/loss) derived from device columns
- sample proportions (%): table_pct, screen_pct, loss_pct
  using the same union-of-timestamps logic as reliability.py
- optional: share among valid gaze only: table_share_valid, screen_share_valid
- bookkeeping: n_figures_used, n_timestamps_total
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import utils


OUTDIR = "../results/analysis"
OUTCSV = os.path.join(OUTDIR, "data_prepared.csv")


# -----------------------------
# Calibration mapping
# -----------------------------

def calibration_for_participant(row: pd.Series) -> dict[str, str]:
    """
    Map participant calibration quality per scene/loss.

    Mirrors logic used in reliability3.py:
    - head_mounted uses 'pupil' for both table & screen (and loss)
    - remote uses 'fovio' for table/loss and 'tobii' for screen
    """
    setup = str(row["setup"])
    if setup == "head_mounted":
        return {
            "table": str(row.get("pupil", "impossible")),
            "screen": str(row.get("pupil", "impossible")),
            "loss": str(row.get("pupil", "impossible")),
        }
    # remote
    return {
        "table": str(row.get("fovio", "impossible")),
        "screen": str(row.get("tobii", "impossible")),
        "loss": str(row.get("fovio", "impossible")),
    }


# -----------------------------
# Gaze aggregation (participant-level)
# -----------------------------

def _read_scene_gazepoints(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    # minimal schema check
    needed = {"timestamp", "x", "y"}
    if not needed.issubset(set(df.columns)):
        return None
    return df


def compute_sample_partitions(participant_id: str, figures: list[str]) -> tuple[float, float, float, float, float, int, int]:
    """
    Returns:
    - table_pct (% over all timestamps in union)
    - screen_pct (% over all timestamps in union)
    - loss_pct (% over all timestamps in union)
    - table_share_valid (% among valid samples only)
    - screen_share_valid (% among valid samples only)
    - n_figures_used
    - n_timestamps_total
    """
    # Timestamp -> [table_has, screen_has, is_loss]
    # is_loss starts True and is set False if any scene has a valid gaze sample at that timestamp.
    d: Dict[int, list[bool]] = {}

    n_fig_used = 0

    for f in figures:
        base = f"{utils.DATASET}/participants/{participant_id}/{f}"

        any_file_found = False
        for scene in utils.SCENES:  # ['table', 'screen']
            gp_path = f"{base}/{scene}/gazepoints.csv"
            df_scene = _read_scene_gazepoints(gp_path)
            if df_scene is None:
                continue
            any_file_found = True

            scene_idx = utils.SCENES.index(scene)
            for _, r in df_scene.iterrows():
                ts = int(r["timestamp"])
                x = float(r["x"])
                y = float(r["y"])

                if ts not in d:
                    d[ts] = [False, False, True]  # table, screen, loss

                if not (np.isnan(x) or np.isnan(y)):
                    d[ts][scene_idx] = True
                    d[ts][2] = False

        if any_file_found:
            n_fig_used += 1

    if len(d) == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, n_fig_used, 0)

    loss = sum(1 for ts in d if d[ts][2])
    table = sum(1 for ts in d if d[ts][utils.SCENES.index("table")])
    screen = sum(1 for ts in d if d[ts][utils.SCENES.index("screen")])

    total = loss + table + screen
    if total <= 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, n_fig_used, len(d))

    table_pct = 100.0 * table / total
    screen_pct = 100.0 * screen / total
    loss_pct = 100.0 * loss / total

    valid = table + screen
    if valid > 0:
        table_share_valid = 100.0 * table / valid
        screen_share_valid = 100.0 * screen / valid
    else:
        table_share_valid = np.nan
        screen_share_valid = np.nan

    return (table_pct, screen_pct, loss_pct, table_share_valid, screen_share_valid, n_fig_used, len(d))


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)

    participants_path = f"{utils.DATASET}/setup/participants.csv"
    df = pd.read_csv(participants_path)

    rows = []
    for _, row in df.iterrows():
        pid = str(row["id"])
        position = str(row.get("position", ""))
        setup = str(row.get("setup", ""))

        # figures included for that participant (same logic: skip <= 0)
        figures_used = [f for f in utils.FIGURES if float(row.get(f, 0)) > 0]

        cal = calibration_for_participant(row)

        # if "impossible" on required device columns, we can still compute loss,
        # but keeping a flag helps downstream filtering.
        calib_impossible = (
            (setup == "head_mounted" and cal["table"] == "impossible")
            or (setup != "head_mounted" and (cal["table"] == "impossible" or cal["screen"] == "impossible"))
        )

        table_pct, screen_pct, loss_pct, table_share_valid, screen_share_valid, n_fig_used, n_ts = (
            compute_sample_partitions(pid, figures_used)
        )

        rows.append(
            {
                "participant": pid,
                "position": position,
                "setup": setup,
                "calibration_table": cal["table"],
                "calibration_screen": cal["screen"],
                "calibration_loss": cal["loss"],
                "calibration_impossible": bool(calib_impossible),
                "table_pct": table_pct,
                "screen_pct": screen_pct,
                "loss_pct": loss_pct,
                "table_share_valid": table_share_valid,
                "screen_share_valid": screen_share_valid,
                "n_figures_used": int(n_fig_used),
                "n_timestamps_total": int(n_ts),
            }
        )
        print(pid)
    out = pd.DataFrame(rows)

    # mild hygiene: consistent category ordering (useful for stats/plots)
    out["position"] = pd.Categorical(out["position"], categories=utils.POSITIONS, ordered=True)
    out["setup"] = pd.Categorical(out["setup"], categories=utils.SETUPS, ordered=True)

    out.to_csv(OUTCSV, index=False)
    print(f"[OK] wrote {OUTCSV} with {len(out)} rows")


if __name__ == "__main__":
    main()
