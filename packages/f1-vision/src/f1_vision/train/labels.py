"""Prediction targets (spec §2.2), computed from the Phase 1 filtered lap table.

- ``pace_delta_next3``: mean clean-lap delta (s) vs the driver's own fresh-tire
  stint baseline over the NEXT ``next_laps`` laps. Predictive, not descriptive.
- ``cliff_within_5``: 1 if within the next ``cliff_horizon`` laps the clean-lap
  delta exceeds ``cliff_threshold_s`` OR the driver pits earlier than the median
  stop lap for that compound at that event.

Labels look into the future BY DESIGN and are only used offline for training;
no feature may be derived from them (spec §4, no-lookahead rule).

Baseline caveat (documented, per spec honesty rule): the fresh-tire baseline is
the mean of the stint's first ``baseline_laps`` clean laps, taken at high(er)
fuel — so late-stint deltas mix degradation with fuel burn-off, which partially
offsets it. The model's context channels (fuel_proxy) let it disentangle.
"""

from __future__ import annotations

import pandas as pd

from f1_vision.config import LabelConfig

LABEL_COLUMNS = ["pace_delta_next3", "cliff_within_5"]
#: Not a label: the most recent clean-lap delta at/before this lap. Feature-safe
#: (backward-looking); it exists for the persistence baseline (spec §2.5).
CURRENT_DELTA = "pace_delta_current"


def add_labels(laps: pd.DataFrame, config: LabelConfig) -> pd.DataFrame:
    """Return ``laps`` (all rows) with label columns; NaN where undefined.

    Requires Phase 1 output-schema columns. A lap gets labels only if its stint
    has a baseline (≥ ``baseline_laps`` clean laps) and at least one clean lap
    inside the pace horizon.
    """
    laps = laps.sort_values(["season", "event", "driver", "lap_number"]).copy()
    laps["pace_delta_next3"] = float("nan")
    laps["cliff_within_5"] = float("nan")
    laps[CURRENT_DELTA] = float("nan")

    race_end = laps.groupby(["season", "event"])["lap_number"].transform("max")
    laps["_race_end"] = race_end

    # Median stop lap per (event, compound), over stints that actually pitted.
    stints = (
        laps.groupby(["season", "event", "driver", "stint_number"])
        .agg(
            end_lap=("lap_number", "max"),
            compound=("compound", "first"),
            race_end=("_race_end", "first"),
        )
        .reset_index()
    )
    pitted = stints[stints["end_lap"] < stints["race_end"]]
    median_stop = pitted.groupby(["season", "event", "compound"])["end_lap"].median()

    for (season, event, _driver, _stint), stint_laps in laps.groupby(
        ["season", "event", "driver", "stint_number"], sort=False
    ):
        clean = stint_laps[stint_laps["is_clean"] & stint_laps["lap_time_s"].notna()]
        if len(clean) < config.baseline_laps:
            continue
        baseline = clean["lap_time_s"].iloc[: config.baseline_laps].mean()
        delta = clean["lap_time_s"] - baseline  # indexed like clean, keyed by lap_number below
        delta_by_lap = pd.Series(delta.to_numpy(), index=clean["lap_number"].to_numpy())

        end_lap = int(stint_laps["lap_number"].max())
        ends_in_pit = end_lap < int(stint_laps["_race_end"].iloc[0])
        med = median_stop.get((season, event, stint_laps["compound"].iloc[0]))
        early_pit = ends_in_pit and med is not None and end_lap < med

        # Backward-looking current delta (persistence baseline input).
        stint_lap_numbers = stint_laps["lap_number"].to_numpy()
        current = delta_by_lap.reindex(stint_lap_numbers).ffill()
        laps.loc[stint_laps.index, CURRENT_DELTA] = current.to_numpy()

        for idx, lap_n in stint_laps["lap_number"].items():
            pace_window = delta_by_lap[
                (delta_by_lap.index > lap_n) & (delta_by_lap.index <= lap_n + config.next_laps)
            ]
            if not pace_window.empty:
                laps.at[idx, "pace_delta_next3"] = pace_window.mean()

            cliff_window = delta_by_lap[
                (delta_by_lap.index > lap_n) & (delta_by_lap.index <= lap_n + config.cliff_horizon)
            ]
            pace_cliff = bool((cliff_window > config.cliff_threshold_s).any())
            pit_cliff = early_pit and end_lap <= lap_n + config.cliff_horizon
            if not cliff_window.empty or pit_cliff:
                laps.at[idx, "cliff_within_5"] = float(pace_cliff or pit_cliff)

    return laps.drop(columns="_race_end")
