"""Event-level train / calibration / test splits (PROJECT_SPEC.md §1.3).

Splitting is ALWAYS by (season, event) — never within a race — to prevent
leakage between laps of the same race.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventSplit:
    """Frames for each split; every (season, event) appears in exactly one."""

    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


def split_by_event(
    laps: pd.DataFrame,
    calibration_fraction: float,
    test_fraction: float,
    seed: int,
) -> EventSplit:
    """Randomly assign whole events to train/calibration/test.

    Fractions apply to the number of unique events (rounded, at least one event
    each for calibration and test when fractions are non-zero).
    """
    if calibration_fraction + test_fraction >= 1.0:
        raise ValueError("calibration_fraction + test_fraction must be < 1")

    events = laps[["season", "event"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(events))

    n_test = max(1, round(len(events) * test_fraction)) if test_fraction > 0 else 0
    n_calib = max(1, round(len(events) * calibration_fraction)) if calibration_fraction > 0 else 0
    if n_test + n_calib >= len(events):
        raise ValueError(
            f"Not enough events ({len(events)}) for calib={n_calib} + test={n_test} "
            "with at least one training event."
        )

    test_events = events.iloc[order[:n_test]]
    calib_events = events.iloc[order[n_test : n_test + n_calib]]
    train_events = events.iloc[order[n_test + n_calib :]]

    def take(selected: pd.DataFrame) -> pd.DataFrame:
        merged = laps.merge(selected, on=["season", "event"], how="inner")
        return merged.reset_index(drop=True)

    return EventSplit(
        train=take(train_events), calibration=take(calib_events), test=take(test_events)
    )
