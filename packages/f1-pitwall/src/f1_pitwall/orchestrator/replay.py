"""Replay engine (spec §3.1): a historical race as a timed, typed event stream.

Events are the only interface — a real live API can be swapped in later by
emitting the same dataclasses. Session time is reconstructed from cumulative
lap times per driver (the processed lap table drops absolute session time);
missing lap times are filled with the driver's median lap. Good enough for
event ordering; documented approximation.

``speed=None`` replays as fast as possible (no sleeps); otherwise wall-clock
gaps between events are divided by ``speed``.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LapCompleted:
    session_time_s: float
    driver: str
    lap_number: int
    lap_time_s: float | None
    position: float | None
    compound: str | None
    tyre_age: float | None
    stint_number: float | None
    stint_lap: int
    is_clean: bool


@dataclass(frozen=True)
class TelemetryFrame:
    """One lap's mini-sector proxy table (the imaging input for that lap)."""

    session_time_s: float
    driver: str
    lap_number: int
    sectors: pd.DataFrame


RaceEvent = LapCompleted | TelemetryFrame


def replay_race(
    laps: pd.DataFrame,
    sectors: pd.DataFrame | None = None,
    speed: float | None = None,
) -> Iterator[RaceEvent]:
    """Yield race events in session-time order for one event's lap table."""
    laps = laps.sort_values(["driver", "lap_number"]).copy()
    fill = laps.groupby("driver")["lap_time_s"].transform("median")
    laps["session_t"] = laps["lap_time_s"].fillna(fill).groupby(laps["driver"]).cumsum()
    laps = laps.sort_values("session_t")

    sector_groups: dict[tuple[str, int], pd.DataFrame] = {}
    if sectors is not None:
        sector_groups = {
            (str(d), int(n)): g for (d, n), g in sectors.groupby(["driver", "lap_number"])
        }

    start = time.monotonic()
    for row in laps.itertuples(index=False):
        t = float(row.session_t)
        if speed is not None:
            lag = start + t / speed - time.monotonic()
            if lag > 0:
                time.sleep(lag)
        key = (str(row.driver), int(row.lap_number))
        if key in sector_groups:
            yield TelemetryFrame(t, key[0], key[1], sector_groups[key])
        yield LapCompleted(
            session_time_s=t,
            driver=str(row.driver),
            lap_number=int(row.lap_number),
            lap_time_s=None if pd.isna(row.lap_time_s) else float(row.lap_time_s),
            position=None if pd.isna(row.position) else float(row.position),
            compound=None if pd.isna(row.compound) else str(row.compound),
            tyre_age=None if pd.isna(row.tyre_age) else float(row.tyre_age),
            stint_number=None if pd.isna(row.stint_number) else float(row.stint_number),
            stint_lap=int(row.stint_lap),
            is_clean=bool(row.is_clean),
        )
