"""Lap filtering rules (PROJECT_SPEC.md §1.2).

Each rule is a pure function ``(laps, config) -> boolean Series`` where True means
DROP. ``apply_filters`` runs them in spec order, flags survivors with ``is_clean``,
and produces the required per-race attrition report.

Input frame: the raw lap schema produced by ``f1_core.ingest.etl.session_to_laps``
(also the schema of the committed Bahrain 2024 fixture). Timedeltas are float
seconds; ``Time`` is cumulative session time at lap completion.

Traffic rule approximation: per-lap telemetry gaps are not available at lap
granularity, so the gap to the car ahead is computed from timing data — the
difference in cumulative ``Time`` to the car one classified position ahead on the
same lap, evaluated at both the end of the lap and the end of the previous lap
(≈ lap start). Lapped traffic between the two cars is not visible to this
approximation; the spec explicitly permits it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import pandas as pd

from f1_core.config import IngestConfig


def mask_traffic(laps: pd.DataFrame, config: IngestConfig) -> pd.Series[bool]:
    """Drop laps run within ``traffic_gap_s`` of the car ahead (dirty air / racing)."""
    d = laps[["Driver", "LapNumber", "Position", "Time"]].copy()
    d = d.sort_values(["LapNumber", "Position"])
    d["gap_end"] = d["Time"] - d.groupby("LapNumber")["Time"].shift(1)
    d = d.sort_values(["Driver", "LapNumber"])
    d["gap_start"] = d.groupby("Driver")["gap_end"].shift(1)
    close = (d["gap_end"] < config.traffic_gap_s) | (d["gap_start"] < config.traffic_gap_s)
    return close.reindex(laps.index).fillna(False)


def mask_in_out_lap(laps: pd.DataFrame, config: IngestConfig) -> pd.Series[bool]:
    """Drop pit in-laps and out-laps."""
    return laps["PitInTime"].notna() | laps["PitOutTime"].notna()


def mask_not_green(laps: pd.DataFrame, config: IngestConfig) -> pd.Series[bool]:
    """Drop laps with any non-green track status (SC/VSC/yellow/red).

    FastF1 ``TrackStatus`` concatenates every status seen during the lap ("1" is
    all-green). Missing status is treated as non-green (conservative).
    """
    status = laps["TrackStatus"].astype("string").fillna("")
    return status.map(lambda s: s == "" or any(ch != "1" for ch in s)).astype(bool)


def mask_first_lap(laps: pd.DataFrame, config: IngestConfig) -> pd.Series[bool]:
    """Drop lap 1 of the race (start chaos)."""
    return laps["LapNumber"].eq(1)


def mask_inaccurate(laps: pd.DataFrame, config: IngestConfig) -> pd.Series[bool]:
    """Drop deleted laps and laps FastF1 marks as not accurate."""
    accurate = laps["IsAccurate"].astype("boolean").fillna(False).astype(bool)
    deleted = laps["Deleted"].astype("boolean").fillna(False).astype(bool)
    return ~accurate | deleted


def mask_wet(laps: pd.DataFrame, config: IngestConfig) -> pd.Series[bool]:
    """Drop wet running: INTERMEDIATE/WET compound, or the whole session marked wet."""
    wet = ~laps["Compound"].isin(config.dry_compounds)
    if "SessionWet" in laps.columns:
        wet |= laps["SessionWet"].astype("boolean").fillna(False).astype(bool)
    return wet


#: Spec §1.2 rule order. Names appear in the attrition report.
FILTER_RULES: dict[str, Callable[[pd.DataFrame, IngestConfig], pd.Series[bool]]] = {
    "traffic": mask_traffic,
    "in_out_lap": mask_in_out_lap,
    "not_green": mask_not_green,
    "first_lap": mask_first_lap,
    "inaccurate": mask_inaccurate,
    "wet": mask_wet,
}


@dataclass(frozen=True)
class FilterResult:
    """Filtered laps plus the per-race attrition report."""

    laps: pd.DataFrame  # input frame + boolean ``is_clean`` column
    attrition: pd.DataFrame  # per (Season, Event, rule): dropped counts + survivors


def apply_filters(laps: pd.DataFrame, config: IngestConfig) -> FilterResult:
    """Run every filter rule; flag survivors and build the attrition report.

    ``dropped_marginal`` counts laps removed by a rule that survived all previous
    rules (sequential, sums to total dropped); ``dropped_total`` counts every lap
    the rule flags regardless of order.
    """
    laps = laps.copy()
    surviving = pd.Series(True, index=laps.index)
    rows: list[dict[str, object]] = []
    groups = cast(
        "dict[tuple[int, str], pd.Index]",
        dict(laps.groupby(["Season", "Event"], sort=False).groups),
    )

    for rule_name, rule in FILTER_RULES.items():
        drop = rule(laps, config)
        for (season, event), idx in groups.items():
            rows.append(
                {
                    "Season": season,
                    "Event": event,
                    "rule": rule_name,
                    "dropped_total": int(drop.loc[idx].sum()),
                    "dropped_marginal": int((drop & surviving).loc[idx].sum()),
                    "remaining": int((surviving & ~drop).loc[idx].sum()),
                }
            )
        surviving &= ~drop

    laps["is_clean"] = surviving
    return FilterResult(laps=laps, attrition=pd.DataFrame(rows))
