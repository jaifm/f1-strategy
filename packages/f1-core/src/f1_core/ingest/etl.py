"""FastF1 ETL: race sessions → filtered lap parquet + attrition report.

All FastF1 API access for the project lives in this module (spec §6: isolate
FastF1 calls). Caching is enabled before any API call; tests never come here —
they run on the committed parquet fixture.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from f1_core.config import IngestConfig
from f1_core.ingest.filters import apply_filters
from f1_core.ingest.schema import LAP_SCHEMA

if TYPE_CHECKING:
    import fastf1

logger = logging.getLogger(__name__)

#: Raw lap columns pulled from FastF1 (mirrors the committed fixture).
RAW_COLUMNS = [
    "Driver",
    "Team",
    "LapNumber",
    "LapTime",
    "Compound",
    "TyreLife",
    "FreshTyre",
    "Stint",
    "PitInTime",
    "PitOutTime",
    "TrackStatus",
    "IsAccurate",
    "Deleted",
    "Position",
    "Time",
]

TIMEDELTA_COLUMNS = ("LapTime", "PitInTime", "PitOutTime", "Time")


def session_to_laps(session: fastf1.core.Session, season: int, event: str) -> pd.DataFrame:
    """Convert a loaded FastF1 session into the raw lap frame (float-second times)."""
    laps = session.laps[RAW_COLUMNS].copy()
    for col in TIMEDELTA_COLUMNS:
        laps[col] = laps[col].dt.total_seconds()
    laps["Season"] = season
    laps["Event"] = event
    laps["TotalLaps"] = session.total_laps

    weather = session.weather_data[["Time", "AirTemp", "TrackTemp", "Rainfall"]].copy()
    weather["Time"] = weather["Time"].dt.total_seconds()
    laps = pd.merge_asof(
        laps.sort_values("Time"),
        weather[["Time", "AirTemp", "TrackTemp"]].sort_values("Time"),
        on="Time",
        direction="nearest",
    )
    # Spec rule 6 input: share of weather samples reporting rain; the wet filter
    # thresholds this (config) so a sensor blip cannot mark a dry race wet.
    laps["RainfallFraction"] = float(weather["Rainfall"].astype(bool).mean())
    return laps


def to_output_schema(laps: pd.DataFrame) -> pd.DataFrame:
    """Map the raw+``is_clean`` frame to the spec output schema and validate it."""
    laps = laps.sort_values(["Season", "Event", "Driver", "LapNumber"])
    out = pd.DataFrame(
        {
            "driver": laps["Driver"],
            "team": laps["Team"],
            "event": laps["Event"],
            "season": laps["Season"],
            "lap_number": laps["LapNumber"],
            "lap_time_s": laps["LapTime"],
            "compound": laps["Compound"],
            "tyre_age": laps["TyreLife"],
            "stint_number": laps["Stint"],
            "stint_lap": laps.groupby(["Season", "Event", "Driver", "Stint"]).cumcount() + 1,
            "track_temp": laps["TrackTemp"],
            "air_temp": laps["AirTemp"],
            # Linear-burn fuel proxy: laps remaining (incl. this one) / total laps.
            "fuel_proxy": (laps["TotalLaps"] - laps["LapNumber"] + 1) / laps["TotalLaps"],
            "position": laps["Position"],
            "is_clean": laps["is_clean"],
        }
    )
    validated: pd.DataFrame = LAP_SCHEMA.validate(out)
    return validated


def _load_session(config: IngestConfig, season: int, round_number: int) -> Any:
    import fastf1

    fastf1.Cache.enable_cache(str(config.cache_dir))
    session = fastf1.get_session(season, round_number, config.session)
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    return session


def run_etl(config: IngestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ingest all configured seasons; write lap parquet + attrition CSV.

    Returns ``(laps, attrition)``. Wet sessions are kept in the lap table but
    fully flagged ``is_clean=False`` via the wet rule, so attrition stays visible.
    """
    import fastf1

    config.cache_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for season in config.seasons:
        schedule = fastf1.get_event_schedule(season, include_testing=False)
        for _, row in schedule.iterrows():
            event_name = str(row["EventName"])
            try:
                session = _load_session(config, season, int(row["RoundNumber"]))
            except Exception:  # noqa: BLE001 — one bad event must not kill the season
                logger.exception("Failed to load %s %s — skipping", season, event_name)
                continue
            frames.append(session_to_laps(session, season, event_name))
            logger.info("Loaded %s %s: %d laps", season, event_name, len(frames[-1]))

    if not frames:
        raise RuntimeError("ETL produced no laps — check seasons/config.")

    raw = pd.concat(frames, ignore_index=True)
    result = apply_filters(raw, config)
    laps = to_output_schema(result.laps)

    laps_path = config.output_dir / "laps.parquet"
    attrition_path = config.output_dir / "filter_attrition.csv"
    laps.to_parquet(laps_path, index=False)
    result.attrition.to_csv(attrition_path, index=False)
    logger.info(
        "Wrote %d laps (%d clean, %.1f%%) -> %s; attrition -> %s",
        len(laps),
        int(laps["is_clean"].sum()),
        100 * laps["is_clean"].mean(),
        laps_path,
        attrition_path,
    )
    return laps, result.attrition
