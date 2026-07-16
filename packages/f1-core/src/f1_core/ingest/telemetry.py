"""Per-lap telemetry access (FastF1 isolation: all API calls stay in f1_core.ingest)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from f1_core.config import IngestConfig

TELEMETRY_COLUMNS = ["Time", "Speed", "RPM", "nGear", "Throttle", "Brake", "X", "Y"]


def load_race_telemetry_session(config: IngestConfig, season: int, event: str | int) -> Any:
    """Load a race session with telemetry (big download on first call, then cached)."""
    import fastf1

    config.cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(config.cache_dir))
    session = fastf1.get_session(season, event, config.session)
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    return session


def lap_telemetry(session: Any, driver: str, lap_number: int) -> pd.DataFrame:
    """Merged pos+car telemetry for one lap, restricted to the channels we use."""
    lap = session.laps.pick_drivers(driver).pick_laps(lap_number).iloc[0]
    telemetry: pd.DataFrame = lap.get_telemetry()
    return telemetry[TELEMETRY_COLUMNS].copy()
