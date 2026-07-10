"""Build the committed parquet lap fixture from one cached FastF1 race session.

Run from the repo root (network required once; afterwards served from cache):

    uv run python tests/fixtures/build_fixture.py

Produces ``tests/fixtures/bahrain_2024_race_laps.parquet`` — the raw-ish lap table
(timedeltas converted to float seconds) used by golden-file tests. Tests must never
hit the FastF1 API; they read this file only.
"""

from pathlib import Path

import fastf1
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "fastf1_cache"
OUT_PATH = Path(__file__).resolve().parent / "bahrain_2024_race_laps.parquet"

SEASON = 2024
EVENT = "Bahrain"
SESSION = "R"

# Columns kept in the fixture — enough to exercise every Phase 1 filter rule
# and the output schema, while keeping the committed file small.
KEEP_COLUMNS = [
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


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    session = fastf1.get_session(SEASON, EVENT, SESSION)
    session.load(laps=True, telemetry=False, weather=True, messages=False)

    laps = session.laps[KEEP_COLUMNS].copy()
    for col in ("LapTime", "PitInTime", "PitOutTime", "Time"):
        laps[col] = laps[col].dt.total_seconds()
    laps["Season"] = SEASON
    laps["Event"] = EVENT
    laps["TotalLaps"] = session.total_laps

    weather = session.weather_data[["Time", "AirTemp", "TrackTemp"]].copy()
    weather["Time"] = weather["Time"].dt.total_seconds()
    # Attach ambient conditions to each lap by merging on nearest timestamp.
    laps = pd.merge_asof(
        laps.sort_values("Time"), weather.sort_values("Time"), on="Time", direction="nearest"
    )

    laps.to_parquet(OUT_PATH, index=False)
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {len(laps)} laps -> {OUT_PATH} ({size_kb:.0f} KiB)")


if __name__ == "__main__":
    main()
