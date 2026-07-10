"""Golden-file checks on the committed Bahrain 2024 lap fixture.

Pins the schema and basic invariants that Phase 1 ingest/filter code will rely on.
"""

from pathlib import Path

import pandas as pd
import pytest

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "bahrain_2024_race_laps.parquet"

EXPECTED_COLUMNS = {
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
    "Season",
    "Event",
    "TotalLaps",
    "AirTemp",
    "TrackTemp",
}


@pytest.fixture(scope="module")
def laps() -> pd.DataFrame:
    return pd.read_parquet(FIXTURE)


def test_schema(laps: pd.DataFrame) -> None:
    assert set(laps.columns) == EXPECTED_COLUMNS


def test_shape_and_content(laps: pd.DataFrame) -> None:
    assert len(laps) == 1129
    assert laps["Driver"].nunique() == 20
    assert (laps["Season"] == 2024).all()
    assert (laps["Event"] == "Bahrain").all()
    assert laps["TotalLaps"].iloc[0] == 57


def test_lap_times_plausible(laps: pd.DataFrame) -> None:
    clean = laps.loc[laps["IsAccurate"], "LapTime"].dropna()
    assert not clean.empty
    # Bahrain race laps: pole-ish pace to heavy-fuel/worn-tire pace.
    assert clean.between(85.0, 120.0).all()


def test_compounds_are_dry(laps: pd.DataFrame) -> None:
    # 2024 Bahrain GP was dry — fixture should contain only dry compounds.
    assert set(laps["Compound"].dropna().unique()) <= {"SOFT", "MEDIUM", "HARD"}
