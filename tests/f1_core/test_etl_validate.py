"""Tests for session_to_laps (fake session, no API) and the MC spread validator."""

import numpy as np
import pandas as pd

from f1_core.config import SimulationConfig
from f1_core.ingest.etl import RAW_COLUMNS, session_to_laps
from f1_core.models.quantile import QuantileModelParams
from f1_core.simulate.validate import validate_stint_spread

from .synth import make_synthetic


class FakeSession:
    """Minimal stand-in for a loaded fastf1 session."""

    total_laps = 3

    def __init__(self) -> None:
        td = lambda s: pd.to_timedelta(s, unit="s")  # noqa: E731
        self.laps = pd.DataFrame(
            {
                "Driver": ["VER", "VER", "VER"],
                "Team": ["Red Bull"] * 3,
                "LapNumber": [1.0, 2.0, 3.0],
                "LapTime": td([95.0, 94.0, 96.0]),
                "Compound": ["SOFT"] * 3,
                "TyreLife": [1.0, 2.0, 3.0],
                "FreshTyre": [True, True, True],
                "Stint": [1.0] * 3,
                "PitInTime": td([np.nan, np.nan, np.nan]),
                "PitOutTime": td([np.nan, np.nan, np.nan]),
                "TrackStatus": ["1"] * 3,
                "IsAccurate": [True] * 3,
                "Deleted": [False] * 3,
                "Position": [1.0] * 3,
                "Time": td([100.0, 195.0, 290.0]),
            }
        )[RAW_COLUMNS]
        self.weather_data = pd.DataFrame(
            {
                "Time": td([50.0, 250.0]),
                "AirTemp": [25.0, 26.0],
                "TrackTemp": [35.0, 40.0],
                "Rainfall": [False, True],
            }
        )


def test_session_to_laps() -> None:
    laps = session_to_laps(FakeSession(), 2024, "Testville")
    assert len(laps) == 3
    assert laps["LapTime"].tolist() == [95.0, 94.0, 96.0]  # timedeltas -> seconds
    assert laps["TotalLaps"].iloc[0] == 3
    assert (laps["RainfallFraction"] == 0.5).all()
    # weather merged on nearest timestamp: lap at t=100 -> sample at t=50
    assert laps.sort_values("LapNumber")["TrackTemp"].tolist() == [35.0, 40.0, 40.0]


def test_validate_stint_spread_on_synthetic() -> None:
    laps = make_synthetic(n_events=5)
    result = validate_stint_spread(
        laps,
        "Race0",
        QuantileModelParams(n_estimators=60, min_child_samples=20),
        SimulationConfig(n_sims=500),
    )
    assert result["event"] == "Race0"
    assert result["n_stints"] > 0
    assert 0.0 <= result["coverage"] <= 1.0
    assert 0.0 <= result["calibrated_rho_day"] <= 0.8
    assert result["spread_ratio"] > 0
