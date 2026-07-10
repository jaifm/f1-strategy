"""Unit tests for lap-filter rules (synthetic frames) + fixture integration."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from f1_core.config import IngestConfig
from f1_core.ingest.etl import to_output_schema
from f1_core.ingest.filters import (
    FILTER_RULES,
    apply_filters,
    mask_first_lap,
    mask_in_out_lap,
    mask_inaccurate,
    mask_not_green,
    mask_traffic,
    mask_wet,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "bahrain_2024_race_laps.parquet"
CONFIG = IngestConfig()


def make_laps(**overrides: list[Any]) -> pd.DataFrame:
    """Two clean laps (driver A leading driver B by 10 s) with optional overrides."""
    base: dict[str, list[Any]] = {
        "Driver": ["A", "B"],
        "Team": ["T1", "T2"],
        "LapNumber": [2, 2],
        "LapTime": [95.0, 95.5],
        "Compound": ["SOFT", "SOFT"],
        "TyreLife": [2.0, 2.0],
        "FreshTyre": [True, True],
        "Stint": [1.0, 1.0],
        "PitInTime": [None, None],
        "PitOutTime": [None, None],
        "TrackStatus": ["1", "1"],
        "IsAccurate": [True, True],
        "Deleted": [False, False],
        "Position": [1.0, 2.0],
        "Time": [200.0, 210.0],
        "Season": [2024, 2024],
        "Event": ["Testville", "Testville"],
        "TotalLaps": [57, 57],
        "AirTemp": [25.0, 25.0],
        "TrackTemp": [35.0, 35.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestTrafficRule:
    def test_close_gap_at_lap_end_drops_follower_only(self) -> None:
        laps = make_laps(Time=[200.0, 201.0])  # B crosses 1.0 s behind A
        assert mask_traffic(laps, CONFIG).tolist() == [False, True]

    def test_clear_air_keeps_both(self) -> None:
        laps = make_laps(Time=[200.0, 210.0])
        assert mask_traffic(laps, CONFIG).tolist() == [False, False]

    def test_gaps_never_computed_across_races(self) -> None:
        # Regression: with two races concatenated, cars on the same lap number of
        # DIFFERENT events must not be compared. Both races are clear-air here,
        # but naive lap-number grouping would see a 1.0 s cross-race "gap".
        race_a = make_laps(Time=[200.0, 210.0])
        race_b = make_laps(Event=["Otherton", "Otherton"], Time=[201.0, 220.0])
        laps = pd.concat([race_a, race_b], ignore_index=True)
        assert mask_traffic(laps, CONFIG).tolist() == [False, False, False, False]

    def test_close_gap_at_lap_start_drops_next_lap(self) -> None:
        # Lap 2: B is 1.0 s behind. Lap 3: B falls to 8 s behind, but started close.
        laps = pd.concat(
            [
                make_laps(Time=[200.0, 201.0]),
                make_laps(LapNumber=[3, 3], Time=[295.0, 303.0]),
            ],
            ignore_index=True,
        )
        assert mask_traffic(laps, CONFIG).tolist() == [False, True, False, True]


class TestSimpleRules:
    def test_in_lap_and_out_lap_dropped(self) -> None:
        laps = make_laps(PitInTime=[None, 300.0])
        assert mask_in_out_lap(laps, CONFIG).tolist() == [False, True]
        laps = make_laps(PitOutTime=[150.0, None])
        assert mask_in_out_lap(laps, CONFIG).tolist() == [True, False]

    @pytest.mark.parametrize(
        ("status", "dropped"),
        [("1", False), ("12", True), ("4", True), ("67", True), ("", True)],
    )
    def test_track_status(self, status: str, dropped: bool) -> None:
        laps = make_laps(TrackStatus=[status, "1"])
        assert mask_not_green(laps, CONFIG).tolist() == [dropped, False]

    def test_first_lap_dropped(self) -> None:
        laps = make_laps(LapNumber=[1, 2])
        assert mask_first_lap(laps, CONFIG).tolist() == [True, False]

    def test_inaccurate_or_deleted_dropped(self) -> None:
        laps = make_laps(IsAccurate=[False, True], Deleted=[False, True])
        assert mask_inaccurate(laps, CONFIG).tolist() == [True, True]

    def test_wet_compound_dropped(self) -> None:
        laps = make_laps(Compound=["INTERMEDIATE", "SOFT"])
        assert mask_wet(laps, CONFIG).tolist() == [True, False]

    def test_wet_session_drops_everything(self) -> None:
        laps = make_laps()
        laps["RainfallFraction"] = 0.5
        assert mask_wet(laps, CONFIG).tolist() == [True, True]

    def test_rainfall_blip_does_not_mark_session_wet(self) -> None:
        laps = make_laps()
        laps["RainfallFraction"] = 0.01  # single sensor blip in a dry race
        assert mask_wet(laps, CONFIG).tolist() == [False, False]


class TestApplyFiltersOnFixture:
    @pytest.fixture(scope="class")
    def fixture_laps(self) -> pd.DataFrame:
        return pd.read_parquet(FIXTURE)

    def test_attrition_report_structure(self, fixture_laps: pd.DataFrame) -> None:
        result = apply_filters(fixture_laps, CONFIG)
        assert len(result.attrition) == len(FILTER_RULES)  # one race in fixture
        assert list(result.attrition["rule"]) == list(FILTER_RULES)
        # Sequential survivors must be non-increasing and end at the clean count.
        remaining = result.attrition["remaining"].tolist()
        assert remaining == sorted(remaining, reverse=True)
        assert remaining[-1] == int(result.laps["is_clean"].sum())

    def test_marginal_drops_reconcile(self, fixture_laps: pd.DataFrame) -> None:
        result = apply_filters(fixture_laps, CONFIG)
        dropped = result.attrition["dropped_marginal"].sum()
        assert dropped + int(result.laps["is_clean"].sum()) == len(fixture_laps)

    def test_clean_fraction_plausible(self, fixture_laps: pd.DataFrame) -> None:
        result = apply_filters(fixture_laps, CONFIG)
        clean = result.laps["is_clean"].mean()
        # A dry race should keep a meaningful share of laps, but never all of them.
        assert 0.2 < clean < 0.9

    def test_output_schema_validates(self, fixture_laps: pd.DataFrame) -> None:
        result = apply_filters(fixture_laps, CONFIG)
        out = to_output_schema(result.laps)
        assert len(out) == len(fixture_laps)
        assert out["fuel_proxy"].iloc[0] <= 1.0
        # stint_lap restarts at 1 for every (driver, stint)
        firsts = out.groupby(["driver", "stint_number"])["stint_lap"].min()
        assert (firsts == 1).all()
