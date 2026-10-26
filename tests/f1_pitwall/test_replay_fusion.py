"""Replay engine, persistence state, fusion rule, and window movement."""

import numpy as np
import pandas as pd
import pytest

from f1_core.config import F1CoreConfig, FusionConfig
from f1_core.models.cqr import CQRCalibrator
from f1_core.simulate.engine import RaceContext
from f1_pitwall.cli import resolve_event
from f1_pitwall.orchestrator.replay import LapCompleted, TelemetryFrame, replay_race
from f1_pitwall.orchestrator.service import DriverState, PitwallOrchestrator, apply_fusion

FUSION = FusionConfig()


def make_laps(n_laps: int = 6) -> pd.DataFrame:
    rows = []
    for driver, base in [("A", 90.0), ("B", 91.0)]:
        for lap in range(1, n_laps + 1):
            rows.append(
                {
                    "driver": driver,
                    "lap_number": lap,
                    "lap_time_s": base,
                    "position": 1.0 if driver == "A" else 2.0,
                    "compound": "SOFT",
                    "tyre_age": float(lap),
                    "stint_number": 1.0,
                    "stint_lap": lap,
                    "is_clean": True,
                    "event": "Testville",
                    "season": 2024,
                }
            )
    return pd.DataFrame(rows)


class TestReplay:
    def test_events_ordered_and_complete(self) -> None:
        events = list(replay_race(make_laps(), speed=None))
        assert len(events) == 12  # 2 drivers x 6 laps, no telemetry
        times = [e.session_time_s for e in events]
        assert times == sorted(times)
        # A (90s laps) always completes each lap before B (91s laps)
        assert [e.driver for e in events[:2]] == ["A", "B"]

    def test_telemetry_frames_interleaved(self) -> None:
        laps = make_laps(2)
        sectors = pd.DataFrame(
            {"driver": ["A", "A"], "lap_number": [1, 1], "sector": [0, 1], "e_lat": [1.0, 2.0]}
        )
        events = list(replay_race(laps, sectors=sectors, speed=None))
        frames = [e for e in events if isinstance(e, TelemetryFrame)]
        assert len(frames) == 1 and frames[0].driver == "A"
        # the frame precedes A's LapCompleted for lap 1
        idx = events.index(frames[0])
        nxt = events[idx + 1]
        assert isinstance(nxt, LapCompleted) and nxt.driver == "A" and nxt.lap_number == 1


def lap(driver: str, n: int, t: float, stint: float = 1.0, clean: bool = True) -> LapCompleted:
    return LapCompleted(
        session_time_s=n * 90.0,
        driver=driver,
        lap_number=n,
        lap_time_s=t,
        position=1.0,
        compound="SOFT",
        tyre_age=float(n),
        stint_number=stint,
        stint_lap=n,
        is_clean=clean,
    )


class TestDriverState:
    def test_baseline_then_delta(self) -> None:
        s = DriverState(FUSION)
        for n, t in enumerate([90.0, 90.2, 89.8], start=1):
            s.update(lap("A", n, t))
        assert s.current_delta == 0.0  # still baseline
        s.update(lap("A", 4, 91.5))
        assert s.current_delta == pytest.approx(1.5)
        assert s.p_cliff == pytest.approx(1.0)  # 1.5 / scale 1.5

    def test_stint_change_resets(self) -> None:
        s = DriverState(FUSION)
        for n, t in enumerate([90.0, 90.0, 90.0, 93.0], start=1):
            s.update(lap("A", n, t))
        assert s.current_delta > 0
        s.update(lap("A", 5, 89.0, stint=2.0))
        assert s.current_delta == 0.0

    def test_dirty_laps_ignored(self) -> None:
        s = DriverState(FUSION)
        for n, t in enumerate([90.0, 90.0, 90.0], start=1):
            s.update(lap("A", n, t))
        s.update(lap("A", 4, 130.0, clean=False))
        assert s.current_delta == 0.0


class TestApplyFusion:
    def test_median_shift_decays_and_spread_inflates(self) -> None:
        q = np.tile([89.0, 90.0, 91.0], (10, 1)).astype(float)
        fused = apply_fusion(q, current_lap=2, stop_lap=8, delta=1.0, p_cliff=1.0, config=FUSION)
        # lap 3 (row 2): full shift; lap 6 (row 5): half (half-life 3)
        assert fused[2, 1] == pytest.approx(91.0)
        assert fused[5, 1] == pytest.approx(90.5)
        # upper spread doubled (alpha=1, p=1); lower spread untouched
        assert fused[2, 2] - fused[2, 1] == pytest.approx(2.0)
        assert fused[2, 1] - fused[2, 0] == pytest.approx(1.0)
        # laps outside (current, stop] untouched
        assert (fused[8:] == q[8:]).all()
        assert (fused[:2] == q[:2]).all()


class FakeModel:
    """Low degradation (0.01 s/lap): without fusion, stopping late always wins;
    the fused 3 s delta must be what pulls the window forward."""

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        q50 = 90.0 + 0.01 * features["tyre_age"].to_numpy()
        return pd.DataFrame({"q05": q50 - 1, "q50": q50, "q95": q50 + 1})


class TestWindowMovesWithFusion:
    def orchestrator(self, fusion_enabled: bool) -> PitwallOrchestrator:
        ctx = RaceContext(
            event="Testville",
            season=2024,
            driver="A",
            team="T",
            track_temp=35.0,
            air_temp=25.0,
            total_laps=57,
        )
        orch = PitwallOrchestrator(
            FakeModel(),  # type: ignore[arg-type]
            CQRCalibrator(margin=0.0, target_coverage=0.9),
            ctx,
            F1CoreConfig(),
            fusion_enabled=fusion_enabled,
        )
        # baseline 3 clean laps at 90, then a collapse to 93 (delta 3s)
        for n, t in enumerate([90.0, 90.0, 90.0, 93.0, 93.0], start=1):
            orch.on_lap(lap("A", n, t))
        return orch

    def test_collapsing_driver_window_pulls_forward(self) -> None:
        fused = self.orchestrator(True).stop_window("A", 5, "SOFT")
        anchored = self.orchestrator(False).stop_window("A", 5, "SOFT")
        assert fused is not None and anchored is not None
        assert fused.current_delta == pytest.approx(3.0)
        assert fused.p_cliff == 1.0
        assert fused.stop_p50 < anchored.stop_p50  # window moves earlier
        assert fused.compute_s < 1.0  # spec: < 1 s per lap update


def test_resolve_event() -> None:
    laps = pd.DataFrame(
        {
            "event": ["Bahrain Grand Prix", "Bahrain Grand Prix", "Austrian Grand Prix"],
            "season": [2022, 2024, 2024],
        }
    )
    assert resolve_event(laps, "Bahrain 2022") == ("Bahrain Grand Prix", 2022)
    assert resolve_event(laps, "bahrain") == ("Bahrain Grand Prix", 2024)  # latest season
    with pytest.raises(SystemExit):
        resolve_event(laps, "Monaco 2024")
    with pytest.raises(SystemExit):
        resolve_event(laps, "Bahrain 2023")  # no data for that season


def test_prior_races_excludes_this_and_later_races() -> None:
    from f1_pitwall.orchestrator.service import prior_races

    clean = pd.DataFrame(
        {
            "season": [2023, 2024, 2024, 2024, 2025],
            "round": [20, 19, 20, 21, 1],
            "event": ["Mexico", "Austin", "Mexico", "Brazil", "Australia"],
        }
    )
    kept = prior_races(clean, 2024, "Mexico")
    assert list(zip(kept.season, kept.event, strict=True)) == [(2023, "Mexico"), (2024, "Austin")]
    with pytest.raises(ValueError):
        prior_races(clean, 2022, "Mexico")
