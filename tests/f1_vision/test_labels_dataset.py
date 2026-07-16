"""Labels (§2.2) and dataset windowing on synthetic laps with known outcomes."""

import numpy as np
import pandas as pd
import pytest

from f1_vision.config import F1VisionConfig, LabelConfig
from f1_vision.imaging.tab2img import ChannelScaler
from f1_vision.proxies.physics import PROXY_CHANNELS
from f1_vision.train.dataset import build_samples
from f1_vision.train.labels import add_labels

CFG = LabelConfig()


def make_stint(
    driver: str = "A",
    laps: range = range(2, 13),
    slow_from: int = 10,
    slow_by: float = 2.0,
    stint: float = 1.0,
) -> pd.DataFrame:
    """Clean stint: baseline 90s, laps >= slow_from run slow_by seconds slower."""
    rows = []
    for i, lap in enumerate(laps):
        rows.append(
            {
                "season": 2024,
                "event": "Testville",
                "driver": driver,
                "lap_number": lap,
                "lap_time_s": 90.0 + (slow_by if lap >= slow_from else 0.0),
                "compound": "SOFT",
                "tyre_age": float(i + 1),
                "stint_number": stint,
                "stint_lap": i + 1,
                "fuel_proxy": (30 - lap + 1) / 30,
                "track_temp": 35.0,
                "air_temp": 25.0,
                "team": "T",
                "position": 1.0,
                "is_clean": True,
            }
        )
    return pd.DataFrame(rows)


def race_filler(driver: str = "Z") -> pd.DataFrame:
    """A stint reaching lap 30 so the race doesn't end at the test stint's last lap."""
    return make_stint(driver=driver, laps=range(2, 31), slow_from=99)


class TestLabels:
    def test_pace_delta_next3(self) -> None:
        labeled = add_labels(pd.concat([make_stint(), race_filler()], ignore_index=True), CFG)
        me = labeled[labeled.driver == "A"].set_index("lap_number")
        # lap 7: next 3 laps are 8, 9 (delta 0) and 10 (delta 2) -> mean 2/3
        assert me.loc[7, "pace_delta_next3"] == pytest.approx(2 / 3)
        # lap 2: laps 3-5 all at baseline
        assert me.loc[2, "pace_delta_next3"] == pytest.approx(0.0)
        # last lap has no future laps -> NaN
        assert np.isnan(me.loc[12, "pace_delta_next3"])

    def test_cliff_from_pace_threshold(self) -> None:
        labeled = add_labels(pd.concat([make_stint(), race_filler()], ignore_index=True), CFG)
        me = labeled[labeled.driver == "A"].set_index("lap_number")
        assert me.loc[5, "cliff_within_5"] == 1.0  # window 6-10 includes the 2s jump
        assert me.loc[4, "cliff_within_5"] == 0.0  # window 5-9 stays at baseline

    def test_cliff_from_early_pit(self) -> None:
        # A pits after lap 12; B and C after lap 20 -> median 20, A is early.
        frame = pd.concat(
            [
                make_stint("A", range(2, 13), slow_from=99),
                make_stint("B", range(2, 21), slow_from=99),
                make_stint("C", range(2, 21), slow_from=99),
                race_filler(),
            ],
            ignore_index=True,
        )
        labeled = add_labels(frame, CFG)
        a = labeled[labeled.driver == "A"].set_index("lap_number")
        b = labeled[labeled.driver == "B"].set_index("lap_number")
        assert a.loc[8, "cliff_within_5"] == 1.0  # pit at 12 within horizon, earlier than median
        assert b.loc[8, "cliff_within_5"] == 0.0

    def test_short_stint_gets_no_labels(self) -> None:
        frame = pd.concat([make_stint("A", range(2, 4)), race_filler()], ignore_index=True)
        labeled = add_labels(frame, CFG)
        assert labeled.loc[labeled.driver == "A", "pace_delta_next3"].isna().all()


class TestDatasetWindows:
    def build(self) -> tuple:
        vision = F1VisionConfig()
        k = vision.imaging.k_sectors
        laps = pd.concat([make_stint(), race_filler()], ignore_index=True)
        rng = np.random.default_rng(5)
        tables = []
        for driver, lap in laps[["driver", "lap_number"]].itertuples(index=False):
            t = pd.DataFrame({ch: rng.uniform(0, 1, k) for ch in PROXY_CHANNELS})
            t["sector"] = range(k)
            t["season"], t["event"], t["driver"], t["lap_number"] = 2024, "Testville", driver, lap
            tables.append(t)
        sectors = pd.concat(tables, ignore_index=True)
        fit_frame = sectors.copy()
        for c in ("tyre_age", "fuel_proxy", "track_temp", "stint_lap"):
            fit_frame[c] = rng.uniform(0, 30, len(fit_frame))
        scaler = ChannelScaler.fit(fit_frame)
        return laps, sectors, scaler, vision

    def test_shapes_and_stint_scoped_windows(self) -> None:
        laps, sectors, scaler, vision = self.build()
        ds = build_samples(laps, sectors, scaler, vision)
        L, K = vision.imaging.window_laps, vision.imaging.k_sectors
        assert ds.frames.shape[1:] == (L, K, 5, 5)
        assert ds.frames.dtype == np.float32
        assert ds.frames.min() >= 0.0 and ds.frames.max() <= 1.0
        assert len(ds.frames) == len(ds.pace_delta) == len(ds.cliff) == len(ds.meta)
        # anchors need L laps of history within the stint: driver A stint starts
        # lap 2, so the first possible anchor is lap 6
        a_meta = ds.meta[ds.meta.driver == "A"]
        assert a_meta["lap_number"].min() >= 6
        # last lap (12) has NaN pace label -> excluded
        assert 12 not in set(a_meta["lap_number"])
