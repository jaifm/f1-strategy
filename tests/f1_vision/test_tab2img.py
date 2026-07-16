"""Tab2Img: sector binning, grid layout, blends, and scaler behavior."""

import numpy as np
import pandas as pd
import pytest

from f1_vision.imaging.tab2img import (
    ALL_CHANNELS,
    GRID_CELLS,
    ChannelScaler,
    lap_frames,
    minisector_table,
)
from f1_vision.proxies.physics import PROXY_CHANNELS

K = 20


def make_proxies(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    p = pd.DataFrame({ch: rng.uniform(0, 1, n) for ch in PROXY_CHANNELS})
    p["time_s"] = np.arange(n) / 5.0
    p["speed_ms"] = 50.0  # constant speed -> equal-distance == equal-time bins
    return p


class TestMiniSectors:
    def test_shape_and_no_gaps(self) -> None:
        table = minisector_table(make_proxies(), K)
        assert table.shape == (K, len(PROXY_CHANNELS))
        assert not table.isna().any().any()

    def test_energy_uses_max_and_spike_lands_in_right_sector(self) -> None:
        p = make_proxies()
        p["e_brake"] = 0.0
        p.loc[210, "e_brake"] = 99.0  # sample 210/400 -> sector 10
        table = minisector_table(p, K)
        assert table["e_brake"].idxmax() == 10
        assert table["e_brake"].max() == 99.0  # max, not mean

    def test_zero_distance_raises(self) -> None:
        p = make_proxies()
        p["speed_ms"] = 0.0
        with pytest.raises(ValueError):
            minisector_table(p, K)


class TestLapFrames:
    def test_layout_blends_and_shape(self) -> None:
        table = minisector_table(make_proxies(), K)
        context = {"tyre_age": 0.5, "fuel_proxy": 0.8, "track_temp": 0.3, "stint_lap": 0.5}
        frames = lap_frames(table, context)
        assert frames.shape == (K, 5, 5)
        # named cells carry their channel
        assert frames[0, 0, 0] == table["e_lat"].iloc[0]
        assert frames[0, 4, 4] == context["stint_lap"]
        # blends are means of adjacent quadrant cells
        assert frames[0, 2, 0] == pytest.approx((frames[0, 1, 0] + frames[0, 3, 0]) / 2)
        assert frames[0, 0, 2] == pytest.approx((frames[0, 0, 1] + frames[0, 0, 3]) / 2)
        assert frames[0, 2, 2] == pytest.approx(
            (frames[0, 1, 1] + frames[0, 1, 3] + frames[0, 3, 1] + frames[0, 3, 3]) / 4
        )

    def test_grid_covers_all_channels_once(self) -> None:
        assert sorted(GRID_CELLS.values()) == sorted(ALL_CHANNELS)


class TestScaler:
    def make_tables(self) -> pd.DataFrame:
        rng = np.random.default_rng(4)
        rows = []
        for event, scale in [("A", 1.0), ("B", 10.0)]:
            t = pd.DataFrame({ch: rng.uniform(0, scale, 50) for ch in ALL_CHANNELS})
            t["event"] = event
            rows.append(t)
        return pd.concat(rows, ignore_index=True)

    def test_per_event_unit_range(self) -> None:
        tables = self.make_tables()
        scaler = ChannelScaler.fit(tables)
        b = scaler.transform(tables[tables.event == "B"], "B")
        assert b["e_lat"].max() == pytest.approx(1.0)
        assert b["e_lat"].min() == pytest.approx(0.0)
        # event A's scale must not leak into B (B spans 0-10, A only 0-1)
        assert scaler.per_event["B"].loc["e_lat", "max"] > 5.0

    def test_unseen_event_falls_back_to_global_and_clips(self) -> None:
        tables = self.make_tables()
        scaler = ChannelScaler.fit(tables)
        out = scaler.transform(tables.head(10) * 100, "UnseenGP")
        assert out[ALL_CHANNELS].max().max() <= 1.0
        assert out[ALL_CHANNELS].min().min() >= 0.0

    def test_save_load_roundtrip(self, tmp_path: object) -> None:
        scaler = ChannelScaler.fit(self.make_tables())
        path = f"{tmp_path}/scaler.pkl"
        scaler.save(path)
        loaded = ChannelScaler.load(path)
        src = self.make_tables().head(5)
        pd.testing.assert_frame_equal(loaded.transform(src, "A"), scaler.transform(src, "A"))
