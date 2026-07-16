"""Tab2Img: proxy time series → mini-sector tables → 5×5 energy-image frames.

Spec §2.4. A lap is split into K=20 equal-distance mini-sectors; each sector is
aggregated (mean for continuous channels, max for the spiky energy proxies) and
rendered as one 5×5 frame → a lap is [K, 5, 5]. A training sample is the last
L=5 laps: **shape [L, K, 5, 5]** (documented choice; the CNN sees one frame per
step, the LSTM runs over the L·K flattened sequence).

Fixed handcrafted layout (Grad-CAM stays interpretable; not learned in v1):

    quadrant I  (energy,  top-left):  e_lat, e_brake / e_traction, kappa
    quadrant II (dynamics, top-right): g_lat, g_long / speed_ms, rpm
    quadrant III (driver, bottom-left): throttle, brake / steer_proxy, gear
    quadrant IV (context, bottom-right): tyre_age, fuel_proxy / track_temp, stint_lap

The center row/column (9 cells) are means of the adjacent quadrant cells so 3×3
kernels see cross-quadrant interaction.

Normalization: per-event min/max to [0, 1] from TRAIN-split laps only, persisted
(``ChannelScaler``). Events never seen in training fall back to global train
stats (whole events are held out, so no per-event stats can exist for them).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

from f1_vision.proxies.physics import PROXY_CHANNELS

#: Spiky energy proxies aggregate with max; everything else with mean.
ENERGY_MAX_CHANNELS = ["e_lat", "e_brake", "e_traction"]
CONTEXT_CHANNELS = ["tyre_age", "fuel_proxy", "track_temp", "stint_lap"]
ALL_CHANNELS = PROXY_CHANNELS + CONTEXT_CHANNELS

#: (row, col) -> channel for the 16 quadrant cells of the 5×5 grid.
GRID_CELLS: dict[tuple[int, int], str] = {
    (0, 0): "e_lat",
    (0, 1): "e_brake",
    (1, 0): "e_traction",
    (1, 1): "kappa",
    (0, 3): "g_lat",
    (0, 4): "g_long",
    (1, 3): "speed_ms",
    (1, 4): "rpm",
    (3, 0): "throttle",
    (3, 1): "brake",
    (4, 0): "steer_proxy",
    (4, 1): "gear",
    (3, 3): "tyre_age",
    (3, 4): "fuel_proxy",
    (4, 3): "track_temp",
    (4, 4): "stint_lap",
}


def minisector_table(proxies: pd.DataFrame, k_sectors: int) -> pd.DataFrame:
    """[k_sectors × 12] aggregate table over equal-distance bins of one lap."""
    dist = cumulative_trapezoid(proxies["speed_ms"], proxies["time_s"], initial=0.0)
    if dist[-1] <= 0:
        raise ValueError("Lap covers no distance")
    sector = np.minimum((dist / dist[-1] * k_sectors).astype(int), k_sectors - 1)
    agg = {ch: ("max" if ch in ENERGY_MAX_CHANNELS else "mean") for ch in PROXY_CHANNELS}
    table = proxies.groupby(sector).agg(agg)
    # Guard sparse laps: fill any empty sector from its neighbors.
    return table.reindex(range(k_sectors)).ffill().bfill()


def lap_frames(sectors: pd.DataFrame, context: dict[str, float]) -> np.ndarray:
    """[K, 5, 5] frames for one lap from its (normalized) sector table + context.

    Context channels (tyre age, fuel, track temp, stint lap) are constant across
    the lap's K frames.
    """
    k = len(sectors)
    values = sectors.copy()
    for name in CONTEXT_CHANNELS:
        values[name] = context[name]

    frames = np.zeros((k, 5, 5))
    for (i, j), channel in GRID_CELLS.items():
        frames[:, i, j] = values[channel].to_numpy()
    # Center row/col: means of vertically / horizontally adjacent quadrant cells.
    for j in (0, 1, 3, 4):
        frames[:, 2, j] = (frames[:, 1, j] + frames[:, 3, j]) / 2
    for i in (0, 1, 3, 4):
        frames[:, i, 2] = (frames[:, i, 1] + frames[:, i, 3]) / 2
    frames[:, 2, 2] = (frames[:, 1, 1] + frames[:, 1, 3] + frames[:, 3, 1] + frames[:, 3, 3]) / 4
    return frames


@dataclass
class ChannelScaler:
    """Per-event min/max → [0, 1], fit on train-split laps only. Persistable."""

    per_event: dict[str, pd.DataFrame]  # event -> DataFrame(index=channel, cols=[min, max])
    global_stats: pd.DataFrame

    @classmethod
    def fit(cls, tables: pd.DataFrame) -> ChannelScaler:
        """``tables``: concatenated sector tables + context columns + an ``event`` column."""

        def stats(frame: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                {"min": frame[ALL_CHANNELS].min(), "max": frame[ALL_CHANNELS].max()}
            )

        return cls(
            per_event={str(e): stats(g) for e, g in tables.groupby("event")},
            global_stats=stats(tables),
        )

    def transform(self, values: pd.DataFrame, event: str) -> pd.DataFrame:
        stats = self.per_event.get(event, self.global_stats)
        lo, hi = stats["min"], stats["max"]
        span = (hi - lo).replace(0.0, 1.0)
        cols = [c for c in values.columns if c in ALL_CHANNELS]
        out = values.copy()
        out[cols] = ((values[cols] - lo[cols]) / span[cols]).clip(0.0, 1.0)
        return out

    def save(self, path: Path | str) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path | str) -> ChannelScaler:
        scaler = pickle.loads(Path(path).read_bytes())
        if not isinstance(scaler, ChannelScaler):
            raise TypeError(f"{path} does not contain a ChannelScaler")
        return scaler
