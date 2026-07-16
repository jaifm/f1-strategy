"""Physics proxies from ~4 Hz lap telemetry (spec §2.3).

Every formula and its assumptions:

- **Savitzky-Golay smoothing** (window ~1.5 s, polyorder 2) is applied to X, Y
  and Speed BEFORE any differentiation — raw GPS is never differentiated.
- **Path curvature** ``κ = |x'y'' − y'x''| / (x'² + y'²)^1.5`` from smoothed
  x(t), y(t). FastF1 positions are in decimeters → converted to meters; Speed
  is km/h → m/s.
- **Lateral energy proxy** ``E_lat = v³·κ`` — lateral force ∝ mv²κ, work rate
  scales with v; mass is constant across a stint and dropped everywhere.
- **Brake thermal proxy** ``E_brake = max(0, −d/dt(½v²)) = max(0, −v·a_long)``.
- **Traction work proxy** ``E_traction = max(0, a_long)·v``.
- **G proxies** ``g_lat = v²κ/g``, ``g_long = a_long/g``.
- **Steering proxy**: steering angle is not public — smoothed dκ/dt is used as
  a steering-rate proxy (spec note).

All proxies are PROXIES of unmeasured quantities (no fuel mass, no brake or
tire temperature is public); never treat them as measured values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from f1_vision.config import ProxyConfig

G = 9.81
#: Output channels, in Tab2Img quadrant order (spec §2.4 table).
PROXY_CHANNELS = [
    "e_lat",
    "e_brake",
    "e_traction",
    "kappa",
    "g_lat",
    "g_long",
    "speed_ms",
    "rpm",
    "throttle",
    "brake",
    "steer_proxy",
    "gear",
]


def compute_proxies(telemetry: pd.DataFrame, config: ProxyConfig) -> pd.DataFrame:
    """Proxy channels for one lap of FastF1 telemetry.

    Expects columns ``Time`` (timedelta or seconds), ``Speed`` (km/h), ``X``,
    ``Y`` (decimeters), ``Throttle``, ``Brake``, ``RPM``, ``nGear``. Rows must
    be time-ordered. Raises if the lap is shorter than the SavGol window.
    """
    time = telemetry["Time"]
    t_raw = (
        time.dt.total_seconds().to_numpy()
        if pd.api.types.is_timedelta64_dtype(time)
        else time.to_numpy(dtype=float)
    )
    # Resample to a uniform grid: merged FastF1 telemetry interleaves the car and
    # position streams, so raw dt ranges 7ms-1s and differentiating across it
    # explodes. Derivatives below assume this constant spacing.
    t = np.arange(t_raw[0], t_raw[-1], 1.0 / config.resample_hz)
    if len(t) < config.savgol_window:
        raise ValueError(f"Lap has {len(t)} resampled points < savgol_window")

    def channel(name: str) -> np.ndarray:
        return np.interp(t, t_raw, telemetry[name].to_numpy(dtype=float))

    def smooth(values: np.ndarray) -> np.ndarray:
        return savgol_filter(values, config.savgol_window, config.savgol_polyorder)

    x = smooth(channel("X")) / 10.0  # dm -> m
    y = smooth(channel("Y")) / 10.0
    v = smooth(channel("Speed")) / 3.6  # km/h -> m/s

    dx, dy = np.gradient(x, t), np.gradient(y, t)
    ddx, ddy = np.gradient(dx, t), np.gradient(dy, t)
    speed_sq = dx**2 + dy**2
    kappa = np.abs(dx * ddy - dy * ddx) / np.maximum(speed_sq, 1.0) ** 1.5

    a_long = np.gradient(v, t)
    frame = pd.DataFrame(
        {
            "time_s": t,
            "e_lat": v**3 * kappa,
            "e_brake": np.maximum(0.0, -v * a_long),
            "e_traction": np.maximum(0.0, a_long) * v,
            "kappa": kappa,
            "g_lat": v**2 * kappa / G,
            "g_long": a_long / G,
            "speed_ms": v,
            "rpm": channel("RPM"),
            "throttle": channel("Throttle"),
            "brake": channel("Brake"),
            "steer_proxy": np.abs(np.gradient(smooth(kappa), t)),
            "gear": channel("nGear"),
        }
    )
    return frame
