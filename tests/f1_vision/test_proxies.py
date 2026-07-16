"""Proxy physics against synthetic trajectories with known ground truth."""

import numpy as np
import pandas as pd
import pytest

from f1_vision.config import ProxyConfig
from f1_vision.proxies.physics import PROXY_CHANNELS, G, compute_proxies

CONFIG = ProxyConfig()
HZ = 5.0


def make_telemetry(x_m: np.ndarray, y_m: np.ndarray, v_ms: np.ndarray) -> pd.DataFrame:
    n = len(x_m)
    return pd.DataFrame(
        {
            "Time": np.arange(n) / HZ,
            "Speed": v_ms * 3.6,  # engine expects km/h
            "X": x_m * 10.0,  # and decimeters
            "Y": y_m * 10.0,
            "Throttle": np.full(n, 50.0),
            "Brake": np.zeros(n),
            "RPM": np.full(n, 10000.0),
            "nGear": np.full(n, 5.0),
        }
    )


def test_circle_recovers_curvature_and_g_lat() -> None:
    # Constant 50 m/s on a 250 m radius circle: kappa = 1/250, g_lat = v^2/(r*g).
    r, v = 250.0, 50.0
    t = np.arange(0, 30, 1 / HZ)
    theta = v * t / r
    tel = make_telemetry(r * np.cos(theta), r * np.sin(theta), np.full(len(t), v))
    p = compute_proxies(tel, CONFIG)
    inner = p.iloc[10:-10]  # ignore filter edge effects
    assert inner["kappa"].mean() == pytest.approx(1 / r, rel=0.05)
    assert inner["g_lat"].mean() == pytest.approx(v**2 / (r * G), rel=0.05)
    assert inner["g_long"].abs().max() < 0.1  # no longitudinal accel
    assert inner["e_lat"].mean() == pytest.approx(v**3 / r, rel=0.05)


def test_straight_line_braking() -> None:
    # Straight line, 80 -> 40 m/s at -5 m/s^2: E_brake = v*|a|, no curvature.
    a, v0 = -5.0, 80.0
    t = np.arange(0, 8, 1 / HZ)
    v = v0 + a * t
    x = v0 * t + 0.5 * a * t**2
    tel = make_telemetry(x, np.zeros(len(t)), v)
    p = compute_proxies(tel, CONFIG)
    inner = p.iloc[10:-10]
    assert inner["g_long"].mean() == pytest.approx(a / G, rel=0.05)
    assert inner["kappa"].max() < 1e-3
    assert (inner["e_traction"] == 0).all()
    expected_e_brake = inner["speed_ms"] * abs(a)
    assert inner["e_brake"].to_numpy() == pytest.approx(expected_e_brake.to_numpy(), rel=0.1)


def test_output_schema_and_short_lap_guard() -> None:
    t = np.arange(0, 4, 1 / HZ)
    tel = make_telemetry(t * 50, np.zeros(len(t)), np.full(len(t), 50.0))
    p = compute_proxies(tel, CONFIG)
    assert set(PROXY_CHANNELS) <= set(p.columns)
    with pytest.raises(ValueError):
        compute_proxies(tel.head(5), CONFIG)


def test_timedelta_time_accepted() -> None:
    t = np.arange(0, 4, 1 / HZ)
    tel = make_telemetry(t * 50, np.zeros(len(t)), np.full(len(t), 50.0))
    tel["Time"] = pd.to_timedelta(tel["Time"], unit="s")
    p = compute_proxies(tel, CONFIG)
    assert p["speed_ms"].iloc[10] == pytest.approx(50.0, rel=0.01)
