"""Shared synthetic lap generator for f1-core tests.

Known generating process (linear tyre-age slope + fuel effect + Gaussian noise)
so physics constraints and coverage are checkable. Per-event offsets are small:
held-out events are UNSEEN categories, so a large offset breaks exchangeability
and no conformal method could cover (the real-data answer is per-event
normalization, spec §1.3).
"""

import numpy as np
import pandas as pd


def make_synthetic(n_events: int = 8, race_laps: int = 50, seed: int = 42) -> pd.DataFrame:
    """Laps for ``n_events`` races, 4 drivers, two stints per race."""
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_events):
        for driver, team in [("A", "T1"), ("B", "T1"), ("C", "T2"), ("D", "T2")]:
            stop = race_laps // 2 + int(rng.integers(-5, 5))
            for lap in range(2, race_laps + 1):
                stint = 1 if lap <= stop else 2
                tyre_age = lap if stint == 1 else lap - stop
                fuel = (race_laps - lap + 1) / race_laps
                rows.append(
                    {
                        "season": 2024,
                        "event": f"Race{e}",
                        "driver": driver,
                        "team": team,
                        "lap_number": lap,
                        "compound": "SOFT" if stint == 1 else "HARD",
                        "tyre_age": float(tyre_age),
                        "fuel_proxy": fuel,
                        "track_temp": 35.0 + e,
                        "air_temp": 25.0 + e,
                        "stint_lap": tyre_age,
                        "is_clean": True,
                        "lap_time_s": 90.0
                        + 0.05 * e
                        + 0.12 * tyre_age
                        + 2.0 * fuel
                        + float(rng.normal(0, 0.35)),
                    }
                )
    return pd.DataFrame(rows)
