"""Output schema for processed lap data (PROJECT_SPEC.md §1.2).

``fuel_proxy`` assumption: fuel load is not public, so ``laps_remaining /
total_laps`` (linear burn) is used as a normalized proxy — 1.0 ≈ full tank on
lap 1, → 1/total_laps on the final lap. It is a proxy, never a measured value.
"""

from __future__ import annotations

import pandera.pandas as pa

LAP_SCHEMA = pa.DataFrameSchema(
    {
        "driver": pa.Column(str),
        "team": pa.Column(str),
        "event": pa.Column(str),
        "season": pa.Column(int),
        "round": pa.Column(int, pa.Check.ge(0)),
        "lap_number": pa.Column(int, pa.Check.ge(1)),
        # Raw laps can carry red-flag stoppages (1000s of seconds); plausibility
        # is enforced on clean laps only via the dataframe-level check below.
        "lap_time_s": pa.Column(float, pa.Check.gt(0), nullable=True),
        "compound": pa.Column(str, nullable=True),
        "tyre_age": pa.Column(float, pa.Check.ge(0), nullable=True),
        "stint_number": pa.Column(float, pa.Check.ge(1), nullable=True),
        "stint_lap": pa.Column(int, pa.Check.ge(1)),
        "track_temp": pa.Column(float, nullable=True),
        "air_temp": pa.Column(float, nullable=True),
        "fuel_proxy": pa.Column(float, pa.Check.in_range(0.0, 1.0)),
        "position": pa.Column(float, pa.Check.in_range(1, 24), nullable=True),
        "is_clean": pa.Column(bool),
    },
    checks=pa.Check(
        lambda df: ~df["is_clean"] | df["lap_time_s"].between(30.0, 600.0),
        name="clean_lap_time_plausible",
        error="clean laps must have a lap time in [30, 600] s",
    ),
    strict=True,
    coerce=True,
)
