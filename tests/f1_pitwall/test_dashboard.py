"""Smoke test: the pit-wall app runs end-to-end against real processed data."""

from pathlib import Path

import pytest

DATA = Path("data/processed/laps.parquet")
APP = Path("packages/f1-pitwall/src/f1_pitwall/dashboard/pitwall.py")


@pytest.mark.skipif(not DATA.exists(), reason="processed laps not present (run ingest)")
def test_app_runs() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    assert len(at.metric) == 2  # win probabilities rendered
