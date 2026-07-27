"""Smoke test: live pit-wall app boots and advances one lap on real data."""

from pathlib import Path

import pytest

DATA = Path("data/processed/laps.parquet")
APP = Path("packages/f1-pitwall/src/f1_pitwall/dashboard/live.py")


@pytest.mark.skipif(not DATA.exists(), reason="processed laps not present (run ingest)")
def test_live_app_boots_and_advances() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception
    at.button[0].click().run()  # advance 1 lap
    assert not at.exception
    assert at.metric[0].value == "1"  # lap counter moved
