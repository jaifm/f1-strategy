"""Temporal split: every training/val race precedes every test race."""

from pathlib import Path

import pandas as pd

from f1_core.config import F1CoreConfig, IngestConfig
from f1_vision.config import F1VisionConfig, TrainConfig
from f1_vision.train.dataset import sectors_path
from f1_vision.train.study import split_events


def test_temporal_split_has_no_future_races(tmp_path: Path) -> None:
    rows = []
    for season in (2023, 2024, 2025):
        for rnd, event in enumerate(["Bahrain", "Jeddah", "Melbourne", "Suzuka"], start=1):
            rows.append({"season": season, "round": rnd, "event": event, "is_clean": True})
            path = sectors_path(tmp_path, season, event)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()  # split_events only checks existence
    pd.DataFrame(rows).to_parquet(tmp_path / "laps.parquet")

    core = F1CoreConfig(ingest=IngestConfig(output_dir=tmp_path))
    vision = F1VisionConfig(train=TrainConfig(split_mode="temporal", test_season=2025, val_races=2))
    splits = split_events(core, vision)

    assert {s for s, _ in splits["test"]} == {2025} and len(splits["test"]) == 4
    # val = the two most recent pre-test races, in chronological order
    assert splits["val"] == [(2024, "Melbourne"), (2024, "Suzuka")]
    assert len(splits["train"]) == 6
    # same event name in different seasons must not collide across splits
    assert (2025, "Bahrain") in splits["test"] and (2023, "Bahrain") in splits["train"]
    assert not set(splits["train"]) & set(splits["test"])
