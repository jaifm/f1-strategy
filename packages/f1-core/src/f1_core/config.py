"""Configuration for f1-core, loaded from YAML (no hardcoded magic numbers in code)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class IngestConfig(BaseModel):
    """Settings for the FastF1 ETL and lap filtering."""

    seasons: list[int] = Field(default=[2024], description="Seasons to ingest.")
    session: str = Field(default="R", description="Session identifier (R = race).")
    cache_dir: Path = Field(default=Path("data/fastf1_cache"))
    output_dir: Path = Field(default=Path("data/processed"))
    traffic_gap_s: float = Field(
        default=1.5, description="Drop a lap if the gap to the car ahead is below this (s)."
    )
    dry_compounds: list[str] = Field(default=["SOFT", "MEDIUM", "HARD"])
    wet_rainfall_fraction: float = Field(
        default=0.1,
        description="Mark a session wet if more than this share of weather samples report rain.",
    )


class F1CoreConfig(BaseModel):
    """Root config for the f1-core package."""

    ingest: IngestConfig = Field(default_factory=IngestConfig)


def load_config(path: Path | str | None = None) -> F1CoreConfig:
    """Load config from YAML; missing file or ``None`` yields defaults."""
    if path is None:
        return F1CoreConfig()
    raw = yaml.safe_load(Path(path).read_text())
    return F1CoreConfig.model_validate(raw or {})
