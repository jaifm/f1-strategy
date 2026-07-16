"""Configuration for f1-vision, loaded from YAML (same pattern as f1-core)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ProxyConfig(BaseModel):
    """Physics-proxy generation settings (spec §2.3)."""

    savgol_window: int = Field(default=11, description="~1.5 s at 4-7 Hz telemetry.")
    savgol_polyorder: int = Field(default=2)
    resample_hz: float = Field(
        default=5.0,
        description=(
            "Uniform grid for derivatives: FastF1 merges car (240ms) and pos (220ms) "
            "streams, so raw sample spacing is wildly irregular (7ms-1s)."
        ),
    )


class ImagingConfig(BaseModel):
    """Tab2Img settings (spec §2.4)."""

    k_sectors: int = Field(default=20, description="Equal-distance mini-sectors per lap.")
    window_laps: int = Field(default=5, description="L: laps of frames per training sample.")


class LabelConfig(BaseModel):
    """Prediction targets (spec §2.2)."""

    baseline_laps: int = Field(
        default=3, description="Fresh-tire clean laps defining stint baseline."
    )
    next_laps: int = Field(default=3, description="Horizon for pace_delta_next3.")
    cliff_horizon: int = Field(default=5, description="Lookahead laps for cliff_within_5.")
    cliff_threshold_s: float = Field(default=1.5, description="Pace delta defining a cliff.")


class F1VisionConfig(BaseModel):
    proxies: ProxyConfig = Field(default_factory=ProxyConfig)
    imaging: ImagingConfig = Field(default_factory=ImagingConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)


def load_config(path: Path | str | None = None) -> F1VisionConfig:
    if path is None:
        return F1VisionConfig()
    raw = yaml.safe_load(Path(path).read_text())
    return F1VisionConfig.model_validate(raw or {})
