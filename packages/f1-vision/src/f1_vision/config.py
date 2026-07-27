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


class TrainConfig(BaseModel):
    """Training/study settings (spec §2.5)."""

    val_fraction: float = Field(default=0.15, description="Events held out for early stopping.")
    test_fraction: float = Field(default=0.2, description="Events held out for the report.")
    split_seed: int = Field(default=11)
    epochs: int = Field(default=60)
    batch_size: int = Field(default=256)
    lr: float = Field(default=1e-3)
    weight_decay: float = Field(default=1e-4)
    lambda_cliff: float = Field(default=0.5, description="λ in L = Huber + λ·BCE.")
    patience: int = Field(default=8, description="Early-stopping epochs without val improvement.")
    hidden: int = Field(default=128)
    embed_dim: int = Field(default=64)
    seed: int = Field(default=7)
    device: str = Field(
        default="cpu",
        description="Training device. MPS hangs on nn.LSTM (observed 2026-07); CPU is fast enough.",
    )


class F1VisionConfig(BaseModel):
    proxies: ProxyConfig = Field(default_factory=ProxyConfig)
    imaging: ImagingConfig = Field(default_factory=ImagingConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)


def load_config(path: Path | str | None = None) -> F1VisionConfig:
    if path is None:
        return F1VisionConfig()
    raw = yaml.safe_load(Path(path).read_text())
    return F1VisionConfig.model_validate(raw or {})
