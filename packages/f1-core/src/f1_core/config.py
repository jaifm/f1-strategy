"""Configuration for f1-core, loaded from YAML (no hardcoded magic numbers in code)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

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


class ModelConfig(BaseModel):
    """Hyperparameters for the quantile LightGBM model (and MSE baseline).

    Defaults were tuned on 2024 data: shallow, heavily regularized trees keep the
    quantile-crossing rate within the spec's 2% threshold (deep trees hit ~27%).
    """

    quantiles: list[float] = Field(default=[0.05, 0.5, 0.95])
    n_estimators: int = Field(default=150)
    learning_rate: float = Field(default=0.05)
    num_leaves: int = Field(default=7)
    min_child_samples: int = Field(default=200)
    seed: int = Field(default=7)


class EvaluationConfig(BaseModel):
    """Settings for the calibration study (spec §1.3 deliverable)."""

    mode: Literal["single_split", "leave_one_event_out"] = Field(
        default="leave_one_event_out",
        description=(
            "leave_one_event_out rotates every event through the test slot and pools "
            "coverage — single-split coverage on one season swings ±10% with the seed."
        ),
    )
    calibration_fraction: float = Field(default=0.2)
    test_fraction: float = Field(default=0.2)
    split_seed: int = Field(default=11)
    target_coverage: float = Field(default=0.9)
    coverage_tolerance: float = Field(default=0.02, description="Acceptance band: 90% ± 2%.")
    per_event_normalization: bool = Field(
        default=False,
        description=(
            "Model lap-time delta to the event's fastest clean lap instead of raw "
            "lap time (spec §1.3 fallback for poor cross-track generalization)."
        ),
    )
    artifacts_dir: Path = Field(default=Path("artifacts/phase1"))


class SimulationConfig(BaseModel):
    """Monte Carlo engine settings (spec §1.4)."""

    n_sims: int = Field(default=2000)
    rho_day: float = Field(
        default=0.65,
        description="Variance share: driver-day factor. Calibrated on 2024 (validate.py).",
    )
    rho_stint: float = Field(
        default=0.0,
        description="Variance share: tire-set factor. Inseparable from day on one season.",
    )
    pit_loss_mean_s: float = Field(default=22.0)
    pit_loss_std_s: float = Field(default=1.5)
    sc_rate_per_lap: float = Field(default=0.0, description="Safety car: v2, must stay 0.")
    seed: int = Field(default=7)


class FusionConfig(BaseModel):
    """Phase 3 fusion rule (spec §3.2). Pace/cliff inputs come from persistence
    (best §2.5 model): current clean-lap delta vs the stint baseline."""

    half_life_laps: float = Field(default=3.0, description="Decay of the median shift.")
    alpha: float = Field(default=1.0, description="Spread inflation: 1 + alpha * p_cliff.")
    n_sims: int = Field(default=500, description="Fast per-lap MC re-run.")
    baseline_laps: int = Field(default=3, description="Clean laps forming the stint baseline.")
    cliff_score_scale_s: float = Field(
        default=1.5,
        description="p_cliff = clip(current_delta / scale, 0, 1) — persistence heuristic.",
    )
    candidate_window_laps: int = Field(default=15, description="Stop laps evaluated ahead.")
    second_compound: str = Field(default="HARD", description="Tire fitted at the candidate stop.")


class F1CoreConfig(BaseModel):
    """Root config for the f1-core package."""

    ingest: IngestConfig = Field(default_factory=IngestConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)


def load_config(path: Path | str | None = None) -> F1CoreConfig:
    """Load config from YAML; missing file or ``None`` yields defaults."""
    if path is None:
        return F1CoreConfig()
    raw = yaml.safe_load(Path(path).read_text())
    return F1CoreConfig.model_validate(raw or {})
