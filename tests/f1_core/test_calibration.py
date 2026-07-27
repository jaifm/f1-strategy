"""End-to-end test of the calibration study on synthetic laps."""

from pathlib import Path

import pandas as pd
import pytest

from f1_core.config import EvaluationConfig, ModelConfig
from f1_core.evaluate.calibration import run_calibration_study

from .synth import make_synthetic

FAST_MODEL = ModelConfig(n_estimators=80, min_child_samples=20)


@pytest.fixture(scope="module")
def synthetic() -> pd.DataFrame:
    return make_synthetic(n_events=10)


def run(synthetic: pd.DataFrame, tmp_path: Path, **overrides: object) -> object:
    eval_config = EvaluationConfig(
        mode="single_split",
        calibration_fraction=0.2,
        test_fraction=0.2,
        split_seed=3,
        artifacts_dir=tmp_path / "artifacts",
        **overrides,  # type: ignore[arg-type]
    )
    return run_calibration_study(synthetic, FAST_MODEL, eval_config)


def test_study_metrics_and_artifacts(synthetic: pd.DataFrame, tmp_path: Path) -> None:
    report = run(synthetic, tmp_path)

    assert 0.84 <= report.metrics["picp_cqr_pooled"] <= 0.97
    # Quantile model must beat the MSE baseline on tail pinball loss.
    for col in ("q05", "q95"):
        assert (
            report.metrics[f"pinball_{col}_quantile_model"]
            < report.metrics[f"pinball_{col}_mse_baseline"]
        )

    out = tmp_path / "artifacts"
    for name in (
        "metrics.json",
        "coverage_per_event.csv",
        "interval_width_vs_tyre_age.csv",
        "picp_per_event.png",
        "interval_width_vs_tyre_age.png",
    ):
        assert (out / name).exists(), name

    assert set(report.coverage_per_event.columns) == {"event", "n_laps", "picp_cqr"}
    assert {"compound", "tyre_age", "n_laps", "upper_spread_s", "lower_spread_s"} <= set(
        report.width_by_tyre_age.columns
    )


def test_normalized_study_runs(synthetic: pd.DataFrame, tmp_path: Path) -> None:
    report = run(synthetic, tmp_path, per_event_normalization=True)
    assert report.metrics["per_event_normalization"] is True
    # Denormalized intervals must be on the raw lap-time scale, not delta scale.
    assert report.metrics["picp_cqr_pooled"] > 0.5


def test_walk_forward_uses_only_prior_rounds(synthetic: pd.DataFrame, tmp_path: Path) -> None:
    eval_config = EvaluationConfig(
        mode="walk_forward",
        test_season=2024,
        walk_calib_events=2,
        per_event_normalization=True,
        artifacts_dir=tmp_path / "artifacts_wf",
    )
    report = run_calibration_study(synthetic, FAST_MODEL, eval_config)
    n_events = synthetic["event"].nunique()
    # first folds lack history (need >= 1 train + 2 calib events) and are skipped
    assert report.metrics["n_folds"] == n_events - 3
    assert "Race0" not in set(report.coverage_per_event["event"])
    assert "Race9" in set(report.coverage_per_event["event"])
    assert 0.5 <= report.metrics["picp_cqr_pooled"] <= 1.0


def test_leave_one_event_out_pools_every_event(synthetic: pd.DataFrame, tmp_path: Path) -> None:
    eval_config = EvaluationConfig(
        mode="leave_one_event_out",
        calibration_fraction=0.2,
        split_seed=3,
        per_event_normalization=True,
        artifacts_dir=tmp_path / "artifacts_loo",
    )
    report = run_calibration_study(synthetic, FAST_MODEL, eval_config)
    n_events = synthetic["event"].nunique()
    assert report.metrics["n_folds"] == n_events
    assert len(report.coverage_per_event) == n_events
    assert report.metrics["n_test_laps_pooled"] == len(synthetic)
    assert 0.84 <= report.metrics["picp_cqr_pooled"] <= 0.97
