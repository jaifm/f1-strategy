"""Calibration study — the spec §1.3 deliverable.

Trains the quantile model and the required MSE baseline on event-level splits of
clean laps, applies CQR, and produces:

- ``metrics.json`` — pinball losses (quantile vs baseline), PICP pre/post CQR,
  interval widths, per-event band compliance.
- ``coverage_per_event.csv`` + ``picp_per_event.png`` — held-out coverage per
  race against the 90% ± 2% acceptance band.
- ``interval_width_vs_tyre_age.csv`` + ``.png`` — mean (q95−q50) and (q50−q05)
  by tyre age on held-out laps: the widening upper spread IS the tire-cliff
  signal (spec acceptance criterion).

Evaluation modes (config): ``leave_one_event_out`` (default) rotates every event
through the test slot and pools predictions — on a single season, single-split
coverage swings roughly ±10% with the split seed because only a handful of
events land in each bucket. ``single_split`` remains for quick runs and tests.

Per-event normalization (config flag): models the delta to the event's fastest
clean lap instead of raw lap time, then adds the benchmark back to predictions.
Required in practice: the raw global model reached 16% raw coverage on held-out
2024 events (base pace differs by tens of seconds across tracks and a held-out
event is an unseen category). For a held-out race the benchmark uses that race's
own clean laps — in a live setting it would come from practice/qualifying pace
instead; documented here so the assumption is never silent.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from f1_core.config import EvaluationConfig, ModelConfig
from f1_core.evaluate.metrics import mean_interval_width, picp, pinball_loss
from f1_core.models.cqr import fit_cqr
from f1_core.models.quantile import (
    TARGET,
    MSEBaselineModel,
    QuantileLapModel,
    QuantileModelParams,
)
from f1_core.models.splits import split_by_event

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

BENCHMARK = "event_benchmark_s"


@dataclass
class CalibrationReport:
    """Everything the study produced, for programmatic use and tests."""

    metrics: dict[str, Any]
    coverage_per_event: pd.DataFrame
    width_by_tyre_age: pd.DataFrame


@dataclass
class _FoldResult:
    """Held-out predictions from one train/calibration/test fold."""

    test: pd.DataFrame
    raw: pd.DataFrame
    adjusted: pd.DataFrame
    baseline_pred: pd.Series[float]
    cqr_margin: float


def _params_from_config(model_config: ModelConfig) -> QuantileModelParams:
    return QuantileModelParams(
        quantiles=tuple(model_config.quantiles),
        n_estimators=model_config.n_estimators,
        learning_rate=model_config.learning_rate,
        num_leaves=model_config.num_leaves,
        min_child_samples=model_config.min_child_samples,
        seed=model_config.seed,
    )


def add_event_benchmark(laps: pd.DataFrame) -> pd.DataFrame:
    """Attach each event's fastest clean lap (the normalization benchmark)."""
    laps = laps.copy()
    laps[BENCHMARK] = laps.groupby(["season", "event"])[TARGET].transform("min")
    return laps


def _run_fold(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    params: QuantileModelParams,
    eval_config: EvaluationConfig,
) -> _FoldResult:
    normalize = eval_config.per_event_normalization

    def norm(frame: pd.DataFrame) -> pd.DataFrame:
        if not normalize:
            return frame
        adjusted = frame.copy()
        adjusted[TARGET] = adjusted[TARGET] - adjusted[BENCHMARK]
        return adjusted

    def denorm(intervals: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
        return intervals.add(frame[BENCHMARK], axis=0) if normalize else intervals

    model = QuantileLapModel(params).fit(norm(train))
    baseline = MSEBaselineModel(params).fit(norm(train))
    calibrator = fit_cqr(
        norm(calibration)[TARGET],
        model.predict(norm(calibration)),
        eval_config.target_coverage,
    )

    raw = denorm(model.predict(norm(test)), test)
    adjusted = denorm(calibrator.apply(model.predict(norm(test))), test)
    baseline_pred = baseline.predict(norm(test))
    if normalize:
        baseline_pred = baseline_pred + test[BENCHMARK]

    return _FoldResult(
        test=test,
        raw=raw,
        adjusted=adjusted,
        baseline_pred=baseline_pred,
        cqr_margin=calibrator.margin,
    )


def _walk_forward_folds(
    clean: pd.DataFrame,
    params: QuantileModelParams,
    eval_config: EvaluationConfig,
) -> list[_FoldResult]:
    """One fold per test-season event, trained on strictly earlier rounds.

    Expanding window in (season, round) order: earlier test-season events join
    the training set of later folds. The CQR calibration set is the most recent
    ``walk_calib_events`` prior events — conformity drifts with regulations and
    tire generations, so recent races calibrate better than random ones.
    """
    keys = (
        clean[["season", "round", "event"]]
        .drop_duplicates()
        .sort_values(["season", "round"])
        .reset_index(drop=True)
    )
    if (keys["round"] == 0).any():
        raise ValueError("walk_forward needs the `round` column — re-run ingest")
    test_keys = keys[keys["season"] == eval_config.test_season]
    if test_keys.empty:
        raise ValueError(f"No events for test_season {eval_config.test_season}")

    folds = []
    for i, row in enumerate(test_keys.itertuples(index=False)):
        prior = keys[
            (keys["season"] < row.season)
            | ((keys["season"] == row.season) & (keys["round"] < row.round))
        ]
        calib_keys = prior.tail(eval_config.walk_calib_events)
        train_keys = prior.iloc[: len(prior) - len(calib_keys)]
        if train_keys.empty or calib_keys.empty:
            logger.info("skipping %s: insufficient history", row.event)
            continue

        def take(selected: pd.DataFrame) -> pd.DataFrame:
            return clean.merge(selected[["season", "event"]], on=["season", "event"])

        test = clean[(clean["season"] == row.season) & (clean["event"] == row.event)]
        folds.append(_run_fold(take(train_keys), take(calib_keys), test, params, eval_config))
        logger.info(
            "walk-forward fold %d/%d: %s (train %d events) done",
            i + 1,
            len(test_keys),
            row.event,
            len(train_keys),
        )
    return folds


def _loo_folds(
    clean: pd.DataFrame,
    params: QuantileModelParams,
    eval_config: EvaluationConfig,
) -> list[_FoldResult]:
    """One fold per event: that event is the test set, the rest split train/calib."""
    events = sorted(clean["event"].unique())
    rng = np.random.default_rng(eval_config.split_seed)
    folds = []
    for i, held_out in enumerate(events):
        rest_events = [e for e in events if e != held_out]
        n_calib = max(1, round(len(rest_events) * eval_config.calibration_fraction))
        calib_events = set(rng.choice(rest_events, size=n_calib, replace=False))
        test = clean[clean["event"] == held_out]
        rest = clean[clean["event"] != held_out]
        calibration = rest[rest["event"].isin(calib_events)]
        train = rest[~rest["event"].isin(calib_events)]
        folds.append(_run_fold(train, calibration, test, params, eval_config))
        logger.info("LOO fold %d/%d: %s done", i + 1, len(events), held_out)
    return folds


def run_calibration_study(
    laps: pd.DataFrame,
    model_config: ModelConfig,
    eval_config: EvaluationConfig,
) -> CalibrationReport:
    """Run the full study and write artifacts to ``eval_config.artifacts_dir``."""
    clean = laps[laps["is_clean"] & laps[TARGET].notna()].reset_index(drop=True)
    logger.info("Calibration study (%s) on %d clean laps", eval_config.mode, len(clean))

    if eval_config.per_event_normalization:
        clean = add_event_benchmark(clean)

    params = _params_from_config(model_config)
    if eval_config.mode == "walk_forward":
        folds = _walk_forward_folds(clean, params, eval_config)
    elif eval_config.mode == "leave_one_event_out":
        folds = _loo_folds(clean, params, eval_config)
    else:
        split = split_by_event(
            clean,
            calibration_fraction=eval_config.calibration_fraction,
            test_fraction=eval_config.test_fraction,
            seed=eval_config.split_seed,
        )
        folds = [_run_fold(split.train, split.calibration, split.test, params, eval_config)]

    # Pool held-out predictions across folds; every test event appears exactly once.
    test = pd.concat([f.test for f in folds]).reset_index(drop=True)
    raw = pd.concat([f.raw for f in folds]).reset_index(drop=True)
    adjusted = pd.concat([f.adjusted for f in folds]).reset_index(drop=True)
    baseline_pred = pd.concat([f.baseline_pred for f in folds]).reset_index(drop=True)
    y_test = test[TARGET]

    coverage_rows = []
    for event, group in test.groupby("event", sort=True):
        coverage_rows.append(
            {
                "event": event,
                "n_laps": len(group),
                "picp_cqr": picp(group[TARGET], adjusted.loc[group.index]),
            }
        )
    coverage_per_event = pd.DataFrame(coverage_rows)

    target = eval_config.target_coverage
    tol = eval_config.coverage_tolerance
    within = coverage_per_event["picp_cqr"].between(target - tol, target + tol)

    metrics: dict[str, Any] = {
        "mode": eval_config.mode,
        "n_folds": len(folds),
        "n_test_laps_pooled": len(test),
        "per_event_normalization": eval_config.per_event_normalization,
        "target_coverage": target,
        "picp_raw_pooled": picp(y_test, raw),
        "picp_cqr_pooled": picp(y_test, adjusted),
        "events_within_band": f"{int(within.sum())}/{len(coverage_per_event)}",
        "mean_cqr_margin_s": float(np.mean([f.cqr_margin for f in folds])),
        "mean_width_raw_s": mean_interval_width(raw),
        "mean_width_cqr_s": mean_interval_width(adjusted),
    }
    for q, col in zip(params.quantiles, raw.columns, strict=True):
        metrics[f"pinball_{col}_quantile_model"] = pinball_loss(y_test, raw[col], q)
        metrics[f"pinball_{col}_mse_baseline"] = pinball_loss(y_test, baseline_pred, q)

    # Spread vs age must be conditioned on compound: pooled across compounds the
    # widening washes out (a hard tire at age 30 is healthy, a soft at 20 is
    # dying), and long stints only survive at low-degradation tracks.
    width = (
        pd.DataFrame(
            {
                "compound": test["compound"],
                "tyre_age": test["tyre_age"],
                "upper_spread_s": adjusted["q95"] - adjusted["q50"],
                "lower_spread_s": adjusted["q50"] - adjusted["q05"],
            }
        )
        .groupby(["compound", "tyre_age"])
        .agg(
            n_laps=("upper_spread_s", "size"),
            upper_spread_s=("upper_spread_s", "mean"),
            lower_spread_s=("lower_spread_s", "mean"),
        )
        .reset_index()
    )
    for compound, group in width[width["n_laps"] >= 10].groupby("compound"):
        fresh = group.loc[group["tyre_age"] <= 5, "upper_spread_s"].mean()
        worn = group.loc[group["tyre_age"] >= 20, "upper_spread_s"].mean()
        if pd.notna(fresh) and pd.notna(worn):
            metrics[f"upper_spread_widening_{compound}_s"] = float(worn - fresh)

    _write_artifacts(eval_config, metrics, coverage_per_event, width)
    return CalibrationReport(
        metrics=metrics,
        coverage_per_event=coverage_per_event,
        width_by_tyre_age=width,
    )


def _write_artifacts(
    eval_config: EvaluationConfig,
    metrics: dict[str, Any],
    coverage_per_event: pd.DataFrame,
    width: pd.DataFrame,
) -> None:
    out = eval_config.artifacts_dir
    out.mkdir(parents=True, exist_ok=True)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    coverage_per_event.to_csv(out / "coverage_per_event.csv", index=False)
    width.to_csv(out / "interval_width_vs_tyre_age.csv", index=False)

    target = metrics["target_coverage"]
    tol = eval_config.coverage_tolerance

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(coverage_per_event["event"], coverage_per_event["picp_cqr"], color="#4477aa")
    ax.axhline(target, color="black", linestyle="--", linewidth=1, label=f"target {target:.0%}")
    ax.axhspan(target - tol, target + tol, color="green", alpha=0.15, label=f"±{tol:.0%} band")
    pooled = metrics["picp_cqr_pooled"]
    ax.axhline(pooled, color="#cc3311", linewidth=1.5, label=f"pooled {pooled:.1%}")
    ax.set_ylabel("PICP (post-CQR)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Held-out coverage per event ({metrics['mode']})")
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "picp_per_event.png", dpi=150)
    plt.close(fig)

    trimmed = width[width["n_laps"] >= 10]  # drop sparse extreme-age bins
    colors = {"SOFT": "#cc3311", "MEDIUM": "#ddaa33", "HARD": "#4477aa"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for compound, group in trimmed.groupby("compound"):
        ax.plot(
            group["tyre_age"],
            group["upper_spread_s"],
            marker="o",
            markersize=3,
            color=colors.get(str(compound)),
            label=f"{compound}: q95 − q50",
        )
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Upper interval spread, q95 − q50 (s)")
    ax.set_title("Risk spread vs tyre age by compound — widening is the tire-cliff signal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "interval_width_vs_tyre_age.png", dpi=150)
    plt.close(fig)

    logger.info("Artifacts written to %s", out)
