"""Orchestrator + fusion rule (spec §3.1–3.2).

f1-core provides the pre-race strategy anchor (quantile model + CQR + MC);
the live modifier is PERSISTENCE — the §2.5 study's best model: the driver's
current clean-lap delta vs their stint baseline, and a cliff score derived
from it. Fusion changes the MC engine's *inputs*, never the GBM:

1. Median shift: the current delta is added to the q50 path for the remaining
   laps of the current stint, decaying with half-life ``half_life_laps``.
2. Spread inflation: (q95 − q50) for those laps is widened by
   ``1 + alpha * p_cliff``.
3. A fast MC (``n_sims``) re-runs per lap; the optimal-stop window is the
   p25–p75 of the per-simulation argmin over candidate stop laps (common
   random numbers across candidates).

No torch in this process: persistence needs none, and importing torch next to
LightGBM deadlocks (dual libomp, see artifacts/phase2/FINDINGS.md) — which is
also why the dashboard's Grad-CAM overlay is deferred (documented deviation).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from f1_core.config import F1CoreConfig, FusionConfig
from f1_core.evaluate.calibration import BENCHMARK, _params_from_config, add_event_benchmark
from f1_core.models.cqr import CQRCalibrator, fit_cqr
from f1_core.models.quantile import TARGET, QuantileLapModel
from f1_core.models.splits import split_by_event
from f1_core.simulate.engine import RaceContext, Stint, lap_quantiles, simulate_totals
from f1_pitwall.orchestrator.replay import LapCompleted


def train_strategy_model(
    clean_laps: pd.DataFrame, config: F1CoreConfig
) -> tuple[QuantileLapModel, CQRCalibrator]:
    """Pre-race anchor: quantile model + CQR, event-split calibration."""
    split = split_by_event(
        clean_laps,
        calibration_fraction=0.2,
        test_fraction=0.0,
        seed=config.evaluation.split_seed,
    )

    def norm(f: pd.DataFrame) -> pd.DataFrame:
        g = f.copy()
        g[TARGET] = g[TARGET] - g[BENCHMARK]
        return g

    model = QuantileLapModel(_params_from_config(config.model)).fit(norm(split.train))
    calibrator = fit_cqr(
        norm(split.calibration)[TARGET],
        model.predict(norm(split.calibration)),
        config.evaluation.target_coverage,
    )
    return model, calibrator


@dataclass
class DriverState:
    """Persistence state: stint baseline and current delta."""

    config: FusionConfig
    stint: float | None = None
    baseline_times: list[float] = field(default_factory=list)
    current_delta: float = 0.0

    def update(self, lap: LapCompleted) -> None:
        if lap.stint_number != self.stint:  # new tires: reset
            self.stint = lap.stint_number
            self.baseline_times = []
            self.current_delta = 0.0
        if not lap.is_clean or lap.lap_time_s is None:
            return
        if len(self.baseline_times) < self.config.baseline_laps:
            self.baseline_times.append(lap.lap_time_s)
            return
        self.current_delta = lap.lap_time_s - float(np.mean(self.baseline_times))

    @property
    def p_cliff(self) -> float:
        return float(np.clip(self.current_delta / self.config.cliff_score_scale_s, 0.0, 1.0))


def apply_fusion(
    quantiles: np.ndarray,
    current_lap: int,
    stop_lap: int,
    delta: float,
    p_cliff: float,
    config: FusionConfig,
) -> np.ndarray:
    """Fusion rule on a strategy's [total_laps, 3] quantile array (rows = laps 1..N)."""
    fused = quantiles.copy()
    laps_ahead = np.arange(1, stop_lap - current_lap + 1, dtype=float)
    rows = slice(current_lap, stop_lap)  # 0-indexed rows for laps current_lap+1 .. stop_lap
    decay = 0.5 ** ((laps_ahead - 1) / config.half_life_laps)
    shift = delta * decay
    upper = (fused[rows, 2] - fused[rows, 1]) * (1 + config.alpha * p_cliff)
    fused[rows, 1] += shift
    fused[rows, 0] += shift
    fused[rows, 2] = fused[rows, 1] + upper
    return fused


@dataclass
class LapUpdate:
    """Per-lap output for the focal driver."""

    lap_number: int
    current_delta: float
    p_cliff: float
    stop_p25: float
    stop_p50: float
    stop_p75: float
    compute_s: float


class PitwallOrchestrator:
    """Feeds replay events into fusion state and re-runs the fast MC per lap."""

    def __init__(
        self,
        model: QuantileLapModel,
        calibrator: CQRCalibrator,
        ctx: RaceContext,
        config: F1CoreConfig,
        fusion_enabled: bool = True,
    ) -> None:
        self.model, self.calibrator, self.ctx, self.config = model, calibrator, ctx, config
        self.fusion_enabled = fusion_enabled
        self.states: dict[str, DriverState] = {}
        self.positions: dict[str, float] = {}

    def state(self, driver: str) -> DriverState:
        return self.states.setdefault(driver, DriverState(self.config.fusion))

    def on_lap(self, lap: LapCompleted) -> None:
        self.state(lap.driver).update(lap)
        if lap.position is not None:
            self.positions[lap.driver] = lap.position

    def stop_window(self, driver: str, current_lap: int, compound: str) -> LapUpdate | None:
        """Optimal-stop band over candidate stop laps for the focal driver."""
        cfg = self.config.fusion
        total = self.ctx.total_laps
        candidates = list(
            range(current_lap + 1, min(current_lap + cfg.candidate_window_laps, total - 3))
        )
        if len(candidates) < 2:
            return None
        start = time.perf_counter()
        state = self.state(driver)
        delta = state.current_delta if self.fusion_enabled else 0.0
        p_cliff = state.p_cliff if self.fusion_enabled else 0.0

        sim_config = self.config.simulation.model_copy(update={"n_sims": cfg.n_sims})
        totals = []
        for stop in candidates:
            strategy = (Stint(compound, stop), Stint(cfg.second_compound, total))
            q = lap_quantiles(self.model, strategy, self.ctx, self.calibrator)
            q = apply_fusion(q, current_lap, stop, delta, p_cliff, cfg)
            remaining = q[current_lap:]
            end_laps = np.array([stop, total])
            stint_ids = np.searchsorted(end_laps, np.arange(current_lap + 1, total + 1))
            totals.append(
                simulate_totals(
                    remaining, stint_ids, 1, sim_config, np.random.default_rng(sim_config.seed)
                )
            )
        stacked = np.stack(totals)  # [n_candidates, n_sims]
        best = np.array(candidates)[np.argmin(stacked, axis=0)]
        p25, p50, p75 = np.percentile(best, [25, 50, 75])
        return LapUpdate(
            lap_number=current_lap,
            current_delta=state.current_delta,
            p_cliff=state.p_cliff,
            stop_p25=float(p25),
            stop_p50=float(p50),
            stop_p75=float(p75),
            compute_s=time.perf_counter() - start,
        )


def build_race_context(
    clean_laps: pd.DataFrame, event: str, season: int, driver: str
) -> RaceContext:
    """Race context from a race's clean laps (same recipe as the v1 dashboard)."""
    laps_e = add_event_benchmark(
        clean_laps[(clean_laps["event"] == event) & (clean_laps["season"] == season)]
    )
    return RaceContext(
        event=event,
        season=season,
        driver=driver,
        team=str(laps_e.loc[laps_e["driver"] == driver, "team"].mode().iloc[0]),
        track_temp=float(laps_e["track_temp"].median()),
        air_temp=float(laps_e["air_temp"].median()),
        total_laps=int(laps_e["lap_number"].max()),
        benchmark_s=float(laps_e[BENCHMARK].iloc[0]),
    )
