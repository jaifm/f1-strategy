"""Vectorized Monte Carlo race engine (spec §1.4).

Per-lap times are sampled from an asymmetric triangular distribution
(min=q05, mode=q50, max=q95) predicted for that lap's state, with correlated
noise decomposed per simulation:

- ``day``: one N(0,1) draw per sim, applied to every lap ("on it today or not")
- ``stint``: one draw per (sim, stint) — tire-set-to-set variation
- ``lap``: the triangular deviation, scaled down

Variance shares (``rho_day``, ``rho_stint``, remainder = lap) keep the per-lap
marginal variance ≈ the calibrated quantile spread; ``validate.py`` checks the
resulting spread against a held-out race.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1_core.config import SimulationConfig
from f1_core.models.cqr import CQRCalibrator
from f1_core.models.quantile import QuantileLapModel


@dataclass(frozen=True)
class Stint:
    compound: str
    end_lap: int  # last lap on this tire set; final stint ends on total_laps


Strategy = tuple[Stint, ...]


@dataclass(frozen=True)
class RaceContext:
    """Fixed per-race inputs for feature building."""

    event: str
    season: int
    driver: str
    team: str
    track_temp: float
    air_temp: float
    total_laps: int
    benchmark_s: float = 0.0  # event base pace; 0.0 → totals are relative


def strategy_features(strategy: Strategy, ctx: RaceContext) -> pd.DataFrame:
    """One feature row per lap, matching the model's training schema."""
    if strategy[-1].end_lap != ctx.total_laps:
        raise ValueError("Last stint must end on the final lap")
    rows = []
    start = 1
    for stint in strategy:
        if stint.end_lap < start:
            raise ValueError(f"Stint ending lap {stint.end_lap} starts after lap {start}")
        for lap in range(start, stint.end_lap + 1):
            rows.append(
                {
                    "tyre_age": float(lap - start + 1),
                    "compound": stint.compound,
                    "fuel_proxy": (ctx.total_laps - lap + 1) / ctx.total_laps,
                    "track_temp": ctx.track_temp,
                    "air_temp": ctx.air_temp,
                    "stint_lap": lap - start + 1,
                    "event": ctx.event,
                    "team": ctx.team,
                    "driver": ctx.driver,
                }
            )
        start = stint.end_lap + 1
    return pd.DataFrame(rows)


def lap_quantiles(
    model: QuantileLapModel,
    strategy: Strategy,
    ctx: RaceContext,
    calibrator: CQRCalibrator | None = None,
) -> np.ndarray:
    """[n_laps, 3] array of (q05, q50, q95) lap times for a strategy.

    Pass the CQR calibrator from the calibration study: raw quantile intervals
    undercover (~60% per lap) and simulating from them collapses the spread.
    """
    intervals = model.predict(strategy_features(strategy, ctx))
    if calibrator is not None:
        intervals = calibrator.apply(intervals)
    return intervals.to_numpy() + ctx.benchmark_s


def simulate_totals(
    quantiles: np.ndarray,
    stint_ids: np.ndarray,
    n_stops: int,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Total race times, shape [n_sims]. Fully vectorized — no loop over sims."""
    if config.sc_rate_per_lap > 0:
        raise NotImplementedError("Safety-car modeling is v2 (spec: interface must accept it)")
    a, m, b = quantiles.T
    mu = (a + m + b) / 3
    sigma = np.sqrt((a**2 + m**2 + b**2 - a * m - a * b - m * b) / 18)
    rho_lap = 1.0 - config.rho_day - config.rho_stint
    if rho_lap <= 0:
        raise ValueError("rho_day + rho_stint must be < 1")

    n = config.n_sims
    tri = rng.triangular(a, m, b, size=(n, len(a)))
    lap_part = np.sqrt(rho_lap) * (tri - mu)
    # ponytail: day/stint factors use the mean lap sigma, not per-lap sigma;
    # refine if validate.py shows scale drift.
    day_part = rng.standard_normal((n, 1)) * np.sqrt(config.rho_day) * sigma.mean()
    stint_draws = rng.standard_normal((n, int(stint_ids.max()) + 1))
    stint_part = stint_draws[:, stint_ids] * np.sqrt(config.rho_stint) * sigma.mean()

    totals: np.ndarray = (mu + lap_part + day_part + stint_part).sum(axis=1)
    if n_stops:
        totals += rng.normal(config.pit_loss_mean_s, config.pit_loss_std_s, (n, n_stops)).sum(
            axis=1
        )
    return totals


@dataclass
class RaceResult:
    """Simulated total-time distributions keyed by strategy name."""

    totals: dict[str, np.ndarray]

    def total_time_distribution(self, name: str) -> np.ndarray:
        return self.totals[name]

    def win_probability(self, name: str) -> float:
        """P(this strategy has the lowest total time across all simulated)."""
        stacked = np.stack(list(self.totals.values()))
        winner = np.argmin(stacked, axis=0)
        return float((winner == list(self.totals).index(name)).mean())

    def percentile_report(self) -> pd.DataFrame:
        rows = {
            name: dict(zip(["p5", "p50", "p95"], np.percentile(t, [5, 50, 95]), strict=True))
            for name, t in self.totals.items()
        }
        return pd.DataFrame(rows).T.rename_axis("strategy")


def simulate_race(
    strategies: dict[str, Strategy],
    model: QuantileLapModel,
    ctx: RaceContext,
    config: SimulationConfig,
    rng: np.random.Generator | None = None,
    calibrator: CQRCalibrator | None = None,
) -> RaceResult:
    # Common random numbers: every strategy replays the same simulated "days",
    # so the day factor cancels in comparisons instead of blurring them.
    base_seed = int((rng or np.random.default_rng(config.seed)).integers(2**32))
    totals = {}
    for name, strategy in strategies.items():
        q = lap_quantiles(model, strategy, ctx, calibrator)
        end_laps = np.array([s.end_lap for s in strategy])
        stint_ids = np.searchsorted(end_laps, np.arange(1, ctx.total_laps + 1))
        totals[name] = simulate_totals(
            q, stint_ids, len(strategy) - 1, config, np.random.default_rng(base_seed)
        )
    return RaceResult(totals=totals)
