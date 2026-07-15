"""MC engine tests: speed, correctness of sampling, strategy comparison."""

import time

import numpy as np
import pandas as pd
import pytest

from f1_core.config import SimulationConfig
from f1_core.simulate.engine import (
    RaceContext,
    RaceResult,
    Stint,
    simulate_race,
    simulate_totals,
    strategy_features,
)

RNG = np.random.default_rng(0)
CTX = RaceContext(
    event="Testville",
    season=2024,
    driver="A",
    team="T1",
    track_temp=35.0,
    air_temp=25.0,
    total_laps=57,
)


class FakeModel:
    """q50 = 90 + 0.01*tyre_age (low deg, so pit loss dominates), ±1s band."""

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        q50 = 90.0 + 0.01 * features["tyre_age"].to_numpy()
        return pd.DataFrame({"q05": q50 - 1, "q50": q50, "q95": q50 + 1})


def quantiles(n_laps: int = 57) -> np.ndarray:
    q50 = np.full(n_laps, 90.0)
    return np.column_stack([q50 - 1, q50, q50 + 1])


def test_strategy_features_stint_structure() -> None:
    strategy = (Stint("SOFT", 20), Stint("HARD", 57))
    f = strategy_features(strategy, CTX)
    assert len(f) == 57
    assert f["tyre_age"].iloc[19] == 20 and f["tyre_age"].iloc[20] == 1  # reset at stop
    assert set(f["compound"].unique()) == {"SOFT", "HARD"}
    assert f["fuel_proxy"].iloc[0] == 1.0


def test_strategy_must_cover_race() -> None:
    with pytest.raises(ValueError):
        strategy_features((Stint("SOFT", 30),), CTX)


def test_speed_2000_sims_under_2s() -> None:
    config = SimulationConfig(n_sims=2000)
    start = time.perf_counter()
    totals = simulate_totals(quantiles(), np.zeros(57, dtype=int), 1, config, RNG)
    elapsed = time.perf_counter() - start
    assert totals.shape == (2000,)
    assert elapsed < 2.0


def test_totals_plausible_and_pit_loss_added() -> None:
    config = SimulationConfig(n_sims=5000, seed=1)
    no_stop = simulate_totals(
        quantiles(), np.zeros(57, dtype=int), 0, config, np.random.default_rng(1)
    )
    one_stop = simulate_totals(
        quantiles(), np.zeros(57, dtype=int), 1, config, np.random.default_rng(1)
    )
    assert abs(no_stop.mean() - 57 * 90.0) < 1.0
    assert abs((one_stop.mean() - no_stop.mean()) - config.pit_loss_mean_s) < 1.0


def test_correlated_noise_inflates_race_variance() -> None:
    independent = SimulationConfig(n_sims=4000, rho_day=0.0, rho_stint=0.0, seed=2)
    correlated = SimulationConfig(n_sims=4000, rho_day=0.4, rho_stint=0.2, seed=2)
    stint_ids = np.zeros(57, dtype=int)
    v_ind = simulate_totals(quantiles(), stint_ids, 0, independent, np.random.default_rng(2)).var()
    v_cor = simulate_totals(quantiles(), stint_ids, 0, correlated, np.random.default_rng(2)).var()
    assert v_cor > 2 * v_ind  # day factor doesn't average out over 57 laps


def test_safety_car_not_implemented() -> None:
    config = SimulationConfig(sc_rate_per_lap=0.1)
    with pytest.raises(NotImplementedError):
        simulate_totals(quantiles(), np.zeros(57, dtype=int), 0, config, RNG)


def test_simulate_race_api() -> None:
    strategies = {
        "one_stop": (Stint("MEDIUM", 30), Stint("HARD", 57)),
        "two_stop": (Stint("SOFT", 18), Stint("SOFT", 38), Stint("SOFT", 57)),
    }
    result = simulate_race(strategies, FakeModel(), CTX, SimulationConfig(n_sims=1000))  # type: ignore[arg-type]
    assert isinstance(result, RaceResult)
    probs = [result.win_probability(n) for n in strategies]
    assert abs(sum(probs) - 1.0) < 1e-9
    report = result.percentile_report()
    assert list(report.columns) == ["p5", "p50", "p95"]
    assert (report["p5"] <= report["p95"]).all()
    # two_stop pays an extra pit loss with identical pace model -> should lose
    assert result.win_probability("one_stop") > 0.8
    assert result.total_time_distribution("one_stop").shape == (1000,)
