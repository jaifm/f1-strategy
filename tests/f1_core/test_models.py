"""Tests for quantile model, event splits, CQR calibration, and metrics.

Synthetic laps have a known generating process (linear tyre-age slope + fuel
effect + Gaussian noise) so physics constraints and coverage are checkable.
"""

import numpy as np
import pandas as pd
import pytest

from f1_core.evaluate.metrics import mean_interval_width, picp, pinball_loss
from f1_core.models.cqr import fit_cqr
from f1_core.models.quantile import MSEBaselineModel, QuantileLapModel, QuantileModelParams
from f1_core.models.splits import split_by_event

RNG = np.random.default_rng(42)
FAST_PARAMS = QuantileModelParams(n_estimators=80, min_child_samples=20)


def make_synthetic(n_events: int = 8, race_laps: int = 50) -> pd.DataFrame:
    """Laps for ``n_events`` races, 4 drivers, two stints per race."""
    rows = []
    for e in range(n_events):
        for driver, team in [("A", "T1"), ("B", "T1"), ("C", "T2"), ("D", "T2")]:
            stop = race_laps // 2 + int(RNG.integers(-5, 5))
            for lap in range(2, race_laps + 1):
                stint = 1 if lap <= stop else 2
                tyre_age = lap if stint == 1 else lap - stop
                fuel = (race_laps - lap + 1) / race_laps
                rows.append(
                    {
                        "season": 2024,
                        "event": f"Race{e}",
                        "driver": driver,
                        "team": team,
                        "lap_number": lap,
                        "compound": "SOFT" if stint == 1 else "HARD",
                        "tyre_age": float(tyre_age),
                        "fuel_proxy": fuel,
                        "track_temp": 35.0 + e,
                        "air_temp": 25.0 + e,
                        "stint_lap": tyre_age,
                        # Small per-event offset: held-out events are UNSEEN
                        # categories, so a large offset breaks exchangeability
                        # and no conformal method could cover (the real-data
                        # answer is per-event normalization, spec §1.3).
                        "lap_time_s": 90.0
                        + 0.05 * e
                        + 0.12 * tyre_age
                        + 2.0 * fuel
                        + float(RNG.normal(0, 0.35)),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic() -> pd.DataFrame:
    return make_synthetic()


@pytest.fixture(scope="module")
def fitted(synthetic: pd.DataFrame) -> QuantileLapModel:
    return QuantileLapModel(FAST_PARAMS).fit(synthetic)


class TestSplits:
    def test_events_disjoint_and_complete(self, synthetic: pd.DataFrame) -> None:
        split = split_by_event(synthetic, calibration_fraction=0.25, test_fraction=0.25, seed=1)
        sets = {
            name: set(map(tuple, part[["season", "event"]].drop_duplicates().to_numpy()))
            for name, part in [
                ("train", split.train),
                ("calib", split.calibration),
                ("test", split.test),
            ]
        }
        assert not sets["train"] & sets["calib"]
        assert not sets["train"] & sets["test"]
        assert not sets["calib"] & sets["test"]
        assert len(split.train) + len(split.calibration) + len(split.test) == len(synthetic)

    def test_infeasible_fractions_raise(self, synthetic: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            split_by_event(synthetic, calibration_fraction=0.6, test_fraction=0.5, seed=1)


class TestQuantileModel:
    def test_quantiles_never_cross(self, fitted: QuantileLapModel, synthetic: pd.DataFrame) -> None:
        preds = fitted.predict(synthetic)
        assert (preds["q05"] <= preds["q50"]).all()
        assert (preds["q50"] <= preds["q95"]).all()

    def test_degradation_trend_in_tyre_age(
        self, fitted: QuantileLapModel, synthetic: pd.DataFrame
    ) -> None:
        # Hard monotone constraints are unavailable for LightGBM quantile
        # boosters (see quantile.py docstring); check the learned physical
        # trend instead: old tires are predicted meaningfully slower than fresh.
        # Probe at mid-fuel so the grid stays on the training-data manifold.
        grid = pd.concat([synthetic.iloc[[0]]] * 30, ignore_index=True)
        grid["tyre_age"] = np.arange(1.0, 31.0)
        grid["stint_lap"] = np.arange(1, 31)
        grid["fuel_proxy"] = 0.6
        q50 = fitted.predict(grid)["q50"]
        fresh, old = q50.iloc[:5].mean(), q50.iloc[-5:].mean()
        assert old > fresh + 1.0  # true slope: 0.12 s/lap * ~27 laps ≈ 3.2 s

    def test_predict_before_fit_raises(self, synthetic: pd.DataFrame) -> None:
        with pytest.raises(RuntimeError):
            QuantileLapModel(FAST_PARAMS).predict(synthetic)

    def test_save_load_roundtrip(
        self, fitted: QuantileLapModel, synthetic: pd.DataFrame, tmp_path: object
    ) -> None:
        path = f"{tmp_path}/model.pkl"
        fitted.save(path)
        reloaded = QuantileLapModel.load(path)
        pd.testing.assert_frame_equal(reloaded.predict(synthetic), fitted.predict(synthetic))

    def test_beats_mse_baseline_on_tail_pinball(self, synthetic: pd.DataFrame) -> None:
        split = split_by_event(synthetic, calibration_fraction=0.0, test_fraction=0.25, seed=2)
        quantile = QuantileLapModel(FAST_PARAMS).fit(split.train)
        baseline = MSEBaselineModel(FAST_PARAMS).fit(split.train)
        y = split.test["lap_time_s"]
        q_preds = quantile.predict(split.test)
        b_pred = baseline.predict(split.test)
        for q, col in [(0.05, "q05"), (0.95, "q95")]:
            assert pinball_loss(y, q_preds[col], q) < pinball_loss(y, b_pred, q)


class TestCQR:
    def test_post_cqr_coverage_near_target(self, synthetic: pd.DataFrame) -> None:
        split = split_by_event(synthetic, calibration_fraction=0.25, test_fraction=0.25, seed=3)
        model = QuantileLapModel(FAST_PARAMS).fit(split.train)
        calibrator = fit_cqr(split.calibration["lap_time_s"], model.predict(split.calibration), 0.9)
        adjusted = calibrator.apply(model.predict(split.test))
        coverage = picp(split.test["lap_time_s"], adjusted)
        assert 0.84 <= coverage <= 0.97

    def test_margin_widens_interval(self, synthetic: pd.DataFrame) -> None:
        intervals = pd.DataFrame({"q05": [90.0], "q50": [92.0], "q95": [94.0]})
        from f1_core.models.cqr import CQRCalibrator

        wider = CQRCalibrator(margin=0.5, target_coverage=0.9).apply(intervals)
        assert wider["q05"].iloc[0] == 89.5
        assert wider["q95"].iloc[0] == 94.5
        assert wider["q50"].iloc[0] == 92.0


class TestMetrics:
    def test_pinball_loss_asymmetry(self) -> None:
        y = pd.Series([100.0])
        over = pd.Series([101.0])  # prediction above truth
        # At q=0.95 under-prediction is penalized 19x more than over-prediction.
        assert pinball_loss(y, over, 0.95) == pytest.approx(0.05)
        under = pd.Series([99.0])
        assert pinball_loss(y, under, 0.95) == pytest.approx(0.95)

    def test_picp_and_width(self) -> None:
        y = pd.Series([1.0, 2.0, 3.0, 10.0])
        intervals = pd.DataFrame({"q05": [0.0] * 4, "q95": [5.0] * 4})
        assert picp(y, intervals) == pytest.approx(0.75)
        assert mean_interval_width(intervals) == pytest.approx(5.0)
