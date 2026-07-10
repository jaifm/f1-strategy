"""Tests for quantile model, event splits, CQR calibration, and metrics."""

import numpy as np
import pandas as pd
import pytest

from f1_core.evaluate.metrics import mean_interval_width, picp, pinball_loss
from f1_core.models.cqr import fit_cqr
from f1_core.models.quantile import MSEBaselineModel, QuantileLapModel, QuantileModelParams
from f1_core.models.splits import split_by_event

from .synth import make_synthetic

FAST_PARAMS = QuantileModelParams(n_estimators=80, min_child_samples=20)


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
