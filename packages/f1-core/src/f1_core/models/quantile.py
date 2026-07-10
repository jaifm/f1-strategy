"""Quantile LightGBM lap-time model (PROJECT_SPEC.md §1.3).

One booster per quantile with identical features and hyperparameters, trained
with pinball loss. A post-inference guard sorts quantile columns per row so
q05 ≤ q50 ≤ q95 and the crossing frequency is logged (spec threshold: revisit
hyperparameters above 2%).

Physics-constraint deviation from spec §1.3 (documented honestly): LightGBM
refuses ``monotone_constraints`` with the quantile objective ("Cannot use
monotone_constraints in quantile objective"), so monotonicity in ``tyre_age`` /
``fuel_proxy`` is NOT hard-enforced on the quantile boosters — only on the MSE
baseline, where the regression objective supports it. Monotonicity of the
quantile model is instead checked empirically in evaluation.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET = "lap_time_s"

#: Feature -> monotone direction (+1 = prediction non-decreasing in feature).
NUMERIC_FEATURES: dict[str, int] = {
    "tyre_age": 1,
    "fuel_proxy": 1,
    "track_temp": 0,
    "air_temp": 0,
    "stint_lap": 0,
}
CATEGORICAL_FEATURES = ["compound", "event", "team", "driver"]
FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


@dataclass(frozen=True)
class QuantileModelParams:
    """LightGBM hyperparameters shared by every quantile booster."""

    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95)
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 40
    seed: int = 7
    extra: dict[str, Any] = field(default_factory=dict)


class QuantileLapModel:
    """One LightGBM booster per quantile over the shared feature set."""

    def __init__(self, params: QuantileModelParams | None = None) -> None:
        self.params = params or QuantileModelParams()
        self.boosters: dict[float, lgb.LGBMRegressor] = {}
        self.categories: dict[str, pd.CategoricalDtype] = {}

    def _prepare(self, laps: pd.DataFrame, fit: bool) -> pd.DataFrame:
        x = laps[FEATURES].copy()
        for col in CATEGORICAL_FEATURES:
            if fit:
                self.categories[col] = pd.CategoricalDtype(sorted(x[col].dropna().unique()))
            x[col] = x[col].astype(self.categories[col])
        return x

    def fit(self, laps: pd.DataFrame) -> QuantileLapModel:
        """Fit one booster per quantile on clean laps with a lap-time label."""
        data = laps.dropna(subset=[TARGET])
        x = self._prepare(data, fit=True)
        y = data[TARGET]

        for q in self.params.quantiles:
            # NB: no monotone_constraints — unsupported by LightGBM's quantile
            # objective (see module docstring).
            booster = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=self.params.n_estimators,
                learning_rate=self.params.learning_rate,
                num_leaves=self.params.num_leaves,
                min_child_samples=self.params.min_child_samples,
                random_state=self.params.seed,
                verbose=-1,
                **self.params.extra,
            )
            booster.fit(x, y, categorical_feature=CATEGORICAL_FEATURES)
            self.boosters[q] = booster
        return self

    def predict(self, laps: pd.DataFrame) -> pd.DataFrame:
        """Predict all quantiles; returns columns like ``q05, q50, q95``.

        Applies the quantile-crossing guard (row-wise sort) and logs the
        pre-guard crossing frequency.
        """
        if not self.boosters:
            raise RuntimeError("Model not fitted.")
        x = self._prepare(laps, fit=False)
        preds = np.column_stack([self.boosters[q].predict(x) for q in self.params.quantiles])

        crossed = (np.diff(preds, axis=1) < 0).any(axis=1)
        crossing_rate = float(crossed.mean()) if len(crossed) else 0.0
        if crossing_rate > 0:
            level = logging.WARNING if crossing_rate > 0.02 else logging.INFO
            logger.log(
                level,
                "Quantile crossing on %.2f%% of rows (spec threshold 2%%)",
                100 * crossing_rate,
            )
        preds = np.sort(preds, axis=1)

        columns = [f"q{int(round(q * 100)):02d}" for q in self.params.quantiles]
        return pd.DataFrame(preds, columns=columns, index=laps.index)

    def save(self, path: Path | str) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path | str) -> QuantileLapModel:
        model = pickle.loads(Path(path).read_bytes())
        if not isinstance(model, QuantileLapModel):
            raise TypeError(f"{path} does not contain a QuantileLapModel")
        return model


class MSEBaselineModel:
    """Required naive baseline: same features/hyperparameters, MSE objective.

    This is the headline comparison for the whole paradigm (spec §1.3): the
    quantile model must beat this on pinball loss at q05/q95. Its point
    prediction stands in for every quantile when scoring pinball loss.
    """

    def __init__(self, params: QuantileModelParams | None = None) -> None:
        self.params = params or QuantileModelParams()
        self._prep = QuantileLapModel(self.params)
        self.booster: lgb.LGBMRegressor | None = None

    def fit(self, laps: pd.DataFrame) -> MSEBaselineModel:
        data = laps.dropna(subset=[TARGET])
        x = self._prep._prepare(data, fit=True)
        booster = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=self.params.n_estimators,
            learning_rate=self.params.learning_rate,
            num_leaves=self.params.num_leaves,
            min_child_samples=self.params.min_child_samples,
            monotone_constraints=[NUMERIC_FEATURES.get(f, 0) for f in FEATURES],
            random_state=self.params.seed,
            verbose=-1,
        )
        booster.fit(x, data[TARGET], categorical_feature=CATEGORICAL_FEATURES)
        self.booster = booster
        return self

    def predict(self, laps: pd.DataFrame) -> pd.Series[float]:
        if self.booster is None:
            raise RuntimeError("Model not fitted.")
        x = self._prep._prepare(laps, fit=False)
        return pd.Series(self.booster.predict(x), index=laps.index, name="mse_pred")
