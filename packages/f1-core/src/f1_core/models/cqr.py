"""Conformalized Quantile Regression (PROJECT_SPEC.md §1.3).

CQR turns the raw [q05, q95] interval into one with finite-sample coverage
guarantees: conformity scores from a held-out calibration set (whole events)
give a single margin that symmetrically widens (or narrows, if negative) the
interval so empirical coverage hits the target (default 90%).

Reference: Romano, Patterson & Candès, "Conformalized Quantile Regression"
(NeurIPS 2019).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CQRCalibrator:
    """Additive conformity margin for a fixed target coverage."""

    margin: float
    target_coverage: float

    def apply(self, intervals: pd.DataFrame) -> pd.DataFrame:
        """Widen ``q05``/``q95`` by the margin; ``q50`` and other columns pass through."""
        adjusted = intervals.copy()
        adjusted["q05"] = adjusted["q05"] - self.margin
        adjusted["q95"] = adjusted["q95"] + self.margin
        return adjusted


def fit_cqr(
    y_true: pd.Series[float],
    intervals: pd.DataFrame,
    target_coverage: float = 0.9,
) -> CQRCalibrator:
    """Compute the CQR margin from calibration-set predictions.

    Conformity score per lap: ``max(q05 - y, y - q95)`` (positive when the true
    value falls outside the interval). The margin is the
    ``(1 - alpha)(1 + 1/n)`` empirical quantile of the scores.
    """
    if not 0 < target_coverage < 1:
        raise ValueError("target_coverage must be in (0, 1)")
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(intervals["q05"], dtype=float)
    hi = np.asarray(intervals["q95"], dtype=float)
    ok = ~np.isnan(y)
    y, lo, hi = y[ok], lo[ok], hi[ok]
    n = len(y)
    if n == 0:
        raise ValueError("Calibration set is empty.")

    scores = np.maximum(lo - y, y - hi)
    level = min(1.0, (1 - (1 - target_coverage)) * (1 + 1 / n))
    margin = float(np.quantile(scores, level, method="higher"))
    return CQRCalibrator(margin=margin, target_coverage=target_coverage)
