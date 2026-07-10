"""Evaluation metrics: pinball loss, interval coverage, interval width."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pinball_loss(y_true: pd.Series[float], y_pred: pd.Series[float], quantile: float) -> float:
    """Mean pinball (quantile) loss; lower is better."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ok = ~np.isnan(y)
    diff = y[ok] - p[ok]
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def picp(y_true: pd.Series[float], intervals: pd.DataFrame) -> float:
    """Prediction Interval Coverage Probability: share of y inside [q05, q95]."""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(intervals["q05"], dtype=float)
    hi = np.asarray(intervals["q95"], dtype=float)
    ok = ~np.isnan(y)
    return float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok])))


def mean_interval_width(intervals: pd.DataFrame) -> float:
    """Average [q05, q95] width — sharpness; smaller is better at equal coverage."""
    return float((intervals["q95"] - intervals["q05"]).mean())
