"""Correlated-noise validation (spec §1.4): simulated spread vs a held-out race.

For every (driver, stint) in the held-out event with enough clean laps, simulate
that stint's clean-lap total with the engine's noise decomposition (model trained
without the event) and check the actual total lands in the simulated [p5, p95].
Coverage ≈ 90% means the decomposition scale is right; systematically narrow or
wide intervals show up as low/high coverage.

Run: ``python -m f1_core.simulate --event "Bahrain Grand Prix"``
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from f1_core.config import SimulationConfig, load_config
from f1_core.evaluate.calibration import BENCHMARK, _params_from_config, add_event_benchmark
from f1_core.models.cqr import fit_cqr
from f1_core.models.quantile import TARGET, QuantileLapModel, QuantileModelParams
from f1_core.simulate.engine import simulate_totals

logger = logging.getLogger(__name__)


def validate_stint_spread(
    laps: pd.DataFrame,
    event: str,
    model_params: QuantileModelParams,
    sim_config: SimulationConfig,
    min_stint_laps: int = 8,
) -> dict[str, object]:
    usable = laps[laps["is_clean"] & laps[TARGET].notna()].reset_index(drop=True)
    clean = add_event_benchmark(usable)
    train = clean[clean["event"] != event].copy()
    test = clean[clean["event"] == event].copy()
    if test.empty:
        raise ValueError(f"No clean laps for event {event!r}")

    def norm(f: pd.DataFrame) -> pd.DataFrame:
        g = f.copy()
        g[TARGET] = g[TARGET] - g[BENCHMARK]
        return g

    # CQR-calibrate on a held-out fifth of the training events: raw quantile
    # intervals undercover (~60%/lap) and would collapse the simulated spread.
    events = sorted(train["event"].unique())
    rng = np.random.default_rng(sim_config.seed)
    calib_events = set(rng.choice(events, size=max(1, len(events) // 5), replace=False))
    calib = train[train["event"].isin(calib_events)]
    train = train[~train["event"].isin(calib_events)]

    model = QuantileLapModel(model_params).fit(norm(train))
    calibrator = fit_cqr(norm(calib)[TARGET], model.predict(norm(calib)), 0.9)

    # Calibrate the correlated-noise share (spec: "calibrate the decomposition
    # weights against historical stint-time variance"). Per-stint mean residuals
    # on the calibration events measure the systematic driver/day offset the
    # conditional model cannot know for an unseen race.
    calib_n = norm(calib).copy()
    calib_n["resid"] = calib_n[TARGET] - model.predict(calib_n)["q50"].to_numpy()
    stats = (
        calib_n.groupby(["event", "driver", "stint_number"])["resid"]
        .agg(n="size", mean="mean", var="var")
        .query("n >= @min_stint_laps")
    )
    within = float(stats["var"].mean())
    between = float(stats["mean"].var())
    corr_var = max(between - within / float(stats["n"].mean()), 0.0)
    # ponytail: all correlated variance assigned to the day factor (can't split
    # day vs tire-set from one season); split when multi-season data lands.
    rho_day = min(corr_var / (corr_var + within), 0.8)
    sim_config = sim_config.model_copy(update={"rho_day": rho_day, "rho_stint": 0.0})

    rows = []
    for _, g in test.groupby(["driver", "stint_number"]):
        if len(g) < min_stint_laps:
            continue
        q = calibrator.apply(model.predict(norm(g))).to_numpy() + g[BENCHMARK].iloc[0]
        totals = simulate_totals(q, np.zeros(len(g), dtype=int), 0, sim_config, rng)
        actual = float(g[TARGET].sum())
        p5, med, p95 = np.percentile(totals, [5, 50, 95])
        rows.append(
            {
                "driver": str(g["driver"].iloc[0]),
                "stint": int(g["stint_number"].iloc[0]),
                "n_laps": len(g),
                "actual_s": actual,
                "sim_p5": float(p5),
                "sim_p95": float(p95),
                "sim_median": float(med),
                "sim_std": float(totals.std()),
                "covered": bool(p5 <= actual <= p95),
            }
        )

    frame = pd.DataFrame(rows)
    # The static fastest-lap benchmark can shift a whole held-out event (live use
    # would re-anchor on early-race pace), so split bias from spread: the spec's
    # correlated-noise check is about SPREAD.
    resid = frame["actual_s"] - frame["sim_median"]
    bias_per_lap = float(resid.sum() / frame["n_laps"].sum())
    debiased = frame["actual_s"] - bias_per_lap * frame["n_laps"]
    return {
        "event": event,
        "n_stints": len(frame),
        "coverage": float(frame["covered"].mean()),
        "event_bias_s_per_lap": bias_per_lap,
        "coverage_debiased": float(
            ((frame["sim_p5"] <= debiased) & (debiased <= frame["sim_p95"])).mean()
        ),
        "spread_ratio": float(resid.std() / np.sqrt((frame["sim_std"] ** 2).mean())),
        "calibrated_rho_day": rho_day,
        "stints": rows,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/f1_core.yaml")
    parser.add_argument("--event", default="Bahrain Grand Prix")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    laps = pd.read_parquet(config.ingest.output_dir / "laps.parquet")
    result = validate_stint_spread(
        laps, args.event, _params_from_config(config.model), config.simulation
    )
    out = config.evaluation.artifacts_dir / "mc_spread_validation.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['n_stints']} stints, coverage {result['coverage']:.1%} -> {out}")


if __name__ == "__main__":
    main()
