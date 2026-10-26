"""Pit-wall CLI (spec §3.4): ``pitwall --race "Bahrain 2024" --speed 10``.

Default launches the live Streamlit dashboard. ``--headless`` runs the full
replay + fusion loop without UI and prints acceptance stats (per-lap latency,
pacing drift, final stop window); ``--csv`` dumps per-lap updates for the
case-study artifacts.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

DASHBOARD = Path(__file__).parent / "dashboard" / "live.py"


def resolve_event(laps: pd.DataFrame, race: str) -> tuple[str, int]:
    """Match "Bahrain 2024" / "bahrain" against known (event, season) pairs.

    Without a year token, the most recent season of the matched event is used.
    """
    year_match = re.search(r"\b(19|20)\d{2}\b", race)
    token = re.sub(r"\b(19|20)\d{2}\b", "", race).strip().lower()
    events = sorted(laps["event"].unique())
    matches = [e for e in events if token in e.lower()]
    if len(matches) != 1:
        raise SystemExit(f"Race {race!r} matched {matches or 'nothing'}; known: {events}")
    seasons = sorted(laps.loc[laps["event"] == matches[0], "season"].unique())
    season = int(year_match.group()) if year_match else int(seasons[-1])
    if season not in seasons:
        raise SystemExit(f"{matches[0]} has no {season} data; seasons: {seasons}")
    return matches[0], season


def run_headless(args: argparse.Namespace) -> None:
    import numpy as np

    from f1_core.config import load_config
    from f1_core.evaluate.calibration import add_event_benchmark
    from f1_core.models.quantile import TARGET
    from f1_pitwall.orchestrator.replay import LapCompleted, replay_race
    from f1_pitwall.orchestrator.service import (
        PitwallOrchestrator,
        build_race_context,
        prior_races,
        train_strategy_model,
    )

    config = load_config(args.config)
    laps = pd.read_parquet(config.ingest.output_dir / "laps.parquet")
    event, season = resolve_event(laps, args.race)
    clean = add_event_benchmark(
        laps[laps["is_clean"] & laps[TARGET].notna()].reset_index(drop=True)
    )

    # Default: anchor trained only on races before this one (no lookahead).
    train_laps = clean if args.allow_future else prior_races(clean, season, event)
    model, calibrator = train_strategy_model(train_laps, config)
    ctx = build_race_context(clean, event, season, args.driver)
    orchestrator = PitwallOrchestrator(
        model, calibrator, ctx, config, fusion_enabled=not args.no_fusion
    )

    event_laps = laps[(laps["event"] == event) & (laps["season"] == season)]
    speed = None if args.speed <= 0 else args.speed
    updates, session_end = [], 0.0
    wall_start = time.monotonic()
    for ev in replay_race(event_laps, speed=speed):
        if not isinstance(ev, LapCompleted):
            continue
        session_end = max(session_end, ev.session_time_s)
        orchestrator.on_lap(ev)
        if ev.driver == args.driver and ev.compound is not None:
            update = orchestrator.stop_window(args.driver, ev.lap_number, ev.compound)
            if update is not None:
                updates.append(update)
    wall = time.monotonic() - wall_start

    lat = np.array([u.compute_s for u in updates])
    drift = wall - session_end / speed if speed else float("nan")
    print(
        f"event: {event} {season}  driver: {args.driver}  fusion: {not args.no_fusion}"
        f"  anchor: {'ALL races (leaky)' if args.allow_future else 'prior races only'}"
    )
    print(f"laps replayed: {event_laps['lap_number'].nunique()}  updates: {len(updates)}")
    print(
        f"latency mean {lat.mean():.3f}s  p95 {np.percentile(lat, 95):.3f}s  max {lat.max():.3f}s"
    )
    if speed:
        print(f"pacing drift at {speed:g}x: {drift:+.2f}s over {session_end / speed:.0f}s")
    if updates:
        u = updates[-1]
        print(f"final window: p25 {u.stop_p25:.0f} / p50 {u.stop_p50:.0f} / p75 {u.stop_p75:.0f}")
    if args.csv:
        pd.DataFrame([u.__dict__ for u in updates]).to_csv(args.csv, index=False)
        print(f"updates -> {args.csv}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pitwall", description="F1 pit-wall replay dashboard.")
    parser.add_argument("--race", required=True, help='e.g. "Bahrain 2024".')
    parser.add_argument("--speed", type=float, default=1.0, help="Multiplier; <=0 = max speed.")
    parser.add_argument("--driver", default="VER", help="Focal driver for the strategy window.")
    parser.add_argument("--config", default="configs/f1_core.yaml")
    parser.add_argument("--headless", action="store_true", help="No UI; print acceptance stats.")
    parser.add_argument("--no-fusion", action="store_true", help="Disable the live modifier.")
    parser.add_argument("--csv", default=None, help="Write per-lap updates to this CSV.")
    parser.add_argument(
        "--allow-future",
        action="store_true",
        help="Train the anchor on ALL races incl. this and later ones (leaky; for comparison).",
    )
    args = parser.parse_args(argv)

    if args.headless:
        run_headless(args)
        return
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(DASHBOARD),
            "--",
            "--race",
            args.race,
            "--speed",
            str(args.speed),
            "--driver",
            args.driver,
            "--config",
            args.config,
        ],
    )


if __name__ == "__main__":
    main()
