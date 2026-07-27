"""Build sector-table caches: ``python -m f1_vision.train [--event "Bahrain Grand Prix"]``."""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from f1_core.config import load_config as load_core_config
from f1_vision.config import load_config
from f1_vision.train.dataset import build_event_sectors


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/f1_core.yaml")
    parser.add_argument("--event", default=None, help="One event; default: all with clean laps.")
    parser.add_argument(
        "--study", action="store_true", help="Run the §2.5 baseline study on cached sectors."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    core = load_core_config(args.config)
    vision = load_config()

    if args.study:
        from pathlib import Path

        from f1_vision.train.study import run_study

        report = run_study(core, vision, Path("artifacts/phase2"))
        print(report[report.event == "ALL"].to_string(index=False))
        return

    laps = pd.read_parquet(core.ingest.output_dir / "laps.parquet")

    clean_events = laps[laps["is_clean"]].groupby("event").size()
    events = [args.event] if args.event else clean_events[clean_events > 0].index.tolist()
    for event in events:
        season = int(laps.loc[laps["event"] == event, "season"].iloc[0])
        sectors = build_event_sectors(laps, season, event, core.ingest, vision)
        n_laps = len(sectors[["driver", "lap_number"]].drop_duplicates())
        print(f"{event}: {n_laps} laps sectorized")


if __name__ == "__main__":
    main()
