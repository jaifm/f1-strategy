"""Build sector-table caches: ``python -m f1_vision.train [--event "Bahrain Grand Prix"]``."""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import pandas as pd

from f1_core.config import load_config as load_core_config
from f1_vision.config import load_config
from f1_vision.train.dataset import sectorize_job


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/f1_core.yaml")
    parser.add_argument("--event", default=None, help="One event; default: all with clean laps.")
    parser.add_argument(
        "--workers", type=int, default=4, help="Parallel races for the sector build."
    )
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

        out = "phase2_walkforward" if vision.train.split_mode == "temporal" else "phase2"
        report = run_study(core, vision, Path("artifacts") / out)
        print(report[report.event == "ALL"].to_string(index=False))
        return

    laps = pd.read_parquet(core.ingest.output_dir / "laps.parquet")

    pairs = (
        laps.loc[laps["is_clean"], ["season", "event"]]
        .drop_duplicates()
        .sort_values(["season", "event"])
    )
    if args.event:
        pairs = pairs[pairs["event"] == args.event]
    jobs = [
        (laps[(laps["season"] == s) & (laps["event"] == e)], int(s), str(e), core.ingest, vision)
        for s, e in pairs.itertuples(index=False)
    ]
    # Races are independent. Each worker holds a full race session in memory,
    # so the default is capped well below the core count (RAM, not CPU, binds).
    with ProcessPoolExecutor(max(1, args.workers), mp_context=get_context("spawn")) as pool:
        for line in pool.map(sectorize_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
