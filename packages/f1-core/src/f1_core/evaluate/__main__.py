"""Run the calibration study: ``python -m f1_core.evaluate --config configs/f1_core.yaml``."""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from f1_core.config import load_config
from f1_core.evaluate.calibration import run_calibration_study


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="f1-core-evaluate", description=__doc__)
    parser.add_argument("--config", default=None, help="Path to YAML config (default: built-in).")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Override config: enable per-event lap-time normalization.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    eval_config = config.evaluation
    if args.normalize:
        eval_config = eval_config.model_copy(update={"per_event_normalization": True})

    laps = pd.read_parquet(config.ingest.output_dir / "laps.parquet")
    report = run_calibration_study(laps, config.model, eval_config)
    print(json.dumps(report.metrics, indent=2))


if __name__ == "__main__":
    main()
