"""Run the ETL: ``uv run python -m f1_core.ingest [--config configs/f1_core.yaml]``."""

from __future__ import annotations

import argparse
import logging

from f1_core.config import load_config
from f1_core.ingest.etl import run_etl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="f1-core-ingest", description=__doc__)
    parser.add_argument("--config", default=None, help="Path to YAML config (default: built-in).")
    parser.add_argument(
        "--seasons", type=int, nargs="+", default=None, help="Override configured seasons."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config).ingest
    if args.seasons:
        config = config.model_copy(update={"seasons": args.seasons})

    laps, attrition = run_etl(config)
    print(attrition.to_string(index=False))
    print(f"\n{len(laps)} laps total, {int(laps['is_clean'].sum())} clean.")


if __name__ == "__main__":
    main()
