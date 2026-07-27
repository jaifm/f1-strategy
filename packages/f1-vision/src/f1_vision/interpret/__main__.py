"""Build the saliency atlas: ``python -m f1_vision.interpret``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from f1_core.config import load_config as load_core_config
from f1_vision.config import load_config
from f1_vision.imaging.tab2img import ChannelScaler
from f1_vision.interpret.saliency import build_atlas, gradcam_maps
from f1_vision.train.dataset import build_samples, sectors_path
from f1_vision.train.study import load_checkpoint, split_events, take_races


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/f1_core.yaml")
    parser.add_argument("--out", default="artifacts/phase2")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    core = load_core_config(args.config)
    vision = load_config()
    ckpt_dir = core.ingest.output_dir / "vision" / "checkpoints"

    model = load_checkpoint(ckpt_dir / "cnn_lstm.pt")
    scaler = ChannelScaler.load(ckpt_dir / "scaler.pkl")

    laps = pd.read_parquet(core.ingest.output_dir / "laps.parquet")
    test_pairs = split_events(core, vision)["test"]
    sectors = pd.concat(
        [pd.read_parquet(sectors_path(core.ingest.output_dir, s, e)) for s, e in test_pairs],
        ignore_index=True,
    )
    test = build_samples(take_races(laps, test_pairs), sectors, scaler, vision)

    # tyre age at each sample's anchor lap, from the lap table
    age = test.meta.merge(
        laps[["season", "event", "driver", "lap_number", "tyre_age"]],
        on=["season", "event", "driver", "lap_number"],
        how="left",
    )["tyre_age"].to_numpy()

    maps = gradcam_maps(model, test.frames)
    table = build_atlas(maps, age, test.cliff, Path(args.out))
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
