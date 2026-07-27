"""Dataset builder (spec §2.2/2.4): telemetry → per-lap sector tables → samples.

Two stages, so the expensive telemetry pass is cached:

1. ``build_event_sectors``: for every lap of one race, compute proxies and the
   K-mini-sector table; written UN-normalized to
   ``data/processed/vision/sectors_<event>.parquet`` (normalization must use
   train-split statistics, which are only known at training time).
2. ``build_samples``: join sector tables with labels, fit-time ``ChannelScaler``
   applied per event, and window the last L laps of each stint into tensors of
   shape ``[n, L, K, 5, 5]`` (float32) plus label vectors.

Windows never cross a stint boundary (tire state resets at the stop), so the
anchor lap needs ``stint_lap >= L``. Laps whose telemetry is too short for the
SavGol window are skipped and counted.

CLI: ``uv run python -m f1_vision.train --event "Bahrain Grand Prix"`` (one
event) or with no ``--event`` for every event with clean laps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from f1_core.config import IngestConfig
from f1_core.ingest.telemetry import lap_telemetry, load_race_telemetry_session
from f1_vision.config import F1VisionConfig
from f1_vision.imaging.tab2img import CONTEXT_CHANNELS, ChannelScaler, lap_frames, minisector_table
from f1_vision.proxies.physics import compute_proxies
from f1_vision.train.labels import add_labels

logger = logging.getLogger(__name__)

META_COLUMNS = ["season", "event", "driver", "lap_number", "stint_number", "pace_delta_current"]


def sectors_path(output_dir: Path, season: int, event: str) -> Path:
    return output_dir / "vision" / f"sectors_{season}_{event.replace(' ', '_')}.parquet"


def build_event_sectors(
    laps: pd.DataFrame,
    season: int,
    event: str,
    ingest_config: IngestConfig,
    vision_config: F1VisionConfig,
) -> pd.DataFrame:
    """Sector tables for every lap of one race; cached to parquet."""
    path = sectors_path(ingest_config.output_dir, season, event)
    if path.exists():
        return pd.read_parquet(path)

    session = load_race_telemetry_session(ingest_config, season, event)
    event_laps = laps[(laps["event"] == event) & (laps["season"] == season)]
    frames, skipped = [], 0
    for (driver, lap_number), _ in event_laps.groupby(["driver", "lap_number"]):
        try:
            telemetry = lap_telemetry(session, str(driver), int(lap_number))
            proxies = compute_proxies(telemetry, vision_config.proxies)
            table = minisector_table(proxies, vision_config.imaging.k_sectors)
        except (ValueError, IndexError, KeyError):
            skipped += 1
            continue
        table["sector"] = table.index
        table["season"], table["event"] = season, event
        table["driver"], table["lap_number"] = driver, int(lap_number)
        frames.append(table)

    if not frames:
        raise RuntimeError(f"No usable telemetry for {event}")
    sectors = pd.concat(frames, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    sectors.to_parquet(path, index=False)
    logger.info("%s: %d laps sectorized, %d skipped -> %s", event, len(frames), skipped, path)
    return sectors


@dataclass
class VisionDataset:
    """Tensor-ready samples; every array is aligned on the first axis."""

    frames: np.ndarray  # [n, L, K, 5, 5] float32, normalized
    pace_delta: np.ndarray  # [n] float32
    cliff: np.ndarray  # [n] float32
    meta: pd.DataFrame  # [n rows] season/event/driver/lap_number/stint_number


def build_samples(
    laps: pd.DataFrame,
    sectors: pd.DataFrame,
    scaler: ChannelScaler,
    vision_config: F1VisionConfig,
) -> VisionDataset:
    """Window labeled laps into [L, K, 5, 5] samples (see module docstring)."""
    imaging = vision_config.imaging
    labeled = add_labels(laps, vision_config.labels)
    context = labeled.set_index(["season", "event", "driver", "lap_number"])

    samples, pace, cliff, meta_rows = [], [], [], []
    for (season, event, driver, stint), stint_laps in labeled.groupby(
        ["season", "event", "driver", "stint_number"], sort=False
    ):
        event_sectors = sectors[
            (sectors["season"] == season)
            & (sectors["event"] == event)
            & (sectors["driver"] == driver)
        ]
        by_lap = {}
        for lap_n, table in event_sectors.groupby("lap_number"):
            if len(table) != imaging.k_sectors:
                continue
            row = context.loc[(season, event, driver, lap_n)]
            table = table.assign(
                tyre_age=row["tyre_age"],
                fuel_proxy=row["fuel_proxy"],
                track_temp=row["track_temp"],
                stint_lap=row["stint_lap"],
            )
            normalized = scaler.transform(table, str(event))
            ctx = {c: float(normalized[c].iloc[0]) for c in CONTEXT_CHANNELS}
            by_lap[int(lap_n)] = lap_frames(normalized.sort_values("sector"), ctx)

        stint_lap_numbers = sorted(stint_laps["lap_number"].astype(int))
        for anchor_pos in range(imaging.window_laps - 1, len(stint_lap_numbers)):
            window = stint_lap_numbers[anchor_pos - imaging.window_laps + 1 : anchor_pos + 1]
            anchor = window[-1]
            label_row = context.loc[(season, event, driver, anchor)]
            if pd.isna(label_row["pace_delta_next3"]) or pd.isna(label_row["cliff_within_5"]):
                continue
            if any(lap not in by_lap for lap in window):
                continue
            samples.append(np.stack([by_lap[lap] for lap in window]))
            pace.append(float(label_row["pace_delta_next3"]))
            cliff.append(float(label_row["cliff_within_5"]))
            current = float(label_row["pace_delta_current"])
            meta_rows.append(
                (season, event, driver, anchor, stint, 0.0 if pd.isna(current) else current)
            )

    if not samples:
        raise RuntimeError("No samples produced — check sector tables and labels")
    return VisionDataset(
        frames=np.stack(samples).astype(np.float32),
        pace_delta=np.array(pace, dtype=np.float32),
        cliff=np.array(cliff, dtype=np.float32),
        meta=pd.DataFrame(meta_rows, columns=META_COLUMNS),
    )
