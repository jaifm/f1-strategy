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
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
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


def sectorize_job(
    job: tuple[pd.DataFrame, int, str, IngestConfig, F1VisionConfig],
) -> str:
    """One race for a worker process; returns a status line, never raises."""
    laps, season, event, ingest_config, vision_config = job
    try:
        sectors = build_event_sectors(laps, season, event, ingest_config, vision_config)
    except Exception as exc:  # noqa: BLE001 — one bad event must not kill the batch
        return f"{season} {event}: FAILED ({exc})"
    n_laps = len(sectors[["driver", "lap_number"]].drop_duplicates())
    return f"{season} {event}: {n_laps} laps sectorized"


@dataclass
class VisionDataset:
    """Tensor-ready samples; every array is aligned on the first axis."""

    frames: np.ndarray  # [n, L, K, 5, 5] float32, normalized
    pace_delta: np.ndarray  # [n] float32
    cliff: np.ndarray  # [n] float32
    meta: pd.DataFrame  # [n rows] season/event/driver/lap_number/stint_number


def _race_samples(
    job: tuple[pd.DataFrame, pd.DataFrame, ChannelScaler, F1VisionConfig],
) -> tuple[list[np.ndarray], list[float], list[float], list[tuple[object, ...]]]:
    """Samples for ONE race. Module-level so worker processes can pickle it.

    Labels are per-race by construction (every label grouping includes
    season + event), so labelling race-by-race equals labelling all at once.
    """
    laps, sectors, scaler, vision_config = job
    imaging = vision_config.imaging
    labeled = add_labels(laps, vision_config.labels)
    context = labeled.set_index(["season", "event", "driver", "lap_number"])
    # Index sector rows by driver once — a per-stint boolean scan of the whole
    # table was the real cost of assembly (O(stints x rows)).
    sectors_by_driver = {str(d): g for d, g in sectors.groupby("driver")}

    samples, pace, cliff, meta_rows = [], [], [], []
    for (season, event, driver, stint), stint_laps in labeled.groupby(
        ["season", "event", "driver", "stint_number"], sort=False
    ):
        event_sectors = sectors_by_driver.get(str(driver))
        if event_sectors is None:
            continue
        by_lap = {}
        for lap_n, table in event_sectors.groupby("lap_number"):
            if len(table) != imaging.k_sectors:
                continue
            row = context.loc[(season, event, driver, lap_n)]
            # Laps missing a context value (FastF1 omits TyreLife on some laps:
            # 2022 Austria/Monaco, 2025 Miami...) get no frame rather than an
            # imputed one; windows that include them are dropped below.
            if row[CONTEXT_CHANNELS].isna().any():
                continue
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
    return samples, pace, cliff, meta_rows


def build_samples(
    laps: pd.DataFrame,
    sectors: pd.DataFrame,
    scaler: ChannelScaler,
    vision_config: F1VisionConfig,
    workers: int | None = None,
) -> VisionDataset:
    """Window labeled laps into [L, K, 5, 5] samples (see module docstring).

    Races are independent, so they are processed in parallel worker processes
    (``workers`` defaults to all cores; 1 = serial). Workers are *spawned* and
    this module imports neither torch nor lightgbm, so the dual-libomp
    deadlock cannot occur in them. Output order is the input race order.
    """
    keys = ["season", "event"]
    sector_groups = dict(list(sectors.groupby(keys)))
    jobs = [
        (race_laps, sector_groups[k], scaler, vision_config)
        for k, race_laps in laps.groupby(keys, sort=False)
        if k in sector_groups
    ]
    n_workers = min(workers or os.cpu_count() or 1, len(jobs))
    if n_workers <= 1:
        results = [_race_samples(job) for job in jobs]
    else:
        with ProcessPoolExecutor(n_workers, mp_context=get_context("spawn")) as pool:
            results = list(pool.map(_race_samples, jobs))

    samples = [x for r in results for x in r[0]]
    if not samples:
        raise RuntimeError("No samples produced — check sector tables and labels")
    frames = np.stack(samples).astype(np.float32)
    if not np.isfinite(frames).all():
        raise ValueError("Non-finite values in assembled frames — check context inputs")
    return VisionDataset(
        frames=frames,
        pace_delta=np.array([x for r in results for x in r[1]], dtype=np.float32),
        cliff=np.array([x for r in results for x in r[2]], dtype=np.float32),
        meta=pd.DataFrame([x for r in results for x in r[3]], columns=META_COLUMNS),
    )
