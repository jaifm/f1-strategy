"""Fetch race telemetry for every race with clean laps, resuming across runs.

FastF1 caps ~500 API calls/hour. Rate-limit failures are retried after a wait;
ANY OTHER failure is reported and the race skipped — a blanket retry loop once
masked a real schema bug for 12 attempts, so error kinds stay distinguishable.

    uv run python scripts/fetch_telemetry.py [--seasons 2022 2023 2025]
"""

from __future__ import annotations

import argparse
import time

import fastf1
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

WAIT_S = 1200


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023, 2025])
    parser.add_argument("--max-waits", type=int, default=24)
    args = parser.parse_args()

    fastf1.Cache.enable_cache("data/fastf1_cache")
    laps = pd.read_parquet("data/processed/laps.parquet")
    wanted = set(
        laps[laps["is_clean"]][["season", "event"]].drop_duplicates().itertuples(index=False, name=None)
    )

    todo = []
    for season in args.seasons:
        schedule = fastf1.get_event_schedule(season, include_testing=False)
        for _, row in schedule.iterrows():
            if (season, row["EventName"]) in wanted:
                todo.append((season, int(row["RoundNumber"]), str(row["EventName"])))

    print(f"{len(todo)} races to fetch", flush=True)
    waits, failures = 0, []
    while todo and waits <= args.max_waits:
        season, rnd, name = todo[0]
        try:
            session = fastf1.get_session(season, rnd, "R")
            session.load(laps=True, telemetry=True, weather=False, messages=False)
            print(f"done {season} {name}  ({len(todo) - 1} left)", flush=True)
            todo.pop(0)
        except RateLimitExceededError:
            waits += 1
            print(f"rate limited with {len(todo)} left; wait {waits}/{args.max_waits}", flush=True)
            time.sleep(WAIT_S)
        except Exception as exc:  # noqa: BLE001 — report, skip, keep going
            print(f"FAILED {season} {name}: {type(exc).__name__}: {exc}", flush=True)
            failures.append((season, name))
            todo.pop(0)

    print(f"remaining: {len(todo)}  failed: {len(failures)}", flush=True)
    for season, name in failures:
        print(f"  failed: {season} {name}", flush=True)
    raise SystemExit(0 if not todo else 1)


if __name__ == "__main__":
    main()
