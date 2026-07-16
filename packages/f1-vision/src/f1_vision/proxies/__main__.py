"""Proxy sanity harness (spec §2.3, required): plot proxies over a known lap and
check physical plausibility. Artifacts go to artifacts/phase2/.

    uv run python -m f1_vision.proxies --driver VER --lap 9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from f1_core.config import load_config as load_core_config  # noqa: E402
from f1_core.ingest.telemetry import lap_telemetry, load_race_telemetry_session  # noqa: E402
from f1_vision.config import load_config  # noqa: E402
from f1_vision.proxies.physics import compute_proxies  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--event", default="Bahrain Grand Prix")
    parser.add_argument("--driver", default="VER")
    parser.add_argument("--lap", type=int, default=9)
    parser.add_argument("--out", default="artifacts/phase2")
    args = parser.parse_args(argv)

    core = load_core_config("configs/f1_core.yaml")
    session = load_race_telemetry_session(core.ingest, args.season, args.event)
    telemetry = lap_telemetry(session, args.driver, args.lap)
    proxies = compute_proxies(telemetry, load_config().proxies)

    # Physical plausibility gates (spec: peak g_lat 3-5 g in fast corners,
    # E_brake spikes at braking zones). Bounds are generous on purpose: the
    # gate must catch unit errors (10x) not setup differences (1.2x).
    checks = {
        "peak g_lat in 2-6 g": 2.0 < proxies["g_lat"].max() < 6.0,
        "peak braking g in 2-7 g": 2.0 < (-proxies["g_long"].min()) < 7.0,
        "top speed 70-110 m/s": 70.0 < proxies["speed_ms"].max() < 110.0,
        "E_brake spiky (p99 > 10x median)": (
            proxies["e_brake"].quantile(0.99) > 10 * max(proxies["e_brake"].median(), 1e-9)
        ),
        "kappa sane (max < 0.1 1/m)": proxies["kappa"].max() < 0.1,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    channels = ["speed_ms", "g_lat", "g_long", "e_lat", "e_brake", "e_traction"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    for ax, channel in zip(axes.flat, channels, strict=True):
        ax.plot(proxies["time_s"] - proxies["time_s"].iloc[0], proxies[channel], linewidth=0.9)
        ax.set_title(channel)
    fig.suptitle(f"{args.event} {args.season} — {args.driver} lap {args.lap}")
    fig.supxlabel("lap time (s)")
    fig.tight_layout()
    path = out / f"proxy_sanity_{args.driver}_lap{args.lap}.png"
    fig.savefig(path, dpi=150)

    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"plot -> {path}")
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
