"""Command-line entry point for the pit-wall replay dashboard.

Target interface (Phase 3): ``uv run pitwall --race "Bahrain 2024" --speed 10``.
"""

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pitwall", description="F1 pit-wall replay dashboard.")
    parser.add_argument("--race", help='Race to replay, e.g. "Bahrain 2024".')
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier.")
    args = parser.parse_args(argv)
    raise NotImplementedError(
        f"Replay engine not yet implemented (race={args.race!r}, speed={args.speed}). "
        "Built in Phase 3 — see PROJECT_SPEC.md."
    )


if __name__ == "__main__":
    main()
