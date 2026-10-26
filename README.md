# F1 Probabilistic Strategy System

A monorepo containing three linked projects (see [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full specification):

1. **`f1-core`** — probabilistic tire/strategy decision-support: quantile LightGBM with conformal calibration, propagated through a vectorized Monte Carlo race simulator.
2. **`f1-vision`** — deep-learning research lab: telemetry → physics proxies → 5×5 "energy image" frames → CNN-LSTM degradation detector with Grad-CAM interpretability.
3. **`f1-pitwall`** — integration layer: race replay orchestrator fusing both models into a live Streamlit pit-wall dashboard.

## Findings at a glance

Data: the full ground-effect regulation era, **2022–2025** — 101,290 laps, 44,098 clean
laps across 92 races. Every result below is reported honestly, including the
negative ones; the detail lives in each phase's `FINDINGS.md`.

| Question | Answer | Evidence |
|---|---|---|
| Does a quantile model beat a standard MSE model on risk? | **Yes, decisively.** Tail pinball loss: q05 0.152 vs 0.696, q95 0.145 vs 0.248 | [walk-forward](artifacts/phase1_walkforward/FINDINGS.md) |
| Are the uncertainty intervals calibrated on unseen future races? | **Nearly.** 87.4% coverage vs a 90% target, rising from 84% to 92% as the 2025 season accumulates | [walk-forward](artifacts/phase1_walkforward/FINDINGS.md) |
| Does the risk spread blow up with tyre age (the "tyre cliff")? | **No** — flat or slightly shrinking on every compound, and still absent with 3.5× the data | [walk-forward](artifacts/phase1_walkforward/FINDINGS.md) |
| Does the Monte Carlo engine reproduce real race-time spread? | **Yes** — spread ratio ≈ 1 on held-out races; 2000 sims × 57 laps well under 2 s | [phase 1](artifacts/phase1/FINDINGS.md) |
| Does a CNN-LSTM on telemetry "energy images" beat simple baselines? | **No — even with 3–4× the data.** Carrying the current pace forward (persistence) wins on both tasks and has the best pace error at all 21 races of 2025 | [phase 2 walk-forward](artifacts/phase2_walkforward/FINDINGS.md) |
| What does the CNN actually look at? | Longitudinal dynamics and context — *not* the energy channels the design hypothesised | [saliency atlas](artifacts/phase2/saliency_atlas.md) |
| Does live fusion improve strategy calls? | **Yes, in the case study:** on Pérez's Mexico 2024 collapse, fusion recommended boxing on lap 42 — the lap the team actually pitted — while the static model said 47 | [phase 3](artifacts/phase3/FINDINGS.md) |

**The through-line:** in dry, green-flag, clean-air running, tyre degradation is gradual
and strongly autocorrelated. Quantile modelling of that is valuable; hunting for a
sudden cliff in clean-lap data is not, because the filters (and the teams' own pit
calls) remove most of it. Both halves are tested on 2025 using only earlier
races: the strategy model race by race (expanding window), the telemetry lab with
a single 2022–2024 → 2025 cut-off.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12 is provisioned automatically). On macOS, LightGBM also needs OpenMP: `brew install libomp`.

> **Important (macOS/iCloud):** this repo lives in an iCloud-synced folder, and a
> project-local `.venv` breaks intermittently there (see Troubleshooting). Keep the
> venv outside the synced tree — add this to your `~/.zshrc`:
>
> ```bash
> export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/f1-strategy"
> ```

```bash
uv sync                 # install workspace + dev tools
uv run pytest tests/f1_core tests/f1_pitwall   # core tests
uv run pytest tests/f1_vision                  # vision tests (separate: see below)
uv run ruff check .     # lint
uv run ruff format .    # format
uv run mypy             # type-check f1-core (strict)

# pipeline
uv run python -m f1_core.ingest     # ETL (seasons in configs/f1_core.yaml)
uv run python -m f1_core.evaluate   # walk-forward study -> artifacts/phase1_walkforward
uv run python -m f1_core.simulate   # MC spread validation vs a held-out race

# dashboards
uv run streamlit run packages/f1-pitwall/src/f1_pitwall/dashboard/pitwall.py  # v1 strategy compare
uv run pitwall --race "Bahrain 2024" --speed 10             # live replay pit wall (Phase 3)
uv run pitwall --race "Bahrain 2024" --speed 0 --headless   # acceptance stats, no UI
```

### Troubleshooting: `ModuleNotFoundError` for workspace packages (macOS)

uv marks the `.venv` directory hidden for Finder; when the repo lives in an
iCloud-synced folder (e.g. `~/Documents`), the sync agent asynchronously propagates
that flag to files inside — including the editable-install `.pth` files, which
Python ≥ 3.12 then **silently skips**, breaking workspace imports. Every `uv sync`
re-marks the directory, so the breakage recurs.

Permanent fix: relocate the venv outside the synced tree with
`UV_PROJECT_ENVIRONMENT` (see Quickstart). Stopgap if you must use a local `.venv`:
`chflags -R nohidden .venv` after every sync.

## Repository layout

```
packages/
├── f1-core/      # ingest, features, models (quantile GBM + CQR), simulate (MC), evaluate
├── f1-vision/    # proxies, imaging (Tab2Img), models (CNN-LSTM), train, interpret
└── f1-pitwall/   # orchestrator (replay + fusion), dashboard (Streamlit)
tests/            # pytest, mirrors package structure; fixture data committed here
notebooks/        # exploration only; no production logic
data/             # gitignored: FastF1 cache + processed parquet
```

## Conventions

- All FastF1 API calls go through `f1_core.ingest` with the local cache enabled; tests never hit the API (golden-file parquet fixtures only).
- Config via pydantic-settings / YAML — no hardcoded magic numbers.
- Event-level train/calibration/test splits everywhere; never split within a race.
- Honest reporting: baseline comparisons ship in the final artifacts even when the fancy model loses.
- **Never import LightGBM and PyTorch into the same process** (their two OpenMP
  runtimes deadlock on macOS — see `artifacts/phase2/FINDINGS.md`). That is why the
  test suite runs as two invocations and the vision study calls LightGBM via
  `f1_vision.train.lgbm_worker`.
