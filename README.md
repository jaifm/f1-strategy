# F1 Probabilistic Strategy System

A monorepo containing three linked projects (see [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full specification):

1. **`f1-core`** — probabilistic tire/strategy decision-support: quantile LightGBM with conformal calibration, propagated through a vectorized Monte Carlo race simulator.
2. **`f1-vision`** — deep-learning research lab: telemetry → physics proxies → 5×5 "energy image" frames → CNN-LSTM degradation detector with Grad-CAM interpretability.
3. **`f1-pitwall`** — integration layer: race replay orchestrator fusing both models into a live Streamlit pit-wall dashboard.

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
uv run pytest           # run tests
uv run ruff check .     # lint
uv run ruff format .    # format
uv run mypy             # type-check f1-core (strict)

# pipeline
uv run python -m f1_core.ingest     # ETL (seasons in configs/f1_core.yaml)
uv run python -m f1_core.evaluate   # calibration study -> artifacts/phase1
uv run python -m f1_core.simulate   # MC spread validation vs a held-out race

# dashboard
uv run streamlit run packages/f1-pitwall/src/f1_pitwall/dashboard/pitwall.py
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
