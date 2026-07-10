# F1 Probabilistic Strategy System — Full Technical Specification

**Audience:** Claude Code (implementation agent). This document is the single source of truth. Build in phase order. Do not skip acceptance criteria.

**One-line summary:** A monorepo containing (1) a probabilistic tire/strategy decision-support system built on quantile gradient boosting + Monte Carlo simulation, (2) a deep-learning research lab that converts telemetry into spatial "energy images" for a CNN-LSTM degradation detector, and (3) an integration layer that fuses both into a live pit-wall dashboard.

---

## 0. Repository Layout & Tooling

```
f1-strategy/
├── pyproject.toml              # uv-managed workspace; Python 3.11+
├── README.md
├── data/                       # gitignored; FastF1 cache + processed parquet
├── packages/
│   ├── f1-core/                # Project 1: probabilistic DSS
│   │   └── src/f1_core/
│   │       ├── ingest/         # FastF1 ETL, lap filtering
│   │       ├── features/       # feature engineering (shared with f1-vision)
│   │       ├── models/         # quantile LightGBM + conformal calibration
│   │       ├── simulate/       # Monte Carlo race engine
│   │       └── evaluate/       # PICP, backtests
│   ├── f1-vision/              # Project 2: CNN-LSTM research lab
│   │   └── src/f1_vision/
│   │       ├── proxies/        # physics-proxy generation (SavGol, E_lat, ...)
│   │       ├── imaging/        # Tab2Img 5x5 grid builder
│   │       ├── models/         # PyTorch CNN-LSTM, baselines
│   │       ├── train/          # training loops, splits, config
│   │       └── interpret/      # Grad-CAM saliency atlas (captum)
│   └── f1-pitwall/             # Project 3: integration layer
│       └── src/f1_pitwall/
│           ├── orchestrator/   # replay engine + model fusion
│           └── dashboard/      # Streamlit app
├── notebooks/                  # exploration only; no production logic
└── tests/                      # pytest, mirrors package structure
```

**Tooling requirements:**

- Package manager: `uv` with a workspace; each package independently installable.
- Lint/format: `ruff`. Type hints throughout; `mypy` on f1-core (strict optional elsewhere).
- Tests: `pytest` + `pytest-cov`. Every module in `f1-core` gets unit tests; models get golden-file regression tests on a small fixture dataset (one cached race session committed as parquet fixtures, NOT raw FastF1 cache).
- FastF1 caching is mandatory: `fastf1.Cache.enable_cache("data/fastf1_cache")` before any API call. Never hit the API in tests.
- All processed data stored as parquet with explicit schemas (pandera or dataclass-documented).
- Config via pydantic-settings / YAML, not hardcoded constants. Every magic number in this spec (window sizes, thresholds) must live in config.

---

## Phase 1 — `f1-core`: The Probabilistic Decision-Support System

### 1.1 Core paradigm

Traditional lap-time models minimize MSE and converge on the mean, which is blind to asymmetric risk — most importantly the non-linear "tire cliff" where pace suddenly collapses. `f1-core` instead performs uncertainty quantification: it predicts the full conditional distribution of lap time via quantile regression, then propagates that uncertainty through a Monte Carlo race simulator to produce strategy recommendations with calibrated confidence.

```
[FastF1 Telemetry] → [Lap Filter] → [Feature Builder] → [Quantile LightGBM + Conformal Calibration] → [Monte Carlo Engine] → [Strategy Ranking]
```

### 1.2 Data pipeline (`f1_core.ingest`)

Pull race-session lap data via FastF1 for a configurable set of seasons/events (default: 2022–2025, all dry races).

**Lap filtering — a lap is DROPPED if any of the following hold:**

1. **Traffic:** gap to car ahead < 1.5 s at any point on the lap (dirty air / racing). Use `Lap.get_car_data()` gap or position-interpolated gaps; if per-lap gap unavailable, approximate from timing data (`GapToLeader` deltas between consecutive positions).
2. **In-lap or out-lap** (`PitInTime` / `PitOutTime` not null).
3. **Track status not green:** SC, VSC, yellow, red (FastF1 `TrackStatus` != "1" for any part of the lap).
4. **First lap of the race** (start chaos).
5. **Deleted / inaccurate laps** (`IsAccurate == False`).
6. **Wet running:** compound is INTERMEDIATE or WET, or session marked wet. (Wet modeling is out of scope for v1.)

Log filter attrition per race (how many laps survived each rule) — this is a required pipeline output for sanity checking.

**Output schema (per lap, parquet):** `driver, team, event, season, lap_number, lap_time_s, compound, tyre_age, stint_number, stint_lap, track_temp, air_temp, fuel_proxy, position, is_clean`.

- `fuel_proxy`: fuel load is not public. Use `laps_remaining / total_laps` as a normalized proxy (linear burn assumption). Document this assumption in the module docstring.

### 1.3 The Global Physics Model (`f1_core.models`)

**Model:** LightGBM regressors trained with pinball (quantile) loss, one booster per quantile, sharing identical features and hyperparameters.

**Quantiles (semantics — get these right):** lower lap time = better performance.

- **q = 0.05 — best-case pace:** near-optimal lap under perfect conditions.
- **q = 0.50 — median expected pace.**
- **q = 0.95 — risk floor / worst plausible lap:** as the tire degrades this quantile expands away from the median much faster than q0.05 tightens. The widening (q95 − q50) spread vs. tyre age IS the tire-cliff signal.

**Features:** `tyre_age, compound (categorical), fuel_proxy, track_temp, air_temp, stint_lap, event (categorical), team (categorical), driver (categorical)`. Normalize lap time per event by subtracting the event's fastest clean q-lap benchmark if cross-track generalization proves poor; start global, evaluate, then decide.

**Physics constraints:**

- **Monotonic constraint** on `tyre_age`: predicted lap time is non-decreasing in tyre age (LightGBM `monotone_constraints`). Same for `fuel_proxy` in the appropriate direction (more fuel → slower).
- **Quantile-crossing guard:** after inference, sort the three quantile outputs per row so q05 ≤ q50 ≤ q95. Log crossing frequency; if > 2% of rows cross, revisit hyperparameters.

**Calibration — enforce, don't just measure:**

- Split: train / calibration / test by *event* (never split within a race — leakage).
- Apply **Conformalized Quantile Regression (CQR)**: use the calibration set to compute the conformity adjustment that widens/narrows the [q05, q95] interval so empirical coverage is exactly 90%.
- Deliverable: `evaluate/calibration.py` producing PICP plots (target 90% ± 2%), interval-width vs. tyre-age plots, and pinball-loss comparison against a naive MSE LightGBM baseline. **The MSE baseline is required** — it is the headline evidence for the whole paradigm.

### 1.4 Monte Carlo simulation engine (`f1_core.simulate`)

Vectorized NumPy engine. Input: a strategy definition (list of stints: compound + planned stop lap), race length, and the trained quantile model. Output: distribution of total race time over N = 1000+ simulations.

**Per-lap sampling:** sample from an asymmetric triangular distribution with (min = q05, mode = q50, max = q95) predicted for that lap's state.

**Critical correction — correlated noise.** Fully independent per-lap draws underestimate total-race variance (errors average out over 50+ laps). Decompose noise per simulation:

- `driver_day_factor`: one draw per simulation (e.g., N(0, σ_day)), added to every lap — captures "on it today vs. not".
- `stint_factor`: one draw per stint — captures tire-set-to-tire-set variation.
- `lap_noise`: the per-lap triangular draw, scaled so total marginal variance still matches the calibrated quantile spread. Calibrate the decomposition weights against historical stint-time variance (a validation script must verify simulated total-race-time spread ≈ historical spread for held-out races).

**Race-event modeling (config-toggleable):**

- **Pit loss:** per-track pit-lane time loss as a normal distribution (mean/σ from config table; seed with published values, refine from data).
- **Safety car:** Poisson arrival per lap with per-track rate; SC laps run at a fixed pace multiplier and offer a reduced pit loss (cheap stop). v1 can ship without SC but the interface must accept it.
- **Traffic on rejoin:** given the field's simulated positions, a rejoin into traffic adds a per-lap penalty until clear. v1 approximation: static lookup of expected traffic cost by rejoin position.

**Strategy comparison API:**

```python
result = simulate_race(strategies=[strat_a, strat_b], event="Bahrain 2024", n_sims=2000)
result.win_probability(strat_a)          # P(A beats B)
result.total_time_distribution(strat_a)  # array for violin plots
result.percentile_report()               # p5/p50/p95 race time per strategy
```

### 1.5 Dashboard v1 (`f1-pitwall`, thin slice)

A Streamlit "Pit Wall" page: pick event + strategies, see violin plots of total race time per strategy, win-probability matrix, and a slider to shift pit windows with live re-simulation (cache the model; re-run MC only).

### 1.6 Phase 1 acceptance criteria

- [ ] ETL runs end-to-end for one full season with cache; filter-attrition report generated.
- [ ] Quantile model beats MSE baseline on pinball loss for q05/q95 on held-out events.
- [ ] Post-CQR coverage on held-out events within 90% ± 2%.
- [ ] (q95 − q50) demonstrably widens with tyre age (plot artifact committed).
- [ ] MC engine: 2000 sims × 57 laps in < 2 s (vectorized, no Python loop over sims).
- [ ] Simulated race-time spread validated against historical spread (correlated-noise check).
- [ ] Streamlit dashboard runs; strategy comparison interactive.
- [ ] Test coverage ≥ 80% on `f1-core`; golden-file tests on fixture race.

---

## Phase 2 — `f1-vision`: The CNN-LSTM Research Lab

### 2.1 Core paradigm

`f1-core` treats telemetry channels as independent tabular columns; it cannot see the *interaction structure* of forces (e.g., simultaneous braking + cornering saturating the friction circle). `f1-vision` uses representation learning to build an unmeasured "thermal camera": raw ~4 Hz GPS/telemetry is converted into spatially organized energy images, and a CNN-LSTM learns the physical textures of degradation.

```
[4Hz Telemetry] → [SavGol Filter] → [Physics Proxies] → [Tab2Img 5×5 frames] → [CNN encoder → LSTM] → [Degradation prediction]
```

### 2.2 Prediction target (defined explicitly — this was the gap in the original design)

**Primary target (regression):** `pace_delta_next3` — the driver's mean clean-air lap-time delta (vs. their own fresh-tire stint baseline) over the *next 3 laps*, in seconds. This makes the model predictive (what's coming), not descriptive (what happened).

**Auxiliary target (classification head):** `cliff_within_5` — binary, 1 if within the next 5 laps the pace delta exceeds a cliff threshold (config default: +1.5 s/lap vs. stint baseline) or the driver pits earlier than the median stop lap for that compound/event. Multi-task loss: `L = L_huber(pace) + λ · L_bce(cliff)`, λ in config (default 0.5).

Labels are computed from the same filtered lap data as Phase 1 — reuse `f1_core.ingest`.

### 2.3 Physics-proxy generation (`f1_vision.proxies`)

From per-lap car telemetry (Speed, Throttle, Brake, RPM, Gear, X/Y GPS at ~4 Hz):

- **Savitzky-Golay filtering:** smooth X, Y, Speed with an 11-point window (~1.5 s at 4–7 Hz), polyorder 2, before differentiating. Never differentiate raw GPS.
- **Path curvature κ:** from smoothed x(t), y(t): `κ = |x'y'' − y'x''| / (x'² + y'²)^1.5`.
- **Lateral energy proxy:** `E_lat ∝ v³ · κ` (lateral force ∝ v²κ, work rate × v).
- **Brake thermal proxy:** `E_brake = max(0, −d/dt(½v²))` (mass constant, dropped).
- **Traction work proxy:** `E_traction = max(0, a_long) · v`.
- **G proxies:** `g_lat = v²κ / 9.81`, `g_long = a_long / 9.81`.

**Sanity harness (required):** a validation notebook/script that plots proxies over a known lap and checks physical plausibility (peak g_lat 3–5 g in fast corners, E_brake spikes at known braking zones). Commit the plots as artifacts.

### 2.4 Tab2Img transformation (`f1_vision.imaging`)

**Temporal structure (defined explicitly):** each lap is divided into **K = 20 equal-distance mini-sectors**. Each mini-sector is aggregated (mean for continuous channels, max for energy spikes) and rendered as one **5×5 grid** → a lap is a sequence of 20 frames. A training sample is the last **L = 5 laps** of frames (shape: `[L·K, 1, 5, 5]` or `[L, K, 5, 5]` — pick one and document it).

**Fixed handcrafted layout** (keeps Grad-CAM interpretable; do not learn the layout in v1):

| Quadrant | Cells | Features | Latent meaning |
|---|---|---|---|
| I — Energy (top-left) | 4 | E_lat, E_brake, E_traction, κ | Friction work on the contact patch |
| II — Dynamics (top-right) | 4 | g_lat, g_long, Speed, RPM | Physical result of energy input |
| III — Driver (bottom-left) | 4 | Throttle, Brake, Steering*, Gear | Input signature: aggressive vs. smooth |
| IV — Context (bottom-right) | 4 | TyreAge, fuel_proxy, TrackTemp, StintLap | Degradation baseline |
| Center row/col | 9 | Bilinear blends of adjacent quadrants | Smooth spatial transitions so 3×3 kernels see cross-quadrant interaction |

*Steering is not in public FastF1 data — substitute smoothed dκ/dt (steering rate proxy) and note it.

Normalize each channel per-event to [0, 1] using train-split statistics only (persist the scaler).

### 2.5 Model (`f1_vision.models`, PyTorch)

- **CNN encoder:** 2 conv blocks (Conv2d 3×3 → BatchNorm → ReLU), 16 → 32 channels, no pooling (grid is only 5×5), flatten → linear → 64-d embedding per frame.
- **Temporal:** LSTM (hidden 128, 1–2 layers) over the frame sequence; final hidden state → two heads (regression + classification).
- **Training:** AdamW, cosine schedule, early stopping on val pinball/Huber loss. Seeded, deterministic where possible. Config-driven (hydra or plain pydantic YAML).
- **Splits:** by event (entire races held out). Never split within a race.

**Required baselines (the research is worthless without them):**

1. Persistence: current pace delta carried forward.
2. LightGBM on flattened mini-sector aggregate features (tests whether the *image structure* adds anything over the same information as a table).
3. LSTM-only on per-lap aggregate vectors (no CNN).
4. Full CNN-LSTM.

Report MAE / Huber on pace_delta and PR-AUC on cliff_within_5 for all four, per held-out event.

### 2.6 Interpretability (`f1_vision.interpret`)

Grad-CAM (via `captum`) on the CNN encoder, averaged over frames and events, rendered as heatmaps over the 5×5 grid. **Deliverable — the Saliency Atlas:** a generated HTML/markdown report showing attention maps for (a) fresh tires, (b) mid-stint, (c) pre-cliff laps. Hypothesis to verify: attention shifts toward Quadrant I (Energy) as degradation approaches.

### 2.7 Phase 2 acceptance criteria

- [ ] Proxy sanity harness passes physical-plausibility checks; artifacts committed.
- [ ] Dataset builder produces tensors + labels for ≥ 1 full season; reuses Phase 1 filtering.
- [ ] All four baselines trained and compared on identical event-held-out splits.
- [ ] CNN-LSTM beats the flattened-LightGBM baseline on pace_delta MAE (if it doesn't, that finding is documented honestly and the fusion layer in Phase 3 uses the best model, whatever it is).
- [ ] Saliency Atlas generated.
- [ ] Model checkpoints + scalers versioned and loadable via one factory function.

---

## Phase 3 — `f1-pitwall`: The Integration Layer

### 3.1 Architecture

```
                    ┌─→ [f1-core: Quantile GBM + MC] ─→ Strategy Anchor (the map)
[Replay Stream] ────┤
                    └─→ [f1-vision: CNN-LSTM]        ─→ Live Risk Modifier (the eyes)
                                    │
                                    ▼
                    [Fusion: recalibrated live strategy]
```

- **Replay engine (`orchestrator/replay.py`):** replays a historical race from the FastF1 cache as a timed event stream (configurable speed multiplier: 1×, 10×, max). This is the only "live" source in v1; the interface is a generator of typed events (`LapCompleted`, `TelemetryFrame`) so a real live API can be swapped in later.
- **Orchestrator (`orchestrator/service.py`):** async loop feeding both models. f1-core provides the strategy posterior computed pre-race; f1-vision consumes the telemetry stream per lap.

### 3.2 Fusion rule (defined explicitly)

f1-vision outputs per driver per lap: `predicted_pace_delta_next3` and `p_cliff_within_5`. Fusion modifies the MC engine's inputs — not the GBM itself:

1. **Median shift:** add `predicted_pace_delta_next3` to the q50 path for the remaining laps of the current stint (decaying influence, half-life 3 laps).
2. **Spread inflation:** widen (q95 − q50) for the current stint by factor `1 + α · p_cliff_within_5` (α config, default 1.0).
3. Re-run a fast MC (500 sims) each lap; the strategy ranking and optimal-stop-lap window update live.

All three fusion parameters (half-life, α, sims) live in config, and the dashboard displays raw model outputs alongside fused results so miscalibration is visible.

### 3.3 Live dashboard

Streamlit (upgrade path: FastAPI + WebSocket + React if Streamlit rerun latency is unacceptable — measure first). Panels: race replay state (lap, positions, gaps); per-driver current 5×5 energy frame with live Grad-CAM overlay; strategy window chart — recommended stop-lap band (p25–p75 of optimal stop) shifting lap by lap; alert feed — "cliff risk P>0.6 for VER — window pulled forward 3 laps".

### 3.4 Phase 3 acceptance criteria

- [ ] Replay of a full historical race runs end-to-end at 10× speed without drift or crashes.
- [ ] Fusion demonstrably moves strategy windows on a race with a known late-stint collapse (choose a validation race and document the case study).
- [ ] End-to-end latency per lap update < 1 s at 1× replay speed.
- [ ] One-command launch: `uv run pitwall --race "Bahrain 2024" --speed 10`.

---

## 4. Cross-Cutting Requirements

- **No lookahead leakage anywhere:** any feature or label computed for lap N must use only data available at lap N (except training labels, which are clearly future-dated by design and only used offline).
- **Determinism:** all randomness seeded via config; MC engine accepts an RNG.
- **Honest reporting:** if a fancy model loses to a baseline, the spec's goal is a correct system, not a flattering one. Keep the baseline comparisons in the final artifacts.
- **Docs:** each package ships a README with a quickstart; the physics-proxy module documents every formula and its assumptions.

## 5. Suggested Build Order (for Claude Code)

1. Phase 0: scaffold repo, tooling, CI (pytest + ruff), fixture data.
2. Phase 1.2 → 1.3 → 1.4 → 1.5 (each with tests before moving on).
3. Phase 2.3 → 2.4 → 2.2 labels → 2.5 baselines-first → 2.6.
4. Phase 3.
5. After each phase: run the acceptance checklist and produce the listed artifacts.

## 6. Known Risks & Mitigations

- **GPS quality varies by track/season** → proxy sanity harness gates every event before it enters training data.
- **Public data lacks fuel/tire temps** → proxies documented as proxies; never claim measured values.
- **Small effective dataset for DL** (dry clean laps only) → aggressive baselines, event-level CV, and acceptance of a negative result as a valid outcome.
- **FastF1 API changes** → pin the version; isolate all FastF1 calls in `f1_core.ingest`.
