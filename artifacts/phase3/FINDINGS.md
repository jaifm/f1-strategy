# Phase 3 — Findings (f1-pitwall integration layer)

> **Correction (2026-10-26, four-season re-run).** The original case study below
> overstated its result. (1) Its pre-race model was trained on *every* race,
> including Mexico 2024 itself — a lookahead leak. (2) Pérez stopped at the end of
> lap **43**, not 42 (42 was his last *clean* lap; the in-lap is filtered out).
> (3) The fused recommendation was "box next lap" on every lap of the collapse, so
> "recommended lap 42 at lap 41" was the earliest-allowed floor, not a pick of a
> specific lap. See the re-run section at the end; the original text is kept
> unedited below for the record.

## §3.4 acceptance checklist

- [x] Replay of a full historical race end-to-end at 10× without drift or crashes
      (Bahrain 2024, headless: 57 laps over 560 s wall clock, pacing drift **+0.01 s**)
- [x] Fusion demonstrably moves strategy windows on a race with a known
      late-stint collapse — case study below
- [x] End-to-end latency per lap update: **mean 35 ms, p95 45 ms** (gate: < 1 s)
- [x] One-command launch: `uv run pitwall --race "Bahrain 2024" --speed 10`
      (add `--headless` for the no-UI acceptance run)

## Case study: Pérez, Mexico City 2024 (`case_study_PER_mexico.png`)

Stint 2 collapses from lap ~25 (clean-lap delta climbs to **+3.7 s** at lap 40);
PER pitted on lap 42. Head-to-head, fused vs anchor-only (same model, seed, and
candidates; `case_fused.csv` / `case_anchor.csv`):

- Through the healthy stint both agree (shift = 0), as they should.
- At the collapse the fused window pulls **up to 5 laps forward**: on lap 41
  the anchor recommends stopping at lap 47 while fusion says lap 42 —
  **matching the lap the team actually boxed**.
- Raw persistence outputs (delta, p_cliff) are plotted alongside per spec §3.2
  so miscalibration stays visible; p_cliff saturates during the collapse.

## Design decisions

- **Fusion inputs come from persistence** (current clean-lap delta vs stint
  baseline; p_cliff = clip(delta / 1.5 s)) — the §2.5 study's best model, per
  the spec rule "the fusion layer uses the best model, whatever it is."
- **Common random numbers across candidate stop laps**: the day factor cancels
  in the argmin, so the optimal-stop distribution reflects strategy, not luck.
- Replay session time is reconstructed from cumulative lap times (the processed
  table drops absolute session time); documented approximation in replay.py.

## Documented deviations

- **No live Grad-CAM overlay** on the frame panel: it would import torch into a
  process already holding LightGBM (dual-libomp deadlock,
  artifacts/phase2/FINDINGS.md), and §2.6 showed the CNN is weak anyway. The
  5×5 energy frame renders without the overlay; the static atlas serves the
  interpretability need.
- Safety car & rejoin traffic remain v2 (interfaces exist; SC config raises
  NotImplementedError by design).
- Quantile crossing on the full-data anchor fit runs 3.5–5.3% (above the 2%
  guideline that was met in the LOO evaluation); logged by the engine and
  accepted for v1 — the crossing guard sorts affected rows.


## Re-run on 2022–2025 with no lookahead (`case_study_PER_mexico_walkforward.png`)

The pit wall now trains its pre-race model only on races **before** the one it
replays (`prior_races`; `--allow-future` reproduces the old leaky behaviour).
Pérez, Mexico City 2024, stint 2 = laps 21–43, stop at the end of lap 43.

| | live adjustment | pre-race model only |
|---|---|---|
| laps of stint 2 (26–41) recommending "box next lap" | **16 / 16** | 1 / 16 (lap 33) |
| recommendation at lap 41 | stop after lap 42 | stop after lap 45 |
| largest pull-forward vs pre-race model | 6 laps | — |

What holds up:
- **Live adjustment reacts to the collapse and the pre-race model does not.**
  From lap 26 — 17 laps before the team stopped — the fused recommendation is
  "box now" every lap, while the pre-race model keeps pushing the stop 2–5 laps
  out until the stop.
- **Latency unchanged:** mean 38 ms per lap update (gate 1 s).

What does not:
- It never singled out a stop lap; it flagged "stop as soon as possible" for 16
  laps. Whether stopping around lap 26 would have beaten the team's lap 43 is not
  something this replay can say — there is no counterfactual race to compare.

The leak mattered: trained on all races (incl. this one), the pre-race model
alone already reached "box now" on 8/16 stint-2 laps, which made live adjustment
look unnecessary. The honest setup shows the pre-race model alone stays too
optimistic and live adjustment carries the signal.

CSVs: `case_wf_fused.csv`, `case_wf_anchor.csv`, `case_leak_fused.csv`,
`case_leak_anchor.csv` (the original 2024-only runs remain as `case_fused.csv` /
`case_anchor.csv`).
