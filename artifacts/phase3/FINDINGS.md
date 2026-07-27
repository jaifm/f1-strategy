# Phase 3 — Findings (f1-pitwall integration layer)

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
