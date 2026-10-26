# Phase 2 re-run — four seasons, walk-forward (2022–2025)

The §2.5 baseline study repeated on the full regulation era with the same
protocol as f1-core: **train on 2022–2024 (53 races), early-stop on the last 6 of
them, test on all 21 races of 2025.** No 2025 data reaches training. Sample
counts: train 28,932 / val 3,647 / test 11,842 (vs 12,829 in total for the
original 2024-only study). Regenerate with `python -m f1_vision.train --study`.

## Result: persistence still wins — with 3–4× the data

| model | MAE pace_delta_next3 (s) ↓ | PR-AUC cliff_within_5 ↑ |
|---|---|---|
| **persistence** | **0.327** | **0.287** |
| lstm_only | 0.526 | 0.164 |
| lgbm_flat | 0.557 | 0.189 |
| cnn_lstm | 0.564 | 0.188 |

- **Pace:** persistence has the lowest MAE at **all 21** test races. The closest
  any learned model gets is Miami (0.044 s behind).
- **Cliff warning:** persistence has the best PR-AUC overall and at 10/21 races.
  The learned models sit only slightly above the 2025 base rate of 12.3% (the
  PR-AUC of a random scorer), i.e. they carry little cliff signal.
- **The CNN-LSTM now ranks last on pace** — behind the LSTM-only model that sees
  the same information without the image structure. The image layout adds
  nothing measurable.

## Versus the 2024-only study (`artifacts/phase2/`)

| | 2024 random split | 2022–25 walk-forward |
|---|---|---|
| persistence MAE / PR-AUC | 0.342 / 0.335 | 0.327 / 0.287 |
| cnn_lstm MAE / PR-AUC | 0.582 / 0.317 | 0.564 / 0.188 |

More data and an honest time split did not close the gap; the learned models'
cliff PR-AUC fell. Together with the f1-core walk-forward finding (no tyre-cliff
widening on any compound with 3.5× data), this settles the Phase 2 question for
this data: **the negative result is a property of dry, clean-lap racing, not a
small-sample artifact.** Next-few-laps pace is dominated by its current value.

## Fixes that made this run possible

- ~1,000 laps (2022 Austria/Monaco, 2023 Canada, 2025 Miami/Belgium/Australia)
  have no FastF1 tyre age. They produced NaN frames and NaN losses; those laps
  are now skipped (not imputed) and guards fail loudly on non-finite inputs.
- The scaler's sector/lap join omitted `season`, which on multiple seasons
  cross-joined same-named races, including the 2025 test season.
- Assembly and sector building are now parallel per race (output verified
  identical to the serial path).

Artifact: `baseline_comparison.csv` (pooled + per-race MAE and PR-AUC).
