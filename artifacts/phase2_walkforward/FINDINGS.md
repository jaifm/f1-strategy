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

## Saliency atlas on 2025 (`saliency_atlas.md` / `.png`)

Grad-CAM on the four-season CNN, all 21 test races of 2025 (quadrant share of
attention; the remainder falls on the centre blend cells):

| group | n | I energy | II dynamics | III driver | IV context | centre cross |
|---|---|---|---|---|---|---|
| fresh (age ≤ 5) | 386 | 16.0% | 15.8% | 12.5% | 16.5% | 39% |
| mid-stint (10–20) | 5,455 | 17.1% | 15.8% | 14.0% | 14.6% | 38% |
| pre-cliff | 1,451 | 16.8% | 14.9% | 13.0% | 15.6% | 40% |

- **Hypothesis not supported, again.** Energy attention is flat across tyre
  states (16.0 → 17.1 → 16.8%); pre-cliff sits 0.3 points *below* mid-stint.
- **The model's attention barely changes with degradation at all** — the three
  maps are near-identical. That matches its cliff PR-AUC sitting close to the base
  rate: it is not reading a degradation signal, so it has nothing to shift toward.
- **What changed from the 2024 model:** attention is spread evenly over the four
  quadrants (13–17% each) instead of concentrating on dynamics and context
  (29–35%), and about 40% now lands on the centre cross — the blended cells that
  mix neighbouring quadrants — versus 19–27% before. Caveat: those cells are
  averages of quadrant cells, so attribution there is to derived values, not new
  information.
- **Generator fix:** the atlas verdict was a bare "pre-cliff > fresh" check, which
  first labelled this run's 0.8-point wobble "SUPPORTED". It now requires
  pre-cliff to exceed *both* earlier groups by ≥ 2 points (tested both ways). The
  2024 verdict (12.2% → 5.5%) is unchanged under the new rule.

## Fixes that made this run possible

- ~1,000 laps (2022 Austria/Monaco, 2023 Canada, 2025 Miami/Belgium/Australia)
  have no FastF1 tyre age. They produced NaN frames and NaN losses; those laps
  are now skipped (not imputed) and guards fail loudly on non-finite inputs.
- The scaler's sector/lap join omitted `season`, which on multiple seasons
  cross-joined same-named races, including the 2025 test season.
- Assembly and sector building are now parallel per race (output verified
  identical to the serial path).

Artifact: `baseline_comparison.csv` (pooled + per-race MAE and PR-AUC).
