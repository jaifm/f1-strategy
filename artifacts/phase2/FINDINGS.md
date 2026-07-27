# Phase 2 — Findings (f1-vision research lab, 2024 season)

## §2.5 Baseline study — an honest negative result

Four models, identical event-held-out splits (14 train / 3 val / 4 test events;
8,669 / 1,660 / 2,500 samples; test cliff rate 16.6%). Pooled test metrics:

| model | MAE (pace_delta_next3, s) | PR-AUC (cliff_within_5) |
|---|---|---|
| **persistence** | **0.342** | **0.335** |
| lgbm_flat | 0.749 | 0.321 |
| lstm_only | 0.584 | 0.179 |
| cnn_lstm | 0.582 | 0.317 |

**Persistence — the current pace delta carried forward — beats every learned
model on both tasks**, and per event it has the lowest MAE at all four held-out
races (`baseline_comparison.csv`). The CNN-LSTM does not beat the flattened
LightGBM baseline meaningfully on pace MAE, and neither beats "assume tomorrow
looks like today."

Consistent with the Phase 1 finding (artifacts/phase1/FINDINGS.md): clean-lap
degradation in 2024 is gradual and strongly autocorrelated — there is little
3-lap-ahead signal beyond the current delta, and few cliff events survive the
clean-lap filters for a texture model to learn from.

Spec §2.7 rule applied: *"the fusion layer in Phase 3 uses the best model,
whatever it is"* → **Phase 3 fusion uses persistence for the pace path** (its
cliff scores are also the best available, base-rate-adjusted).

Caveats / follow-ups (documented, not excuses):
- Both nets early-stopped at epochs 8–9 (patience 8): val loss plateaued almost
  immediately. The val split's cliff rate (7.7%) is half the train rate —
  3 val events on one season make early stopping noisy. More seasons is the
  highest-leverage change before any architecture work.
- Sample count (12.8k) is small for DL, as spec §6 anticipated ("acceptance of
  a negative result as a valid outcome").
- Saudi Arabia is hard for everyone (PR-AUC ≤ 0.12): few cliffs, short race.

## §2.3 Proxy sanity (artifacts committed)

`proxy_sanity_VER_lap9.png`, `proxy_sanity_HAM_lap30.png`: all plausibility
gates PASS (peak g_lat 2–6 g, braking g, top speed, E_brake spikiness, κ bound)
after resampling merged FastF1 telemetry to a uniform 5 Hz grid — raw merged
streams (7 ms–1 s spacing) explode differentiation (g_lat 79 g before the fix).

## §2.4 Tab2Img (artifact committed)

`tab2img_frames_VER_lap9.png`: straights vs corner sectors produce visibly
distinct 5×5 textures; per-event train-split scaler persisted with checkpoints.

## Engineering note (macOS)

LightGBM (Homebrew libomp) and PyTorch (bundled libomp) deadlock when both
OpenMP runtimes load into one process — confirmed by stack sample (two
libomp.dylib copies parked in `__kmp_fork_barrier`). The study therefore runs
LightGBM via `f1_vision.train.lgbm_worker` in a separate process with .npz file
exchange. Do not import lightgbm and torch into the same process in this repo.
