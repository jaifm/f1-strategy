"""Baseline study (spec §2.5): four models on identical event-held-out splits.

1. Persistence — current pace delta carried forward.
2. LightGBM on flattened mini-sector features (does the *image structure* add
   anything over the same information as a table?).
3. LSTM-only on per-lap aggregate vectors (no CNN).
4. CNN-LSTM (spec §2.5 architecture).

All four share the exact split, scaler, and samples. Reported per held-out
event and pooled: MAE on ``pace_delta_next3``, PR-AUC on ``cliff_within_5``.
Artifacts: ``artifacts/phase2/baseline_comparison.csv`` (+ printed table),
checkpoints + scaler under ``data/processed/vision/checkpoints/`` loadable via
``load_checkpoint`` (one factory, spec §2.7).
"""

from __future__ import annotations

import copy
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

from f1_core.config import F1CoreConfig
from f1_vision.config import F1VisionConfig
from f1_vision.imaging.tab2img import GRID_CELLS, ChannelScaler
from f1_vision.models.nets import CNNLSTM, LSTMOnly, multitask_loss
from f1_vision.train.dataset import VisionDataset, build_samples, sectors_path

logger = logging.getLogger(__name__)

QUAD_ROWS = [i for i, _ in GRID_CELLS]
QUAD_COLS = [j for _, j in GRID_CELLS]


def assemble_datasets(
    core: F1CoreConfig, vision: F1VisionConfig
) -> tuple[dict[str, VisionDataset], ChannelScaler]:
    """Event-level train/val/test datasets sharing one train-fit scaler."""
    cfg = vision.train
    laps = pd.read_parquet(core.ingest.output_dir / "laps.parquet")
    events = sorted(
        e
        for e in laps.loc[laps["is_clean"], "event"].unique()
        if sectors_path(core.ingest.output_dir, e).exists()
    )
    order = np.random.default_rng(cfg.split_seed).permutation(len(events))
    n_test = max(1, round(len(events) * cfg.test_fraction))
    n_val = max(1, round(len(events) * cfg.val_fraction))
    split_events = {
        "test": [events[i] for i in order[:n_test]],
        "val": [events[i] for i in order[n_test : n_test + n_val]],
        "train": [events[i] for i in order[n_test + n_val :]],
    }
    logger.info("split events: %s", {k: len(v) for k, v in split_events.items()})

    def load_sectors(event_list: list[str]) -> pd.DataFrame:
        frames = [pd.read_parquet(sectors_path(core.ingest.output_dir, e)) for e in event_list]
        return pd.concat(frames, ignore_index=True)

    train_sectors = load_sectors(split_events["train"])
    fit_frame = train_sectors.merge(
        laps[
            ["event", "driver", "lap_number", "tyre_age", "fuel_proxy", "track_temp", "stint_lap"]
        ],
        on=["event", "driver", "lap_number"],
    )
    scaler = ChannelScaler.fit(fit_frame)

    datasets = {}
    for split, event_list in split_events.items():
        sectors = train_sectors if split == "train" else load_sectors(event_list)
        split_laps = laps[laps["event"].isin(event_list)]
        datasets[split] = build_samples(split_laps, sectors, scaler, vision)
        logger.info(
            "%s: %d samples, cliff rate %.1f%%",
            split,
            len(datasets[split].frames),
            100 * datasets[split].cliff.mean(),
        )
    return datasets, scaler


def quadrant_values(frames: np.ndarray) -> np.ndarray:
    """[n, L, K, 16]: the 16 information-carrying cells (blends are derived)."""
    return frames[:, :, :, QUAD_ROWS, QUAD_COLS]


def lgbm_baseline(
    x_train: np.ndarray,
    y_pace: np.ndarray,
    y_cliff: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the LightGBM baseline in a separate process via file exchange.

    Never import lightgbm into the study process — its Homebrew libomp
    deadlocks against PyTorch's bundled libomp (see lgbm_worker docstring).
    """
    with tempfile.TemporaryDirectory() as tmp:
        in_path, out_path = f"{tmp}/in.npz", f"{tmp}/out.npz"
        np.savez(in_path, x_train=x_train, y_pace=y_pace, y_cliff=y_cliff, x_test=x_test, seed=seed)
        subprocess.run(
            [sys.executable, "-m", "f1_vision.train.lgbm_worker", in_path, out_path],
            check=True,
        )
        result = np.load(out_path)
        return result["pace"], result["cliff"]


def train_torch_model(
    model: torch.nn.Module,
    x_train: np.ndarray,
    train: VisionDataset,
    x_val: np.ndarray,
    val: VisionDataset,
    cfg: Any,
) -> torch.nn.Module:
    """AdamW + cosine schedule + early stopping on val multitask loss."""
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = model.to(device)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train), torch.tensor(train.pace_delta), torch.tensor(train.cliff)
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    xv = torch.tensor(x_val, device=device)
    yv_pace = torch.tensor(val.pace_delta, device=device)
    yv_cliff = torch.tensor(val.cliff, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    best_loss, best_state, stale = float("inf"), None, 0
    for epoch in range(cfg.epochs):
        model.train()
        for xb, pb, cb in loader:
            xb, pb, cb = xb.to(device), pb.to(device), cb.to(device)
            optimizer.zero_grad()
            pace, cliff = model(xb)
            loss = multitask_loss(pace, cliff, pb, cb, cfg.lambda_cliff)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pace, cliff = model(xv)
            val_loss = float(multitask_loss(pace, cliff, yv_pace, yv_cliff, cfg.lambda_cliff))
        if val_loss < best_loss - 1e-4:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                logger.info("early stop at epoch %d (best val %.4f)", epoch, best_loss)
                break
    model.load_state_dict(best_state)
    return model


def torch_predictions(model: torch.nn.Module, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    pace_out, cliff_out = [], []
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            pace, cliff = model(torch.tensor(x[start : start + 1024], device=device))
            pace_out.append(pace.cpu().numpy())
            cliff_out.append(torch.sigmoid(cliff).cpu().numpy())
    return np.concatenate(pace_out), np.concatenate(cliff_out)


def evaluate(
    name: str, pace_pred: np.ndarray, cliff_score: np.ndarray, ds: VisionDataset
) -> pd.DataFrame:
    rows = [
        {
            "model": name,
            "event": "ALL",
            "n": len(ds.frames),
            "mae": mean_absolute_error(ds.pace_delta, pace_pred),
            "pr_auc": average_precision_score(ds.cliff, cliff_score),
        }
    ]
    for event, idx in ds.meta.groupby("event").groups.items():
        loc = ds.meta.index.get_indexer(idx)
        row = {
            "model": name,
            "event": str(event),
            "n": len(loc),
            "mae": mean_absolute_error(ds.pace_delta[loc], pace_pred[loc]),
            "pr_auc": (
                average_precision_score(ds.cliff[loc], cliff_score[loc])
                if ds.cliff[loc].any()
                else float("nan")
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def load_checkpoint(path: Path | str) -> torch.nn.Module:
    """The one factory (spec §2.7): rebuild and load any saved study model."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model_cls = {"cnn_lstm": CNNLSTM, "lstm_only": LSTMOnly}[payload["kind"]]
    model = model_cls(**payload["kwargs"])
    model.load_state_dict(payload["state"])
    model.eval()
    return model


def run_study(core: F1CoreConfig, vision: F1VisionConfig, out_dir: Path) -> pd.DataFrame:
    cfg = vision.train
    datasets, scaler = assemble_datasets(core, vision)
    train, val, test = datasets["train"], datasets["val"], datasets["test"]

    quad = {k: quadrant_values(d.frames) for k, d in datasets.items()}
    flat = {k: q.reshape(len(q), -1) for k, q in quad.items()}
    lap_agg = {k: q.mean(axis=2) for k, q in quad.items()}  # [n, L, 16]

    results = []

    # 1. Persistence
    results.append(
        evaluate(
            "persistence",
            test.meta["pace_delta_current"].to_numpy(),
            test.meta["pace_delta_current"].to_numpy(),
            test,
        )
    )

    # 2. LightGBM on flattened mini-sector features (subprocess: see _lgbm_child)
    lgb_pace, lgb_cliff = lgbm_baseline(
        flat["train"], train.pace_delta, train.cliff, flat["test"], cfg.seed
    )
    results.append(evaluate("lgbm_flat", lgb_pace, lgb_cliff, test))
    logger.info("lgbm_flat done")

    # 3. LSTM-only on per-lap aggregates
    logger.info("training lstm_only")
    lstm = train_torch_model(
        LSTMOnly(in_dim=lap_agg["train"].shape[-1], hidden=cfg.hidden),
        lap_agg["train"],
        train,
        lap_agg["val"],
        val,
        cfg,
    )
    results.append(evaluate("lstm_only", *torch_predictions(lstm, lap_agg["test"]), test))

    # 4. CNN-LSTM on full frames
    logger.info("training cnn_lstm")
    cnn = train_torch_model(
        CNNLSTM(embed_dim=cfg.embed_dim, hidden=cfg.hidden),
        train.frames,
        train,
        val.frames,
        val,
        cfg,
    )
    results.append(evaluate("cnn_lstm", *torch_predictions(cnn, test.frames), test))

    ckpt_dir = core.ingest.output_dir / "vision" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    scaler.save(ckpt_dir / "scaler.pkl")
    torch.save(
        {
            "kind": "cnn_lstm",
            "kwargs": {"embed_dim": cfg.embed_dim, "hidden": cfg.hidden},
            "state": cnn.cpu().state_dict(),
        },
        ckpt_dir / "cnn_lstm.pt",
    )
    torch.save(
        {
            "kind": "lstm_only",
            "kwargs": {"in_dim": lap_agg["train"].shape[-1], "hidden": cfg.hidden},
            "state": lstm.cpu().state_dict(),
        },
        ckpt_dir / "lstm_only.pt",
    )

    report = pd.concat(results, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_dir / "baseline_comparison.csv", index=False)
    return report
