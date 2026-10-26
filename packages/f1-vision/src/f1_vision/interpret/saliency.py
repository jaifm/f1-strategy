"""Grad-CAM saliency atlas (spec §2.6).

LayerGradCam (captum) on the CNN encoder's second conv layer, targeting the
pace-regression head. Per-sample maps are the mean over the sample's L·K
frames; the atlas averages them for three tire states — (a) fresh (tyre age
≤ 5), (b) mid-stint (10–20), (c) pre-cliff (cliff_within_5 = 1) — and reports
attention share per grid quadrant. Hypothesis under test: attention shifts
toward Quadrant I (Energy) as degradation approaches.

Context for reading the maps: the §2.5 study found the CNN-LSTM loses to
persistence, so these maps explain what a *weak* model attends to — reported
per the spec's deliverable list, not as evidence the textures are predictive.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

QUADRANTS = {
    "I_energy": (slice(0, 2), slice(0, 2)),
    "II_dynamics": (slice(0, 2), slice(3, 5)),
    "III_driver": (slice(3, 5), slice(0, 2)),
    "IV_context": (slice(3, 5), slice(3, 5)),
}


def gradcam_maps(model: torch.nn.Module, frames: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """[n, 5, 5] Grad-CAM maps (pace head target), averaged over each sample's frames."""
    from captum.attr import LayerGradCam

    conv2 = model.encoder.net[3]  # second Conv2d
    gradcam = LayerGradCam(lambda x: model(x)[0], conv2)
    maps = []
    for start in range(0, len(frames), batch_size):
        chunk = frames[start : start + batch_size]
        n, laps, k = chunk.shape[:3]
        attr = gradcam.attribute(torch.tensor(chunk), relu_attributions=True)
        maps.append(attr.reshape(n, laps * k, 5, 5).mean(dim=1).detach().numpy())
    return np.concatenate(maps)


def quadrant_shares(saliency: np.ndarray) -> dict[str, float]:
    """Share of total attention per quadrant (center cells excluded)."""
    total = float(saliency.sum()) or 1.0
    return {
        name: float(saliency[rows, cols].sum()) / total for name, (rows, cols) in QUADRANTS.items()
    }


def build_atlas(
    maps: np.ndarray,
    tyre_age: np.ndarray,
    cliff: np.ndarray,
    out_dir: Path,
    min_shift: float = 0.02,
) -> pd.DataFrame:
    """Write the atlas PNG + markdown report; return the quadrant-share table."""
    groups = {
        "fresh (age ≤ 5)": tyre_age <= 5,
        "mid-stint (10–20)": (tyre_age >= 10) & (tyre_age <= 20),
        "pre-cliff (label = 1)": cliff == 1,
    }
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (name, mask) in zip(axes, groups.items(), strict=True):
        mean_map = maps[mask].mean(axis=0)
        display = mean_map / (mean_map.max() or 1.0)
        ax.imshow(display, cmap="inferno", vmin=0, vmax=1)
        ax.set_title(f"{name}\nn={int(mask.sum())}", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
        rows.append({"group": name, "n": int(mask.sum()), **quadrant_shares(mean_map)})
    fig.suptitle("Grad-CAM saliency atlas — CNN encoder, pace head (test events)")
    fig.tight_layout()
    fig.savefig(out_dir / "saliency_atlas.png", dpi=150)
    plt.close(fig)

    table = pd.DataFrame(rows)
    fresh_i, mid_i, cliff_i = table["I_energy"].tolist()
    # A shift must clear BOTH earlier groups by ``min_shift`` (2 points of
    # attention share by default): a bare "pre-cliff > fresh" comparison
    # labelled a 0.8-point, non-monotonic wobble as support.
    shift = cliff_i - max(fresh_i, mid_i)
    verdict = (
        "SUPPORTED"
        if shift >= min_shift
        else f"NOT SUPPORTED (pre-cliff exceeds earlier groups by {shift:+.1%}, "
        f"threshold {min_shift:.0%})"
    )
    report = (
        "# Saliency Atlas (spec §2.6)\n\n"
        "Grad-CAM on the CNN encoder (pace head), held-out test events, averaged per group.\n"
        "NB: the §2.5 study found this model loses to persistence — the atlas explains a\n"
        "weak model's attention, it is not evidence the textures are predictive.\n\n"
        f"```\n{table.round(3).to_string(index=False)}\n```\n\n"
        f"**Hypothesis** (attention shifts toward Quadrant I / Energy pre-cliff): "
        f"**{verdict}** — Quadrant I share {fresh_i:.1%} (fresh) → {mid_i:.1%} (mid-stint) "
        f"→ {cliff_i:.1%} (pre-cliff).\n\n"
        "![atlas](saliency_atlas.png)\n"
    )
    (out_dir / "saliency_atlas.md").write_text(report)
    logger.info("atlas -> %s", out_dir / "saliency_atlas.md")
    return table
