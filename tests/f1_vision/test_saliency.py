"""Grad-CAM plumbing on a random-weight model + atlas rendering."""

from pathlib import Path

import numpy as np
import torch

from f1_vision.interpret.saliency import build_atlas, gradcam_maps, quadrant_shares
from f1_vision.models.nets import CNNLSTM


def test_gradcam_maps_shape_and_nonnegative() -> None:
    torch.manual_seed(0)
    model = CNNLSTM(embed_dim=16, hidden=32).eval()
    frames = np.random.default_rng(0).uniform(0, 1, (6, 5, 20, 5, 5)).astype(np.float32)
    maps = gradcam_maps(model, frames, batch_size=4)
    assert maps.shape == (6, 5, 5)
    assert np.isfinite(maps).all()
    assert (maps >= 0).all()  # relu_attributions


def test_quadrant_shares_sum_below_one() -> None:
    saliency = np.ones((5, 5))
    shares = quadrant_shares(saliency)
    assert set(shares) == {"I_energy", "II_dynamics", "III_driver", "IV_context"}
    # 16 of 25 cells are quadrant cells
    assert sum(shares.values()) == 16 / 25


def test_build_atlas_writes_artifacts(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    maps = rng.uniform(0, 1, (30, 5, 5))
    age = rng.integers(1, 30, 30).astype(float)
    cliff = (rng.uniform(size=30) > 0.7).astype(np.float32)
    table = build_atlas(maps, age, cliff, tmp_path)
    assert (tmp_path / "saliency_atlas.png").exists()
    assert (tmp_path / "saliency_atlas.md").exists()
    assert len(table) == 3
