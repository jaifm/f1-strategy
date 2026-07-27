"""Model forward shapes, loss, and a tiny overfit check on synthetic frames."""

import numpy as np
import torch

from f1_vision.models.nets import CNNLSTM, LSTMOnly, multitask_loss


def test_cnn_lstm_forward_shapes() -> None:
    model = CNNLSTM(embed_dim=16, hidden=32)
    pace, cliff = model(torch.rand(4, 5, 20, 5, 5))
    assert pace.shape == cliff.shape == (4,)


def test_lstm_only_forward_shapes() -> None:
    model = LSTMOnly(in_dim=16, hidden=32)
    pace, cliff = model(torch.rand(4, 5, 16))
    assert pace.shape == cliff.shape == (4,)


def test_multitask_loss_combines_terms() -> None:
    pace_pred, pace_true = torch.tensor([1.0]), torch.tensor([1.0])
    logit, cliff_true = torch.tensor([100.0]), torch.tensor([1.0])
    # perfect predictions -> both terms ~0
    assert float(multitask_loss(pace_pred, logit, pace_true, cliff_true, 0.5)) < 1e-3
    bad = float(multitask_loss(pace_pred, -logit, pace_true, cliff_true, 0.5))
    assert bad > 1.0  # confident wrong classification dominates


def test_cnn_lstm_can_overfit_tiny_batch() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.uniform(0, 1, (16, 5, 20, 5, 5)).astype(np.float32))
    y_pace = torch.tensor(rng.normal(0, 1, 16).astype(np.float32))
    y_cliff = (y_pace > 0).float()
    model = CNNLSTM(embed_dim=16, hidden=32)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = last = None
    for _ in range(60):
        opt.zero_grad()
        pace, cliff = model(x)
        loss = multitask_loss(pace, cliff, y_pace, y_cliff, 0.5)
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss)
        last = float(loss)
    assert last < first * 0.5  # it learns
