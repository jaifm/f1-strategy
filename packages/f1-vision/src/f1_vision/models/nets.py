"""CNN-LSTM and LSTM-only baselines (spec §2.5).

Input frames: ``[batch, L, K, 5, 5]``. The CNN encoder sees each of the L·K
frames independently (2 conv blocks, no pooling — the grid is only 5×5), the
LSTM runs over the L·K sequence, and the final hidden state feeds two heads:
regression (pace_delta_next3) and classification logit (cliff_within_5).
"""

from __future__ import annotations

from torch import Tensor, nn


class CNNEncoder(nn.Module):
    """2×(Conv3×3 → BN → ReLU), 16→32 channels, flatten → 64-d embedding."""

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, embed_dim),
        )

    def forward(self, frames: Tensor) -> Tensor:  # [n_frames, 1, 5, 5] -> [n_frames, embed]
        return self.net(frames)


class TwoHead(nn.Module):
    """Shared trunk output -> (pace regression, cliff logit)."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.pace = nn.Linear(in_dim, 1)
        self.cliff = nn.Linear(in_dim, 1)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        return self.pace(hidden).squeeze(-1), self.cliff(hidden).squeeze(-1)


class CNNLSTM(nn.Module):
    def __init__(self, embed_dim: int = 64, hidden: int = 128, layers: int = 1) -> None:
        super().__init__()
        self.encoder = CNNEncoder(embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=layers, batch_first=True)
        self.heads = TwoHead(hidden)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:  # x: [b, L, K, 5, 5]
        b, laps, k, h, w = x.shape
        embeds = self.encoder(x.reshape(b * laps * k, 1, h, w)).reshape(b, laps * k, -1)
        _, (h_n, _) = self.lstm(embeds)
        return self.heads(h_n[-1])


class LSTMOnly(nn.Module):
    """Baseline 3: LSTM over per-lap aggregate vectors (no CNN, no image)."""

    def __init__(self, in_dim: int = 16, hidden: int = 128, layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True)
        self.heads = TwoHead(hidden)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:  # x: [b, L, in_dim]
        _, (h_n, _) = self.lstm(x)
        return self.heads(h_n[-1])


def multitask_loss(
    pace_pred: Tensor,
    cliff_logit: Tensor,
    pace_true: Tensor,
    cliff_true: Tensor,
    lambda_cliff: float,
) -> Tensor:
    """L = Huber(pace) + λ·BCE(cliff) (spec §2.2)."""
    huber = nn.functional.huber_loss(pace_pred, pace_true)
    bce = nn.functional.binary_cross_entropy_with_logits(cliff_logit, cliff_true)
    return huber + lambda_cliff * bce
