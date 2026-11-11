"""Shared spatial tactile encoders (CNN & Spatial-Attention CNN) and batch extraction helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class TactileCNN(nn.Module):
    """Tactile CNN backbone — maps ``(B, H, W)`` tactile grids to ``(B, feature_dim)``."""

    def __init__(
        self,
        input_shape: tuple[int, int] = (12, 32),
        feature_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        feature_h = input_shape[0] // 8
        feature_w = input_shape[1] // 8
        if feature_h <= 0 or feature_w <= 0:
            raise ValueError(f"input_shape {input_shape} must be >= (8, 8) for 3x MaxPool2d(2, 2)")
        conv_output_dim = 128 * feature_h * feature_w

        self.fc1 = nn.Linear(conv_output_dim, 512)
        self.fc2 = nn.Linear(512, feature_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class TactileAttentionCNN(nn.Module):
    """Tactile CNN with spatial attention + global avg/max pooling — outputs ``(B, feature_dim)``."""

    def __init__(
        self,
        input_shape: tuple[int, int] = (12, 32),
        feature_dim: int = 256,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        self.attention = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, feature_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x * self.attention(x)
        pooled = torch.cat([self.global_avg_pool(x), self.global_max_pool(x)], dim=1)
        return self.fc(pooled.reshape(x.size(0), -1))


class TactileTokenEncoder(nn.Module):
    """Wraps a tactile CNN backbone to produce ``(B, n_tokens, feature_dim)`` token embeddings."""

    def __init__(
        self,
        encoder_type: str = "attention",
        input_shape: tuple[int, int] = (12, 32),
        feature_dim: int = 256,
        n_tokens: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        if n_tokens <= 0:
            raise ValueError(f"n_tokens must be >= 1, got {n_tokens}")
        self.n_tokens = n_tokens
        self.feature_dim = feature_dim

        if encoder_type == "cnn":
            self.backbone: nn.Module = TactileCNN(input_shape, feature_dim, dropout)
        elif encoder_type == "attention":
            self.backbone = TactileAttentionCNN(input_shape, feature_dim, dropout)
        else:
            raise ValueError(
                f"Unknown tactile encoder_type: {encoder_type!r}. Choose 'cnn' or 'attention'."
            )

        self.token_proj = nn.Linear(feature_dim, n_tokens * feature_dim) if n_tokens > 1 else None

    def forward(self, x: Tensor) -> Tensor:
        feat = self.backbone(x)
        if self.token_proj is None:
            return feat.unsqueeze(1)
        projected = self.token_proj(feat)
        return projected.reshape(feat.size(0), self.n_tokens, self.feature_dim)


def extract_tactile_maps_from_batch(
    batch: dict[str, Tensor],
    *,
    tactile_shape: tuple[int, int] = (12, 32),
    proprio_dim: int = 6,
    n_tactile_sensors: int = 1,
    tactile_keys: list[str] | None = None,
    state_key: str = "observation.state",
) -> tuple[Tensor | None, list[Tensor]]:
    """Split a batch into proprioceptive state and ``(B, rows, cols)`` tactile maps.

    Two explicit modes (no guessing from tensor sizes):

    - ``tactile_keys`` set: read each key. ``(B, H, W)`` tensors (``observation.tactile.*``)
      are used as-is; ``(B, C, H, W)`` image streams (``observation.images.tactile_*``) are
      averaged over channels and nearest-resized to ``tactile_shape``. Note image streams
      arrive already normalized by the policy preprocessor. State is returned unchanged.
    - otherwise: ``observation.state`` must be exactly
      ``proprio_dim + n_tactile_sensors * rows * cols`` wide, as produced by
      ``SOTactileFollower`` (use ``infer_tactile_layout`` to get these numbers).
    """
    rows, cols = tactile_shape
    state = batch.get(state_key)

    if tactile_keys:
        maps: list[Tensor] = []
        for key in tactile_keys:
            if key not in batch:
                raise KeyError(f"Configured tactile key {key!r} not found in batch keys {sorted(batch)}")
            t = batch[key].float()
            if t.dim() == 4:
                gray = t.mean(dim=1, keepdim=True)
                if gray.shape[-2:] != (rows, cols):
                    gray = F.interpolate(gray, size=(rows, cols), mode="nearest")
                t = gray.squeeze(1)
            elif t.dim() != 3 or tuple(t.shape[-2:]) != (rows, cols):
                raise ValueError(f"Unsupported tactile tensor shape {tuple(t.shape)} for key {key!r}")
            maps.append(t)
        return state, maps

    if state is None:
        raise KeyError(f"{state_key!r} is required when tactile is stored in the state (no tactile_keys given)")
    expected = proprio_dim + n_tactile_sensors * rows * cols
    if state.shape[-1] != expected:
        raise ValueError(
            f"{state_key} has {state.shape[-1]} columns but the configured layout needs {expected} "
            f"(proprio_dim={proprio_dim} + {n_tactile_sensors} x {rows}x{cols}). "
            "Derive the layout from the dataset with lerobot_policy_tactile.infer_tactile_layout()."
        )
    lead = state.shape[:-1]
    proprio = state[..., :proprio_dim]
    pads = state[..., proprio_dim:].reshape(*lead, n_tactile_sensors, rows, cols)
    return proprio, [pads[..., i, :, :] for i in range(n_tactile_sensors)]
