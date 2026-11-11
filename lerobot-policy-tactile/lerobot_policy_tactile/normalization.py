"""Recover raw tactile values from normalized ``observation.state`` inside the model.

LeRobot normalizes ``observation.state`` per column before the model sees it (in the
preprocessor pipeline on LeRobot >= 0.4, inside the policy on 0.3.x). For FlexiTac taxels
that is harmful: rarely touched taxels have std close to 0, so contacts become huge values.

Which processor runs depends on the LeRobot version (0.4.x dispatches ``isinstance(cfg,
ACTConfig)`` before plugin processors), so instead of trying to skip normalization we undo
it in the model. Every LeRobot mode is a per-column affine map, so the model stores
``x = x_norm * scale + offset`` for the tactile columns as checkpoint buffers and feeds the
recovered ``[0, 1]`` taxels to the tactile encoder. Proprioception stays normalized.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

# Must match LeRobot's NormalizerProcessorStep default.
EPS = 1e-8

_AFFINE_MODES = {
    "MEAN_STD": ("mean", "std"),
    "MIN_MAX": ("min", "max"),
    "QUANTILES": ("q01", "q99"),
    "QUANTILE10": ("q10", "q90"),
}


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32, device="cpu")
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def _mode_name(mode: Any) -> str:
    name = getattr(mode, "name", None) or getattr(mode, "value", None) or str(mode)
    return str(name).split(".")[-1].upper()


def tactile_denorm_params(
    state_stats: dict[str, Any],
    mode: Any,
    start: int,
    eps: float = EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-column ``(scale, offset)`` such that ``raw = normalized * scale + offset``.

    Args:
        state_stats: ``dataset_stats["observation.state"]``.
        mode: the ``NormalizationMode`` used for STATE features.
        start: index of the first tactile column (``proprio_state_dim``).
    """
    name = _mode_name(mode)
    if name == "IDENTITY":
        raise ValueError("IDENTITY needs no denormalization; use scale=1, offset=0")
    if name not in _AFFINE_MODES:
        raise ValueError(f"Unsupported STATE normalization mode for tactile columns: {name}")

    a_key, b_key = _AFFINE_MODES[name]
    if a_key not in state_stats or b_key not in state_stats:
        raise KeyError(f"{name} normalization needs '{a_key}' and '{b_key}' in observation.state stats")
    a = _as_tensor(state_stats[a_key])[..., start:]
    b = _as_tensor(state_stats[b_key])[..., start:]

    if name == "MEAN_STD":  # x' = (x - mean) / (std + eps)
        return b + eps, a
    # Range modes: x' = 2 (x - lo) / d - 1, with d -> eps where d == 0
    d = b - a
    d = torch.where(d == 0, torch.full_like(d, eps), d)
    return d / 2.0, d / 2.0 + a
