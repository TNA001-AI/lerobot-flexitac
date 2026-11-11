"""Normalization-statistics helpers for tactile columns stored in ``observation.state``."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch

_QUANTILE_KEY = re.compile(r"^q(\d{1,2})$")


def _identity_value(stat_name: str) -> float | None:
    """Stat value that makes MEAN_STD / MIN_MAX / QUANTILES act as identity on ``[0, 1]`` data."""
    fixed = {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 1.0}
    if stat_name in fixed:
        return fixed[stat_name]
    match = _QUANTILE_KEY.match(stat_name)
    if match:
        return int(match.group(1)) / 100.0
    return None


def override_tactile_stats(
    dataset_stats: dict[str, dict[str, Any]] | None,
    proprio_state_dim: int,
    state_key: str = "observation.state",
) -> dict[str, dict[str, Any]] | None:
    """Return a copy of ``dataset_stats`` where tactile state columns are left unnormalized.

    FlexiTac values are already in ``[0, 1]``. Per-column MEAN_STD stats are harmful here:
    taxels that are rarely touched have std close to 0, so a contact would be scaled by
    ``1 / std`` into huge values. Columns ``[proprio_state_dim:]`` therefore get stats that
    make every normalization mode an identity, while proprioceptive columns are untouched.
    """
    if not dataset_stats or state_key not in dataset_stats:
        return dataset_stats

    out = dict(dataset_stats)
    state_stats = dict(out[state_key])
    for stat_name, value in state_stats.items():
        identity = _identity_value(stat_name)
        if identity is None or value is None:
            continue
        if isinstance(value, torch.Tensor):
            patched = value.clone()
        else:
            patched = np.array(value, dtype=np.float64 if not isinstance(value, np.ndarray) else value.dtype, copy=True)
        if patched.ndim == 0 or patched.shape[-1] <= proprio_state_dim:
            continue
        patched[..., proprio_state_dim:] = identity
        state_stats[stat_name] = patched
    out[state_key] = state_stats
    return out
