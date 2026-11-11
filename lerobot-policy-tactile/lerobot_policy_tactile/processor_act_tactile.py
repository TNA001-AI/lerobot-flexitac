"""Pre/post-processor factory for `act_tactile` (resolved by `lerobot.policies.factory`)."""

from __future__ import annotations

from typing import Any

import torch

from .configuration_act_tactile import ACTTactileConfig
from .normalization import override_tactile_stats


def make_act_tactile_pre_post_processors(
    config: ACTTactileConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    """ACT's pipelines, with tactile state columns excluded from normalization."""
    # Imported lazily: processor pipelines only exist in LeRobot >= 0.4.
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    if config.tactile_in_state and config.override_tactile_normalization:
        dataset_stats = override_tactile_stats(dataset_stats, config.proprio_state_dim)
    return make_act_pre_post_processors(config, dataset_stats=dataset_stats)
