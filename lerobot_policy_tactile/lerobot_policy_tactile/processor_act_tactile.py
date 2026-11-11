"""Pre- and post-processor factory for `act_tactile` (discovered by `lerobot.policies.factory`)."""

from __future__ import annotations

from typing import Any

import torch

from lerobot.policies.act.processor_act import make_act_pre_post_processors

from .configuration_act_tactile import ACTTactileConfig


def make_act_tactile_pre_post_processors(
    config: ACTTactileConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    """Build normalization and device-placement pipelines for ``ACTTactilePolicy``."""
    return make_act_pre_post_processors(config, dataset_stats=dataset_stats)
