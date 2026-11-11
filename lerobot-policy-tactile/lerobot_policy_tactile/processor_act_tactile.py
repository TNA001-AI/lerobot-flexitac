"""Pre/post-processor factory for `act_tactile` (resolved by name on LeRobot >= 0.5).

On LeRobot 0.4.x the factory matches ``isinstance(cfg, ACTConfig)`` first and uses ACT's
processors directly; both paths are equivalent because tactile denormalization happens
inside the model (see ``normalization.py``).
"""

from __future__ import annotations

from typing import Any

import torch

from .configuration_act_tactile import ACTTactileConfig


def make_act_tactile_pre_post_processors(
    config: ACTTactileConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    """Same pipelines as ACT."""
    # Imported lazily: processor pipelines only exist in LeRobot >= 0.4.
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    return make_act_pre_post_processors(config, dataset_stats=dataset_stats)
