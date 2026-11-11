"""Configuration for the ACT + TactileTokenEncoder policy plugin (`act_tactile`)."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_tactile")
@dataclass
class ACTTactileConfig(ACTConfig):
    """ACT policy extended with 2D spatial tactile token encoding (`TactileTokenEncoder`)."""

    use_tactile: bool = True
    tactile_encoder_type: str = "attention"  # "cnn" or "attention"
    tactile_input_shape: tuple[int, int] = (12, 32)
    n_tactile_tokens: int = 4
    tactile_dropout: float = 0.3

    # Number of leading proprioceptive joint dimensions in `observation.state` before
    # flattened FlexiTac taxels begin (6 for SO-100/101 single arm).
    proprio_state_dim: int = 6
    n_tactile_sensors: int = 1

    # Optional explicit keys (e.g. ["observation.images.tactile_primary"] or ["observation.tactile.primary"]).
    # Leave empty when tactile is flattened inside `observation.state` via `SOTactileFollower`.
    tactile_keys: list[str] = field(default_factory=list)
