"""Configuration for the ACT + TactileTokenEncoder policy plugin (`act_tactile`)."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_tactile")
@dataclass
class ACTTactileConfig(ACTConfig):
    """ACT extended with 2D spatial tactile tokens (``TactileTokenEncoder``).

    Tactile source (pick one):
    - **State mode** (default, ``tactile_keys`` empty): tactile is the contiguous tail of
      ``observation.state`` as recorded by ``so_tactile_follower``. ``proprio_state_dim`` and
      ``n_tactile_sensors`` must describe it exactly; get them with
      ``infer_tactile_layout(names).cli_args()``. The transformer's state token then only sees
      the proprioceptive part, and tactile columns are excluded from normalization when
      ``override_tactile_normalization`` is set.
    - **Key mode**: list tensor keys (``observation.tactile.*``) or tactile image streams
      (``observation.images.tactile_*``). Listed image keys are routed only to the tactile
      encoder, not to the ResNet backbone.
    """

    use_tactile: bool = True
    tactile_encoder_type: str = "attention"  # "cnn" or "attention"
    tactile_input_shape: tuple[int, int] = (12, 32)
    n_tactile_tokens: int = 4
    tactile_dropout: float = 0.3

    proprio_state_dim: int = 6
    n_tactile_sensors: int = 1
    tactile_keys: list[str] = field(default_factory=list)
    override_tactile_normalization: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tactile_encoder_type not in ("cnn", "attention"):
            raise ValueError(f"tactile_encoder_type must be 'cnn' or 'attention', got {self.tactile_encoder_type!r}")
        self.tactile_input_shape = (int(self.tactile_input_shape[0]), int(self.tactile_input_shape[1]))
        if min(self.tactile_input_shape) < 8:
            raise ValueError(f"tactile_input_shape must be at least 8x8, got {self.tactile_input_shape}")
        if self.n_tactile_tokens < 1 or self.n_tactile_sensors < 1 or self.proprio_state_dim < 0:
            raise ValueError("n_tactile_tokens and n_tactile_sensors must be >= 1, proprio_state_dim >= 0")

    @property
    def tactile_in_state(self) -> bool:
        return self.use_tactile and not self.tactile_keys

    @property
    def n_tactile_inputs(self) -> int:
        return len(self.tactile_keys) if self.tactile_keys else self.n_tactile_sensors

    @property
    def expected_state_dim(self) -> int:
        rows, cols = self.tactile_input_shape
        return self.proprio_state_dim + self.n_tactile_sensors * rows * cols

    def validate_tactile_layout(self) -> None:
        """Raise a clear error if the dataset's state width doesn't match the configured layout."""
        if not self.tactile_in_state:
            return
        state_ft = self.robot_state_feature
        if state_ft is None:
            raise ValueError("act_tactile in state mode needs an 'observation.state' input feature")
        if state_ft.shape[0] != self.expected_state_dim:
            raise ValueError(
                f"observation.state has {state_ft.shape[0]} dims but act_tactile expects "
                f"{self.expected_state_dim} = proprio_state_dim ({self.proprio_state_dim}) + "
                f"n_tactile_sensors ({self.n_tactile_sensors}) x {self.tactile_input_shape}. "
                "Run lerobot_policy_tactile.infer_tactile_layout(state_names).cli_args() to get the right flags."
            )
