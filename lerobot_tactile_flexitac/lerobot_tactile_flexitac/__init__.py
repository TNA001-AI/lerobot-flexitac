"""LeRobot tactile plugin for FlexiTac (12x32) array tactile sensors."""

from .config_flexitac_tactile import FlexiTacTactileConfig
from .flexitac_tactile import (
    FlexiTacTactile,
    decode_tactile_image,
    encode_tactile_map,
)

__all__ = [
    "FlexiTacTactileConfig",
    "FlexiTacTactile",
    "encode_tactile_map",
    "decode_tactile_image",
]
