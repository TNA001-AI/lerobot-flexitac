"""LeRobot tactile plugin for FlexiTac (12x32) array tactile sensors.

Importing this package registers the ``flexitac`` stream device type with LeRobot.
"""

from .config_flexitac_tactile import FlexiTacTactileConfig
from .flexitac_tactile import (
    FlexiTacTactile,
    FrameListener,
    TactileShapeError,
    decode_tactile_image,
    encode_tactile_map,
    frame_to_tactile_map,
    make_flexitac_sensor,
)

__all__ = [
    "FlexiTacTactileConfig",
    "FlexiTacTactile",
    "FrameListener",
    "TactileShapeError",
    "decode_tactile_image",
    "encode_tactile_map",
    "frame_to_tactile_map",
    "make_flexitac_sensor",
]
__version__ = "0.2.0"
