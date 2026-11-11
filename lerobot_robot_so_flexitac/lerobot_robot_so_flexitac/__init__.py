"""SO-100/101/10X follower robot plugin with lossless float32 FlexiTac tactile features.

Because stock LeRobot auto-imports installed ``lerobot_robot_*`` packages on CLI
startup, importing ``lerobot_tactile_flexitac`` here also registers the ``flexitac``
stream sensor type automatically on unmodified stock LeRobot.
"""

try:
    import lerobot_tactile_flexitac  # noqa: F401
except ImportError:
    pass

from .config_so_tactile import FlexiTacSensorEntryConfig, SOTactileFollowerConfig
from .so_tactile import (
    SOTactileFollower,
    flatten_tactile_names,
    unflatten_tactile_state,
)

__all__ = [
    "FlexiTacSensorEntryConfig",
    "SOTactileFollowerConfig",
    "SOTactileFollower",
    "flatten_tactile_names",
    "unflatten_tactile_state",
]
