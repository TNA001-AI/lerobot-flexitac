"""SO-100/101/10X follower robot plugin with lossless float32 FlexiTac tactile features.

Stock LeRobot auto-imports installed ``lerobot_robot_*`` packages; because this package
depends on and imports ``lerobot_tactile_flexitac``, the ``flexitac`` stream device type is
registered at the same time.
"""

from .config_so_tactile import FlexiTacSensorEntryConfig, SOTactileFollowerConfig
from .so_tactile import SOTactileFollower, flatten_tactile_names, unflatten_tactile_state

__all__ = [
    "FlexiTacSensorEntryConfig",
    "SOTactileFollowerConfig",
    "SOTactileFollower",
    "flatten_tactile_names",
    "unflatten_tactile_state",
]
__version__ = "0.2.0"
