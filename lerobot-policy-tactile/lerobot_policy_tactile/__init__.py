"""LeRobot tactile policy plugin and reusable 2D tactile token encoders.

Importing this package registers the ``act_tactile`` policy type.
"""

from .configuration_act_tactile import ACTTactileConfig
from .encoder import TactileAttentionCNN, TactileCNN, TactileTokenEncoder, extract_tactile_maps_from_batch
from .layout import TactileStateLayout, infer_tactile_layout
from .normalization import override_tactile_stats

__all__ = [
    "ACTTactileConfig",
    "TactileCNN",
    "TactileAttentionCNN",
    "TactileTokenEncoder",
    "extract_tactile_maps_from_batch",
    "TactileStateLayout",
    "infer_tactile_layout",
    "override_tactile_stats",
]
__version__ = "0.2.0"
