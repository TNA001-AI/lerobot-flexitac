"""Configuration for the SO-100/101/10X + FlexiTac follower robot plugin."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig

try:
    from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig as _BaseSOConfig

    class _SORobotConfigBase(RobotConfig, _BaseSOConfig):
        pass
except ImportError:
    # Compatibility with LeRobot 0.3.x / 0.4.x
    from lerobot.robots.so101_follower.config_so101_follower import (
        SO101FollowerConfig as _SORobotConfigBase,
    )


@dataclass
class FlexiTacSensorEntryConfig:
    """Per-sensor configuration inside ``SOTactileFollowerConfig.tactile_sensors``."""

    port: str = "/dev/ttyUSB0"
    baud_rate: int = 2_000_000
    shape: tuple[int, int] = (12, 32)
    baseline: float | None = None
    init_frames: int = 30
    threshold: float = 25.0
    noise_scale: float = 30.0
    temporal_alpha: float = 0.2
    process_backend: bool = False

    # Optional session-long raw CSV logging stamped with epoch_ns
    record_raw_csv: bool = False
    raw_csv_dir: str | None = None


@RobotConfig.register_subclass("so_tactile_follower")
@dataclass
class SOTactileFollowerConfig(_SORobotConfigBase):
    """SO-100 / SO-101 follower arm extended with lossless float32 FlexiTac tactile sensors.

    CLI Example::

        lerobot-record \\
          --robot.type=so_tactile_follower \\
          --robot.port=/dev/ttyACM0 \\
          --robot.tactile_sensors='{primary: {port: "/dev/ttyUSB0", baud_rate: 2000000}}'
    """

    tactile_sensors: dict[str, FlexiTacSensorEntryConfig] = field(default_factory=dict)
