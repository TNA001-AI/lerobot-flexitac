"""Configuration for the SO-100/101/10X + FlexiTac follower robot plugin."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from lerobot.robots.config import RobotConfig

try:  # LeRobot >= 0.4 (unified SO follower)
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig as _SOBaseConfig
except ImportError:  # LeRobot 0.3.x
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig as _SOBaseConfig


@dataclass
class FlexiTacSensorEntryConfig:
    """One FlexiTac pad inside ``SOTactileFollowerConfig.tactile_sensors``."""

    port: str = "/dev/ttyUSB0"
    baud_rate: int = 2_000_000
    shape: tuple[int, int] = (12, 32)
    baseline: float | None = None
    init_frames: int = 30
    threshold: float = 25.0
    noise_scale: float = 30.0
    read_timeout_s: float = 1.0
    # "normalized": flexitac per-frame peak-normalized map (original LeFlexiTac datasets).
    # "baseline_subtracted": fixed global scale, preserves absolute pressure across frames.
    signal: Literal["normalized", "baseline_subtracted"] = "normalized"
    process_backend: bool = False

    # A frame older than this is considered stale for the current observation.
    max_age_ms: float = 200.0
    # Stale frames are bridged with the last valid frame (with a warning) for at most this
    # long; after that get_observation() raises instead of recording frozen tactile data.
    stale_timeout_s: float = 2.0

    # Full-rate raw sidecar (every sensor frame, not just every observation), stamped with
    # epoch_ns and the frame seq. Defaults to ~/.cache/lerobot_flexitac/raw_sessions/<ts>/.
    record_raw_csv: bool = False
    raw_csv_dir: str | None = None
    csv_flush_every: int = 100

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or min(self.shape) <= 0:
            raise ValueError(f"shape must be positive (rows, cols), got {self.shape!r}")
        self.shape = (int(self.shape[0]), int(self.shape[1]))
        if self.max_age_ms <= 0 or self.stale_timeout_s <= 0:
            raise ValueError("max_age_ms and stale_timeout_s must be positive")
        if self.csv_flush_every <= 0:
            raise ValueError("csv_flush_every must be positive")


@RobotConfig.register_subclass("so_tactile_follower")
@dataclass
class SOTactileFollowerConfig(_SOBaseConfig):
    """SO-100 / SO-101 follower arm plus lossless float32 FlexiTac tactile sensors.

    CLI example::

        lerobot-record \\
          --robot.type=so_tactile_follower \\
          --robot.port=/dev/ttyACM0 \\
          --robot.tactile_sensors='{primary: {port: "/dev/ttyUSB0"}}'
    """

    tactile_sensors: dict[str, FlexiTacSensorEntryConfig] = field(default_factory=dict)
