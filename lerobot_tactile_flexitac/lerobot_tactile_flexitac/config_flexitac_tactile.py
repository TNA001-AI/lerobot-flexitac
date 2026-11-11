"""Configuration for the FlexiTac (12x32) tactile sensor plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# LeRobot discovers stream devices through the CameraConfig choice registry. We register
# under it so the sensor plugs into `--robot.cameras=...`, but expose Tactile naming.
from lerobot.cameras.configs import CameraConfig as _StreamSensorConfig


@_StreamSensorConfig.register_subclass("flexitac")
@dataclass(kw_only=True)
class FlexiTacTactileConfig(_StreamSensorConfig):
    """Configuration for a FlexiTac tactile sensor connected over USB serial.

    Signal (what the float map means):
    - ``signal="normalized"`` (default): ``flexitac``'s ``frame.normalized``. NOTE: flexitac
      scales every frame by its own peak once contact exceeds ``threshold``, so values are
      relative *within* a frame and absolute pressure is not comparable across frames.
      Matches the original LeFlexiTac datasets.
    - ``signal="baseline_subtracted"``: ``clip(raw - baseline - threshold, 0, 100) / 100``,
      a fixed global scale that preserves absolute pressure across frames.

    Rendering (only affects the ``(H, W, 3)`` stream API):
    - ``render="raw"`` (default): reversible gray encoding at native ``12 x 32`` unless
      ``width``/``height`` are set (nearest-neighbor resize keeps exact block values).
      Record with ``--dataset.video=false`` to keep it lossless.
    - ``render="heatmap"``: EMA-smoothed (``temporal_alpha``) Viridis colormap, default
      ``96 x 256``, for Rerun visualization and demos.
    """

    port: str = "/dev/ttyUSB0"
    baud_rate: int = 2_000_000
    tactile_shape: tuple[int, int] = (12, 32)
    baseline: float | None = None
    init_frames: int = 30
    threshold: float = 25.0
    noise_scale: float = 30.0
    # Max time a single blocking serial frame read may take before flexitac raises.
    read_timeout_s: float = 1.0
    signal: Literal["normalized", "baseline_subtracted"] = "normalized"
    temporal_alpha: float = 0.2
    render: Literal["raw", "heatmap"] = "raw"

    # Run the serial reader in a separate process writing into shared memory. Useful when
    # several sensors stream at 2 Mbaud and GIL contention starves the control loop.
    process_backend: bool = False
    process_connect_timeout_s: float = 15.0
    # Max time connect() waits for the first frame (includes flexitac auto-calibration,
    # which reads `init_frames` frames).
    warmup_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if self.render not in ("raw", "heatmap"):
            raise ValueError(f"render must be 'raw' or 'heatmap', got {self.render!r}")
        if self.signal not in ("normalized", "baseline_subtracted"):
            raise ValueError(f"signal must be 'normalized' or 'baseline_subtracted', got {self.signal!r}")
        if len(self.tactile_shape) != 2 or self.tactile_shape[0] <= 0 or self.tactile_shape[1] <= 0:
            raise ValueError(f"tactile_shape must be positive (rows, cols), got {self.tactile_shape!r}")
        self.tactile_shape = (int(self.tactile_shape[0]), int(self.tactile_shape[1]))
        if not (0.0 < self.temporal_alpha <= 1.0):
            raise ValueError(f"temporal_alpha must be in (0, 1], got {self.temporal_alpha}")
        for name in ("read_timeout_s", "process_connect_timeout_s", "warmup_timeout_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

        if self.fps is None:
            self.fps = 30
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

        rows, cols = self.tactile_shape
        scale = 1 if self.render == "raw" else 8
        if self.height is None:
            self.height = rows * scale
        if self.width is None:
            self.width = cols * scale
