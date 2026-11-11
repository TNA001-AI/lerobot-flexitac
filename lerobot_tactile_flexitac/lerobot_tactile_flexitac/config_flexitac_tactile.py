"""Configuration for the FlexiTac (12x32) tactile sensor plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lerobot.cameras.configs import CameraConfig as _StreamSensorConfig


@_StreamSensorConfig.register_subclass("flexitac")
@dataclass(kw_only=True)
class FlexiTacTactileConfig(_StreamSensorConfig):
    """Configuration for a FlexiTac tactile sensor connected over USB serial.

    Rendering modes:
    - ``render="raw"`` (default): Reversibly encodes normalized ``[0, 1]`` taxel
      readings into ``(H, W, 3)`` ``uint8``. Defaults to native ``(12, 32)``
      resolution unless ``width``/``height`` are explicitly set (nearest-neighbor
      resize preserves exact block values). Record with ``--dataset.video=false``
      (PNG) if lossless ``uint8`` recovery is needed downstream.
    - ``render="heatmap"``: Applies EMA smoothing + Viridis colormap, defaulting
      to ``96 x 256`` (8x upscale) for human-friendly Rerun visualization.
    """

    port: str = "/dev/ttyUSB0"
    baud_rate: int = 2_000_000
    tactile_shape: tuple[int, int] = (12, 32)
    baseline: float | None = None
    init_frames: int = 30
    threshold: float = 25.0
    noise_scale: float = 30.0
    temporal_alpha: float = 0.2
    render: Literal["raw", "heatmap"] = "raw"

    # Multiprocessing + shared_memory backend for multi-sensor setups at 2Mbaud
    process_backend: bool = False
    process_connect_timeout_s: float = 15.0

    def __post_init__(self) -> None:
        if self.render not in ("raw", "heatmap"):
            raise ValueError(f"render must be 'raw' or 'heatmap', got {self.render!r}")
        if len(self.tactile_shape) != 2 or self.tactile_shape[0] <= 0 or self.tactile_shape[1] <= 0:
            raise ValueError(f"tactile_shape must be positive (rows, cols), got {self.tactile_shape!r}")
        if not (0.0 < self.temporal_alpha <= 1.0):
            raise ValueError(f"temporal_alpha must be in (0, 1], got {self.temporal_alpha}")
        if self.process_connect_timeout_s <= 0:
            raise ValueError(
                f"process_connect_timeout_s must be positive, got {self.process_connect_timeout_s}"
            )

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
