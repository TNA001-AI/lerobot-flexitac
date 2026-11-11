"""SO-100/101/10X follower robot with lossless float32 FlexiTac tactile state features."""

from __future__ import annotations

import csv
from functools import cached_property
import logging
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

try:
    from lerobot.robots.so_follower.so_follower import SOFollower as _BaseSOFollower
except ImportError:
    from lerobot.robots.so101_follower.so101_follower import SO101Follower as _BaseSOFollower

from lerobot_tactile_flexitac import FlexiTacTactile, FlexiTacTactileConfig

from .config_so_tactile import FlexiTacSensorEntryConfig, SOTactileFollowerConfig

logger = logging.getLogger(__name__)


def flatten_tactile_names(sensor_name: str, shape: tuple[int, int] = (12, 32)) -> list[str]:
    """Generate element names in C-order matching ``np.ravel()`` for one ``(rows, cols)`` tactile pad."""
    rows, cols = shape
    names = [f"tactile.{sensor_name}.r{r:02d}_c{c:02d}" for r in range(rows) for c in range(cols)]
    assert len(names) == rows * cols
    return names


def unflatten_tactile_state(
    state_names: list[str],
    state_values: NDArray[Any],
    sensor_name: str = "primary",
    shape: tuple[int, int] = (12, 32),
) -> NDArray[np.float32]:
    """Extract and reshape a sensor's flattened ``observation.state`` columns back to ``(..., rows, cols)``."""
    expected_names = flatten_tactile_names(sensor_name, shape)
    name_to_idx = {name: i for i, name in enumerate(state_names)}
    missing = [n for n in expected_names if n not in name_to_idx]
    if missing:
        raise KeyError(
            f"Missing {len(missing)} tactile columns for sensor {sensor_name!r} (first: {missing[0]!r})"
        )
    indices = [name_to_idx[n] for n in expected_names]
    arr = np.asarray(state_values, dtype=np.float32)
    sliced = np.take(arr, indices, axis=-1)
    return sliced.reshape((*arr.shape[:-1], shape[0], shape[1]))


class SOTactileFollower(_BaseSOFollower):
    """SO-100 / SO-101 follower arm recording lossless float32 FlexiTac taxels in ``observation.state``."""

    config_class = SOTactileFollowerConfig
    name = "so_tactile_follower"

    def __init__(
        self,
        config: SOTactileFollowerConfig,
        *,
        tactile_factory: Callable[[str, FlexiTacSensorEntryConfig], Any] | None = None,
    ):
        super().__init__(config)
        self.config: SOTactileFollowerConfig = config
        self._tactile_sensors: dict[str, Any] = {}
        self._tactile_names: dict[str, list[str]] = {}
        self._last_valid_frames: dict[str, NDArray[np.float32]] = {}
        self._csv_files: dict[str, Any] = {}
        self._csv_writers: dict[str, Any] = {}

        for name, scfg in config.tactile_sensors.items():
            self._tactile_names[name] = flatten_tactile_names(name, scfg.shape)
            if tactile_factory is not None:
                self._tactile_sensors[name] = tactile_factory(name, scfg)
            else:
                tcfg = FlexiTacTactileConfig(
                    port=scfg.port,
                    baud_rate=scfg.baud_rate,
                    tactile_shape=scfg.shape,
                    baseline=scfg.baseline,
                    init_frames=scfg.init_frames,
                    threshold=scfg.threshold,
                    noise_scale=scfg.noise_scale,
                    temporal_alpha=scfg.temporal_alpha,
                    render="raw",
                    process_backend=scfg.process_backend,
                )
                self._tactile_sensors[name] = FlexiTacTactile(tcfg)

    # ---- Feature declaration (safe before connect()) ---------------------

    @property
    def _tactile_ft(self) -> dict[str, type]:
        return {
            col_name: float
            for names in self._tactile_names.values()
            for col_name in names
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        # Motors first, then lossless tactile float columns, then RGB/depth cameras
        return {**self._motors_ft, **self._tactile_ft, **self._cameras_ft}

    # ---- Lifecycle -------------------------------------------------------

    def _open_raw_csv_if_needed(self, name: str, scfg: FlexiTacSensorEntryConfig) -> None:
        if not scfg.record_raw_csv:
            return
        out_dir = Path(scfg.raw_csv_dir or f"tactile_raw_sessions/session_{int(time.time())}")
        out_dir.mkdir(parents=True, exist_ok=True)
        fpath = out_dir / f"flexitac_{name}.csv"
        fh = fpath.open("w", newline="", encoding="utf-8")
        writer = csv.writer(fh)
        writer.writerow(["epoch_ns", *self._tactile_names[name]])
        self._csv_files[name] = fh
        self._csv_writers[name] = writer
        logger.info(f"Recording raw tactile sidecar CSV to {fpath}")

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate=calibrate)
        for name, sensor in self._tactile_sensors.items():
            sensor.connect(warmup=True)
            self._open_raw_csv_if_needed(name, self.config.tactile_sensors[name])
            logger.info(f"Tactile sensor '{name}' connected ({ len(self._tactile_names[name]) } float features)")

    def get_observation(self) -> dict[str, Any]:
        obs = super().get_observation()

        for name, sensor in self._tactile_sensors.items():
            expected_shape = self.config.tactile_sensors[name].shape
            try:
                frame = np.asarray(sensor.read_latest_tactile(), dtype=np.float32)
                if frame.shape != expected_shape:
                    raise ValueError(
                        f"Tactile sensor '{name}' returned shape {frame.shape}, expected {expected_shape}"
                    )
                self._last_valid_frames[name] = frame
            except ValueError:
                # Schema mismatch must fail loudly rather than silently misaligning state columns
                raise
            except Exception as exc:
                logger.warning(
                    f"Tactile sensor '{name}' read failed ({exc}); reusing last valid frame."
                )
                frame = self._last_valid_frames.get(
                    name, np.zeros(expected_shape, dtype=np.float32)
                )

            flat = frame.ravel()
            if name in self._csv_writers:
                self._csv_writers[name].writerow([time.time_ns(), *flat.tolist()])

            for col_name, val in zip(self._tactile_names[name], flat, strict=True):
                obs[col_name] = float(val)

        return obs

    def disconnect(self) -> None:
        # Disconnect parent arm first while is_connected remains valid for @check_if_not_connected
        try:
            super().disconnect()
        finally:
            for name, fh in list(self._csv_files.items()):
                try:
                    fh.flush()
                    fh.close()
                except Exception as exc:
                    logger.warning(f"Error closing raw tactile CSV '{name}': {exc}")
            self._csv_files.clear()
            self._csv_writers.clear()

            for name, sensor in self._tactile_sensors.items():
                try:
                    if getattr(sensor, "is_connected", True):
                        sensor.disconnect()
                except Exception as exc:
                    logger.error(f"Error disconnecting tactile sensor '{name}': {exc}")
