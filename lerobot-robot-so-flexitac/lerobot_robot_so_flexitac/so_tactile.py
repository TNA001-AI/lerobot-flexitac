"""SO-100/101/10X follower robot with lossless float32 FlexiTac tactile state features."""

from __future__ import annotations

import csv
import logging
import threading
import time
from collections.abc import Callable
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:  # LeRobot >= 0.4
    from lerobot.robots.so_follower.so_follower import SOFollower as _BaseSOFollower
except ImportError:  # LeRobot 0.3.x
    from lerobot.robots.so101_follower.so101_follower import SO101Follower as _BaseSOFollower

from lerobot_tactile_flexitac import FlexiTacTactile, FlexiTacTactileConfig

from .config_so_tactile import FlexiTacSensorEntryConfig, SOTactileFollowerConfig

logger = logging.getLogger(__name__)

_WARN_INTERVAL_S = 5.0
_TRANSIENT_ERRORS = (TimeoutError, RuntimeError, ConnectionError, OSError)


def flatten_tactile_names(sensor_name: str, shape: tuple[int, int] = (12, 32)) -> list[str]:
    """Feature names for one pad in C order, matching ``np.ravel()``.

    Format: ``tactile.<sensor>.r<row>_c<col>`` (zero-padded to 2 digits). Downstream tools
    (e.g. ``lerobot_policy_tactile.infer_tactile_layout``) rely on this exact format.
    """
    rows, cols = shape
    return [f"tactile.{sensor_name}.r{r:02d}_c{c:02d}" for r in range(rows) for c in range(cols)]


def unflatten_tactile_state(
    state_names: list[str],
    state_values: NDArray[Any],
    sensor_name: str = "primary",
    shape: tuple[int, int] = (12, 32),
) -> NDArray[np.float32]:
    """Select one sensor's columns from ``observation.state`` by name and reshape to ``(..., rows, cols)``."""
    expected = flatten_tactile_names(sensor_name, shape)
    index = {name: i for i, name in enumerate(state_names)}
    missing = [n for n in expected if n not in index]
    if missing:
        raise KeyError(f"Missing {len(missing)} tactile columns for sensor {sensor_name!r} (first: {missing[0]!r})")
    arr = np.asarray(state_values, dtype=np.float32)
    sliced = np.take(arr, [index[n] for n in expected], axis=-1)
    return sliced.reshape((*arr.shape[:-1], shape[0], shape[1]))


def _to_tactile_config(scfg: FlexiTacSensorEntryConfig) -> FlexiTacTactileConfig:
    return FlexiTacTactileConfig(
        port=scfg.port,
        baud_rate=scfg.baud_rate,
        tactile_shape=scfg.shape,
        baseline=scfg.baseline,
        init_frames=scfg.init_frames,
        threshold=scfg.threshold,
        noise_scale=scfg.noise_scale,
        read_timeout_s=scfg.read_timeout_s,
        signal=scfg.signal,
        render="raw",
        process_backend=scfg.process_backend,
    )


class _CsvSidecar:
    """Thread-safe frame listener writing every tactile frame to CSV."""

    def __init__(self, path: Path, names: list[str], flush_every: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh: Any = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(["epoch_ns", "seq", *names])
        self._lock = threading.Lock()
        self._rows = 0
        self._flush_every = flush_every

    def __call__(self, data: NDArray[np.float32], epoch_ns: int, seq: int) -> None:
        with self._lock:
            if self._fh is None:
                return
            self._writer.writerow([epoch_ns, seq, *data.ravel().tolist()])
            self._rows += 1
            if self._rows % self._flush_every == 0:
                self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
                self._fh = None


class SOTactileFollower(_BaseSOFollower):
    """SO-100 / SO-101 follower recording lossless float32 FlexiTac taxels in ``observation.state``."""

    config_class = SOTactileFollowerConfig
    name = "so_tactile_follower"

    # When True, `is_connected` reports only the arm. Used while tearing down so the parent
    # class' connection checks don't refuse to disconnect a half-connected robot.
    _arm_only_connection_check = False

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
        self._last_valid: dict[str, NDArray[np.float32]] = {}
        self._last_ok_mono: dict[str, float] = {}
        self._last_warn: dict[str, float] = {}
        self._sidecars: dict[str, _CsvSidecar] = {}

        for name, scfg in config.tactile_sensors.items():
            self._tactile_names[name] = flatten_tactile_names(name, scfg.shape)
            # Construction does not touch hardware, so `observation_features` works pre-connect.
            self._tactile_sensors[name] = (
                tactile_factory(name, scfg) if tactile_factory is not None else FlexiTacTactile(_to_tactile_config(scfg))
            )

    # ---- features ------------------------------------------------------------

    @property
    def _tactile_ft(self) -> dict[str, type]:
        return {col: float for names in self._tactile_names.values() for col in names}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        # Motors, then tactile floats (contiguous tail of observation.state), then cameras.
        return {**self._motors_ft, **self._tactile_ft, **self._cameras_ft}

    @property
    def is_connected(self) -> bool:
        arm = super().is_connected
        if self._arm_only_connection_check:
            return arm
        return arm and all(s.is_connected for s in self._tactile_sensors.values())

    # ---- lifecycle -------------------------------------------------------------

    def _open_sidecar(self, name: str, scfg: FlexiTacSensorEntryConfig, session: str) -> None:
        if not scfg.record_raw_csv:
            return
        sensor = self._tactile_sensors[name]
        if not hasattr(sensor, "add_frame_listener"):
            raise TypeError(f"Tactile sensor {name!r} does not support frame listeners (needed for record_raw_csv)")
        out_dir = (
            Path(scfg.raw_csv_dir).expanduser()
            if scfg.raw_csv_dir
            else Path.home() / ".cache" / "lerobot_flexitac" / "raw_sessions" / session
        )
        sidecar = _CsvSidecar(out_dir / f"flexitac_{name}.csv", self._tactile_names[name], scfg.csv_flush_every)
        sensor.add_frame_listener(sidecar)
        self._sidecars[name] = sidecar
        logger.info(f"Recording full-rate tactile sidecar for {name!r} to {sidecar.path}")

    def _close_sidecars(self) -> None:
        for name, sidecar in list(self._sidecars.items()):
            sensor = self._tactile_sensors.get(name)
            if sensor is not None and hasattr(sensor, "remove_frame_listener"):
                sensor.remove_frame_listener(sidecar)
            try:
                sidecar.close()
            except Exception as exc:
                logger.warning(f"Error closing tactile sidecar {name!r}: {exc!r}")
        self._sidecars.clear()

    def _disconnect_arm(self) -> None:
        if not getattr(self.bus, "is_connected", False):
            return
        self._arm_only_connection_check = True
        try:
            super().disconnect()
        finally:
            self._arm_only_connection_check = False

    def _disconnect_tactile(self) -> None:
        for name, sensor in self._tactile_sensors.items():
            try:
                if sensor.is_connected:
                    sensor.disconnect()
            except Exception as exc:
                logger.error(f"Error disconnecting tactile sensor {name!r}: {exc!r}")
        self._close_sidecars()

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate=calibrate)
        session = time.strftime("%Y%m%d_%H%M%S")
        try:
            for name, sensor in self._tactile_sensors.items():
                scfg = self.config.tactile_sensors[name]
                self._open_sidecar(name, scfg, session)
                sensor.connect(warmup=True)
                self._last_ok_mono[name] = time.monotonic()
                logger.info(f"Tactile sensor {name!r} connected ({len(self._tactile_names[name])} float features)")
        except BaseException:
            logger.error("Tactile sensor connection failed; rolling back arm and tactile connections")
            self._disconnect_tactile()
            try:
                self._disconnect_arm()
            except Exception as exc:
                logger.error(f"Error disconnecting arm during rollback: {exc!r}")
            raise

    def disconnect(self) -> None:
        try:
            self._disconnect_arm()
        finally:
            self._disconnect_tactile()

    # ---- observation -------------------------------------------------------------

    def _read_tactile(self, name: str, sensor: Any, scfg: FlexiTacSensorEntryConfig) -> NDArray[np.float32]:
        now = time.monotonic()
        try:
            frame = np.asarray(sensor.read_latest_tactile(max_age_ms=scfg.max_age_ms), dtype=np.float32)
        except _TRANSIENT_ERRORS as exc:
            stale_s = now - self._last_ok_mono.get(name, now)
            if stale_s > scfg.stale_timeout_s or name not in self._last_valid:
                raise RuntimeError(
                    f"Tactile sensor {name!r} has delivered no fresh frame for {stale_s:.2f}s "
                    f"(stale_timeout_s={scfg.stale_timeout_s}); refusing to record frozen tactile data. "
                    f"Last error: {exc!r}"
                ) from exc
            if now - self._last_warn.get(name, 0.0) > _WARN_INTERVAL_S:
                logger.warning(f"Tactile sensor {name!r} stale ({exc!r}); reusing last valid frame for {stale_s:.2f}s")
                self._last_warn[name] = now
            return self._last_valid[name]

        if frame.shape != scfg.shape:
            # Schema mismatch would silently misalign every state column: fail loudly.
            raise ValueError(f"Tactile sensor {name!r} returned shape {frame.shape}, expected {scfg.shape}")
        self._last_valid[name] = frame
        self._last_ok_mono[name] = now
        return frame

    def get_observation(self) -> dict[str, Any]:
        obs = super().get_observation()
        for name, sensor in self._tactile_sensors.items():
            frame = self._read_tactile(name, sensor, self.config.tactile_sensors[name])
            for col, val in zip(self._tactile_names[name], frame.ravel(), strict=True):
                obs[col] = float(val)
        return obs
