"""FlexiTac (12x32) tactile sensor implementation for LeRobot."""

from __future__ import annotations

import logging
import multiprocessing as mp
from multiprocessing import shared_memory
import threading
import time
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from lerobot.cameras.camera import Camera as _StreamSensor

from .config_flexitac_tactile import FlexiTacTactileConfig

logger = logging.getLogger(__name__)


def encode_tactile_map(
    tactile: NDArray[np.floating[Any]],
    *,
    render: str = "raw",
    width: int | None = None,
    height: int | None = None,
) -> NDArray[np.uint8]:
    """Encode a 2D normalized ``[0, 1]`` tactile array into an ``(H, W, 3)`` ``uint8`` image.

    - ``render="raw"``: Reversibly maps ``[0, 1]`` to ``[0, 255]`` replicated across 3 channels.
      Uses nearest-neighbor resize when ``(width, height)`` differs from ``(cols, rows)`` so
      block centers retain exact quantized values.
    - ``render="heatmap"``: Applies OpenCV Viridis colormap (converted to RGB).
    """
    arr = np.clip(np.asarray(tactile, dtype=np.float32), 0.0, 1.0)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D tactile map (rows, cols), got shape {arr.shape}")

    rows, cols = arr.shape
    target_w = cols if width is None else int(width)
    target_h = rows if height is None else int(height)

    scaled = np.rint(arr * 255.0).astype(np.uint8)

    if render == "raw":
        rgb = np.stack([scaled, scaled, scaled], axis=-1)
    elif render == "heatmap":
        import cv2

        bgr = cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unknown render mode {render!r}. Expected 'raw' or 'heatmap'.")

    if (target_w, target_h) != (cols, rows):
        import cv2

        rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    return rgb


def decode_tactile_image(
    image: NDArray[np.uint8],
    tactile_shape: tuple[int, int] = (12, 32),
) -> NDArray[np.float32]:
    """Invert ``encode_tactile_map(..., render='raw')`` back to a ``(rows, cols)`` ``float32`` map in ``[0, 1]``."""
    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) uint8 image, got shape {img.shape}")

    rows, cols = tactile_shape
    gray = img.mean(axis=2)
    h, w = gray.shape
    if (h, w) != (rows, cols):
        # Sample block centers corresponding to nearest-neighbor upscale
        row_idx = ((np.arange(rows) + 0.5) * h / rows).astype(int).clip(0, h - 1)
        col_idx = ((np.arange(cols) + 0.5) * w / cols).astype(int).clip(0, w - 1)
        gray = gray[np.ix_(row_idx, col_idx)]

    return (gray / 255.0).astype(np.float32)


def _flexitac_process_worker(
    cfg_dict: dict[str, Any],
    shm_name: str,
    meta_shm_name: str,
    ready_evt: Any,
    stop_evt: Any,
    error_queue: Any,
) -> None:
    """Isolated child-process serial reader writing latest float32 frame into shared memory."""
    shm = None
    meta_shm = None
    sensor = None
    try:
        from flexitac import FlexiTacSensor

        rows, cols = cfg_dict["tactile_shape"]
        shm = shared_memory.SharedMemory(name=shm_name)
        meta_shm = shared_memory.SharedMemory(name=meta_shm_name)
        buf = np.ndarray((rows, cols), dtype=np.float32, buffer=shm.buf)
        # meta: [frame_counter, monotonic_timestamp_s]
        meta = np.ndarray((2,), dtype=np.float64, buffer=meta_shm.buf)

        sensor = FlexiTacSensor(
            cfg_dict["port"],
            rows=rows,
            cols=cols,
            baud=cfg_dict["baud_rate"],
            threshold=cfg_dict["threshold"],
            noise_scale=cfg_dict["noise_scale"],
            init_frames=cfg_dict["init_frames"],
            baseline=cfg_dict["baseline"],
        )
        sensor.open()

        seq = 0.0
        while not stop_evt.is_set():
            try:
                frame = sensor.read_latest()
                data = np.asarray(frame.normalized, dtype=np.float32)
                buf[:] = data
                seq += 1.0
                meta[0] = seq
                meta[1] = time.monotonic()
                if not ready_evt.is_set():
                    ready_evt.set()
            except Exception:
                time.sleep(0.002)
    except Exception as exc:
        try:
            error_queue.put_nowait(repr(exc))
        except Exception:
            pass
    finally:
        if sensor is not None:
            try:
                sensor.close()
            except Exception:
                pass
        if shm is not None:
            shm.close()
        if meta_shm is not None:
            meta_shm.close()


class FlexiTacTactile(_StreamSensor):
    """LeRobot tactile stream device for FlexiTac (12x32) tactile pads.

    Supports both:
    1. Direct lossless ``float32`` tactile matrix reads via :meth:`read_latest_tactile`.
    2. Standard ``(H, W, 3)`` ``uint8`` stream reads (:meth:`read`, :meth:`async_read`,
       :meth:`read_latest`) for zero-change compatibility with ``lerobot-record`` and Rerun.
    """

    def __init__(
        self,
        config: FlexiTacTactileConfig,
        *,
        sensor_factory: Callable[..., Any] | None = None,
    ):
        super().__init__(config)
        self.config: FlexiTacTactileConfig = config
        self.rows, self.cols = config.tactile_shape
        self._sensor_factory = sensor_factory

        if self._sensor_factory is None and not config.process_backend:
            try:
                import flexitac  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "The 'flexitac' driver package is required to use FlexiTacTactile. "
                    "Install it with: pip install flexitac"
                ) from e

        self._sensor: Any = None
        self._connected = False

        # Threading & fault-tolerant frame cache
        self._lock = threading.Lock()
        self._new_frame_evt = threading.Event()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_valid_tactile: NDArray[np.float32] | None = None
        self._prev_filtered: NDArray[np.float32] | None = None
        self._latest_ts: float = 0.0

        # Multiprocessing + shared memory handles
        self._shm: shared_memory.SharedMemory | None = None
        self._meta_shm: shared_memory.SharedMemory | None = None
        self._proc: mp.Process | None = None
        self._proc_stop_evt: Any = None
        self._last_consumed_seq: float = 0.0

    def __str__(self) -> str:
        return f"FlexiTacTactile(port={self.config.port!r}, shape={self.config.tactile_shape})"

    @property
    def is_connected(self) -> bool:
        return self._connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """List available serial ports that may host FlexiTac boards."""
        try:
            from serial.tools import list_ports
        except ImportError:
            return []
        return [
            {"type": "flexitac", "port": p.device, "description": p.description}
            for p in list_ports.comports()
        ]

    # ---- Lossless float32 tactile API ------------------------------------

    def read_latest_tactile(self, max_age_ms: int | None = None) -> NDArray[np.float32]:
        """Return the freshest ``(rows, cols)`` ``float32`` tactile map.

        On transient read glitches, returns the last valid frame instead of zeros so
        downstream policies do not mistake a dropped packet for contact release.
        """
        if not self._connected:
            raise ConnectionError(f"{self} is not connected.")

        if self.config.process_backend and self._shm is not None and self._meta_shm is not None:
            meta = np.ndarray((2,), dtype=np.float64, buffer=self._meta_shm.buf)
            seq, ts = float(meta[0]), float(meta[1])
            if seq == 0.0:
                return np.zeros((self.rows, self.cols), dtype=np.float32)
            if max_age_ms is not None and (time.monotonic() - ts) * 1000.0 > max_age_ms:
                raise TimeoutError(f"{self}: latest shared-memory tactile frame exceeded {max_age_ms}ms")
            buf = np.ndarray((self.rows, self.cols), dtype=np.float32, buffer=self._shm.buf)
            frame = buf.copy()
            self._last_valid_tactile = frame
            return frame

        with self._lock:
            if self._last_valid_tactile is None:
                return np.zeros((self.rows, self.cols), dtype=np.float32)
            if max_age_ms is not None and self._latest_ts > 0.0:
                age_ms = (time.monotonic() - self._latest_ts) * 1000.0
                if age_ms > max_age_ms:
                    raise TimeoutError(f"{self}: latest tactile frame is stale ({age_ms:.1f}ms > {max_age_ms}ms)")
            return self._last_valid_tactile.copy()

    # ---- Rendering & background reader -----------------------------------

    def _render_frame(self, raw: NDArray[np.float32]) -> NDArray[np.uint8]:
        if self.config.render == "heatmap":
            alpha = self.config.temporal_alpha
            if self._prev_filtered is None:
                self._prev_filtered = raw.copy()
            else:
                self._prev_filtered = alpha * raw + (1.0 - alpha) * self._prev_filtered
            to_encode = self._prev_filtered
        else:
            # Keep "raw" free of temporal blending unless alpha < 1 is desired?
            # Preserve exact single-frame taxel values in "raw" mode for invertibility.
            to_encode = raw

        return encode_tactile_map(
            to_encode,
            render=self.config.render,
            width=self.width,
            height=self.height,
        )

    def _poll_once(self) -> NDArray[np.float32]:
        frame = self._sensor.read_latest()
        data = np.asarray(frame.normalized, dtype=np.float32)
        if data.shape != (self.rows, self.cols):
            raise ValueError(
                f"{self}: expected tactile shape {(self.rows, self.cols)}, got {data.shape}"
            )
        return data

    def _thread_read_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                data = self._poll_once()
                with self._lock:
                    self._last_valid_tactile = data
                    self._latest_ts = time.monotonic()
                self._new_frame_evt.set()
            except ValueError:
                logger.exception(f"{self}: fatal tactile shape mismatch; stopping reader")
                break
            except Exception as exc:
                logger.warning(f"{self}: transient read error ({exc}); retaining last valid frame")
                time.sleep(0.005)

    # ---- Lifecycle & Camera-compatible stream API ------------------------

    def connect(self, warmup: bool = True) -> None:
        if self._connected:
            raise RuntimeError(f"{self} is already connected.")

        if self.config.process_backend:
            self._connect_process_backend(warmup=warmup)
            return

        if self._sensor_factory is not None:
            self._sensor = self._sensor_factory(self.config)
        else:
            from flexitac import FlexiTacSensor

            self._sensor = FlexiTacSensor(
                self.config.port,
                rows=self.rows,
                cols=self.cols,
                baud=self.config.baud_rate,
                threshold=self.config.threshold,
                noise_scale=self.config.noise_scale,
                init_frames=self.config.init_frames,
                baseline=self.config.baseline,
            )

        self._sensor.open()
        self._connected = True
        self._stop_evt.clear()
        self._new_frame_evt.clear()
        self._thread = threading.Thread(target=self._thread_read_loop, daemon=True)
        self._thread.start()

        if warmup:
            if not self._new_frame_evt.wait(timeout=5.0):
                self.disconnect()
                raise TimeoutError(f"Timed out waiting for warmup frame from {self}")
        logger.info(f"{self} connected (thread backend, render={self.config.render}).")

    def _connect_process_backend(self, warmup: bool = True) -> None:
        ctx = mp.get_context("spawn")
        nbytes = self.rows * self.cols * np.dtype(np.float32).itemsize
        self._shm = shared_memory.SharedMemory(create=True, size=nbytes)
        self._meta_shm = shared_memory.SharedMemory(
            create=True, size=2 * np.dtype(np.float64).itemsize
        )
        np.ndarray((2,), dtype=np.float64, buffer=self._meta_shm.buf)[:] = 0.0

        ready_evt = ctx.Event()
        self._proc_stop_evt = ctx.Event()
        err_q = ctx.Queue()
        cfg_dict = {
            "port": self.config.port,
            "baud_rate": self.config.baud_rate,
            "tactile_shape": self.config.tactile_shape,
            "baseline": self.config.baseline,
            "init_frames": self.config.init_frames,
            "threshold": self.config.threshold,
            "noise_scale": self.config.noise_scale,
        }
        self._proc = ctx.Process(
            target=_flexitac_process_worker,
            args=(cfg_dict, self._shm.name, self._meta_shm.name, ready_evt, self._proc_stop_evt, err_q),
            daemon=True,
        )
        self._proc.start()
        self._connected = True

        if warmup:
            if not ready_evt.wait(timeout=self.config.process_connect_timeout_s):
                worker_err = None
                try:
                    worker_err = err_q.get_nowait()
                except Exception:
                    pass
                self.disconnect()
                detail = f" (worker error: {worker_err})" if worker_err else ""
                raise TimeoutError(
                    f"Timed out waiting for process backend on {self}{detail}"
                )
        logger.info(f"{self} connected (process backend, render={self.config.render}).")

    def read(self) -> NDArray[np.uint8]:
        if not self._connected:
            raise ConnectionError(f"{self} is not connected.")
        if self.config.process_backend:
            return self._render_frame(self.read_latest_tactile())

        try:
            data = self._poll_once()
            with self._lock:
                self._last_valid_tactile = data
                self._latest_ts = time.monotonic()
        except Exception as exc:
            logger.warning(f"{self}: synchronous read fallback to last valid frame ({exc})")
            data = self.read_latest_tactile()
        return self._render_frame(data)

    def async_read(self, timeout_ms: float = 200) -> NDArray[np.uint8]:
        if not self._connected:
            raise ConnectionError(f"{self} is not connected.")

        if self.config.process_backend and self._meta_shm is not None:
            deadline = time.monotonic() + timeout_ms / 1000.0
            meta = np.ndarray((2,), dtype=np.float64, buffer=self._meta_shm.buf)
            while float(meta[0]) <= self._last_consumed_seq:
                if time.monotonic() >= deadline:
                    if self._last_valid_tactile is not None:
                        return self._render_frame(self._last_valid_tactile)
                    raise TimeoutError(f"{self}: no new frame within {timeout_ms}ms")
                time.sleep(0.001)
            self._last_consumed_seq = float(meta[0])
            return self._render_frame(self.read_latest_tactile())

        if not self._new_frame_evt.wait(timeout=timeout_ms / 1000.0):
            with self._lock:
                if self._last_valid_tactile is None:
                    raise TimeoutError(f"{self}: no tactile frame received within {timeout_ms}ms")
                fallback = self._last_valid_tactile.copy()
            return self._render_frame(fallback)

        self._new_frame_evt.clear()
        return self._render_frame(self.read_latest_tactile())

    def read_latest(self, max_age_ms: int = 500) -> NDArray[np.uint8]:
        raw = self.read_latest_tactile(max_age_ms=max_age_ms)
        return self._render_frame(raw)

    def disconnect(self) -> None:
        if self.config.process_backend:
            if self._proc_stop_evt is not None:
                self._proc_stop_evt.set()
            if self._proc is not None and self._proc.is_alive():
                self._proc.join(timeout=2.0)
                if self._proc.is_alive():
                    self._proc.terminate()
            for shm_obj in (self._shm, self._meta_shm):
                if shm_obj is not None:
                    try:
                        shm_obj.close()
                    except Exception:
                        pass
                    try:
                        shm_obj.unlink()
                    except Exception:
                        pass
            self._shm = None
            self._meta_shm = None
            self._proc = None
            self._connected = False
            logger.info(f"{self} disconnected (process backend).")
            return

        self._stop_evt.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._sensor is not None:
            try:
                self._sensor.close()
            except Exception as exc:
                logger.warning(f"{self}: error closing serial sensor: {exc}")
            self._sensor = None
        self._connected = False
        logger.info(f"{self} disconnected.")
