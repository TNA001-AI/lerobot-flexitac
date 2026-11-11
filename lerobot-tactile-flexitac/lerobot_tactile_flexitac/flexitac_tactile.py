"""FlexiTac (12x32) tactile sensor implementation for LeRobot.

Threading model: exactly one reader owns the serial port. In the default backend this is a
daemon thread; with ``process_backend=True`` it is a spawned child process that writes into
shared memory, mirrored into the parent by a light sync thread. Every public read API is
served from the latest ingested frame, so callers never touch the serial port concurrently.
"""

from __future__ import annotations

import importlib.util
import logging
import multiprocessing as mp
import queue
import threading
import time
from collections.abc import Callable
from multiprocessing import shared_memory
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lerobot.cameras.camera import Camera as _StreamSensor

from .config_flexitac_tactile import FlexiTacTactileConfig

logger = logging.getLogger(__name__)

# (tactile_map, epoch_ns, seq) -> None. Called from the reader / sync thread for every frame.
FrameListener = Callable[[NDArray[np.float32], int, int], None]

_WARN_INTERVAL_S = 5.0


class TactileShapeError(ValueError):
    """Raised when the sensor delivers a frame whose shape differs from ``tactile_shape``."""


def make_flexitac_sensor(cfg: Any) -> Any:
    """Build a ``flexitac.FlexiTacSensor`` (not opened) from a config-like object.

    Module-level so it can be pickled into the ``process_backend`` worker.
    """
    from flexitac import FlexiTacSensor

    rows, cols = cfg.tactile_shape
    return FlexiTacSensor(
        cfg.port,
        rows=rows,
        cols=cols,
        baud=cfg.baud_rate,
        threshold=cfg.threshold,
        noise_scale=cfg.noise_scale,
        init_frames=cfg.init_frames,
        read_timeout_s=cfg.read_timeout_s,
        baseline=cfg.baseline,
    )


def frame_to_tactile_map(frame: Any, sensor: Any, cfg: Any) -> NDArray[np.float32]:
    """Convert a ``flexitac.FlexiTacFrame`` to a ``(rows, cols)`` float32 map in ``[0, 1]``."""
    if cfg.signal == "baseline_subtracted":
        baseline = getattr(sensor, "baseline", None)
        if baseline is None:
            raise RuntimeError("signal='baseline_subtracted' requires a calibrated sensor baseline")
        signal = np.asarray(frame.raw, dtype=np.float32) - np.asarray(baseline, dtype=np.float32)
        # Same clipping range flexitac uses internally (0..100 ADC counts above threshold).
        data = np.clip(signal - float(cfg.threshold), 0.0, 100.0) / 100.0
    else:
        data = np.asarray(frame.normalized, dtype=np.float32)

    data = np.ascontiguousarray(data, dtype=np.float32)
    if data.shape != tuple(cfg.tactile_shape):
        raise TactileShapeError(f"expected tactile shape {tuple(cfg.tactile_shape)}, got {data.shape}")
    return data


def encode_tactile_map(
    tactile: NDArray[np.floating[Any]],
    *,
    render: str = "raw",
    width: int | None = None,
    height: int | None = None,
) -> NDArray[np.uint8]:
    """Encode a 2D ``[0, 1]`` tactile map into an ``(H, W, 3)`` ``uint8`` image.

    - ``render="raw"``: reversible, ``[0, 1] -> [0, 255]`` replicated across 3 channels.
      Resizing uses nearest-neighbor so block centers keep the exact quantized value.
    - ``render="heatmap"``: OpenCV Viridis colormap converted to RGB (not invertible).
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

        rgb = cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
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
    """Invert ``encode_tactile_map(..., render='raw')`` back to ``(rows, cols)`` float32."""
    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) uint8 image, got shape {img.shape}")

    rows, cols = tactile_shape
    gray = img.astype(np.float32).mean(axis=2)
    h, w = gray.shape
    if (h, w) != (rows, cols):
        row_idx = ((np.arange(rows) + 0.5) * h / rows).astype(int).clip(0, h - 1)
        col_idx = ((np.arange(cols) + 0.5) * w / cols).astype(int).clip(0, w - 1)
        gray = gray[np.ix_(row_idx, col_idx)]
    return (gray / 255.0).astype(np.float32)


def _process_worker(
    sensor_factory: Callable[[Any], Any],
    cfg_dict: dict[str, Any],
    shm_name: str,
    meta_name: str,
    lock: Any,
    ready_evt: Any,
    stop_evt: Any,
    err_q: Any,
) -> None:
    """Child process: own the serial port and publish frames into shared memory.

    Shared layout: ``buf`` float32 ``(rows, cols)``; ``meta`` int64 ``[seq, monotonic_ns,
    epoch_ns]``. Both are only touched while holding ``lock`` so reads are never torn.
    """
    cfg = SimpleNamespace(**cfg_dict)
    rows, cols = cfg.tactile_shape
    shm = meta_shm = sensor = None

    def _report(msg: str) -> None:
        try:
            err_q.put_nowait(msg)
        except Exception:  # nosec B110 - best-effort error reporting
            pass

    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        meta_shm = shared_memory.SharedMemory(name=meta_name)
        buf = np.ndarray((rows, cols), dtype=np.float32, buffer=shm.buf)
        meta = np.ndarray((3,), dtype=np.int64, buffer=meta_shm.buf)

        sensor = sensor_factory(cfg)
        sensor.open()
        seq = 0
        last_report = 0.0
        while not stop_evt.is_set():
            try:
                data = frame_to_tactile_map(sensor.read_latest(), sensor, cfg)
            except TactileShapeError:
                raise
            except Exception as exc:
                now = time.monotonic()
                if now - last_report > _WARN_INTERVAL_S:
                    _report(f"transient: {exc!r}")
                    last_report = now
                time.sleep(0.05)
                continue
            seq += 1
            with lock:
                buf[:] = data
                meta[0] = seq
                meta[1] = time.monotonic_ns()
                meta[2] = time.time_ns()
            ready_evt.set()
    except Exception as exc:
        _report(f"fatal: {exc!r}")
    finally:
        if sensor is not None:
            try:
                sensor.close()
            except Exception:  # nosec B110
                pass
        for s in (shm, meta_shm):
            if s is not None:
                s.close()


class FlexiTacTactile(_StreamSensor):
    """LeRobot tactile stream device for FlexiTac (12x32) tactile pads.

    Two access paths:
    1. :meth:`read_latest_tactile` returns the exact ``(rows, cols)`` float32 map (used by
       ``lerobot_robot_so_flexitac``).
    2. The standard stream API (:meth:`read`, :meth:`async_read`, :meth:`read_latest`)
       returns ``(H, W, 3)`` uint8 so stock ``lerobot-record`` and Rerun work unchanged.

    Staleness is never hidden: if no fresh frame arrives, :meth:`read_latest` /
    :meth:`read_latest_tactile` raise ``TimeoutError`` once ``max_age_ms`` is exceeded, and
    :meth:`async_read` raises ``TimeoutError`` instead of repeating an old frame.
    """

    def __init__(
        self,
        config: FlexiTacTactileConfig,
        *,
        sensor_factory: Callable[[Any], Any] | None = None,
    ):
        self._connected = False
        super().__init__(config)
        self.config: FlexiTacTactileConfig = config
        self.rows, self.cols = config.tactile_shape

        if sensor_factory is None and importlib.util.find_spec("flexitac") is None:
            raise ImportError(
                "The 'flexitac' driver package is required to use FlexiTacTactile. "
                "Install it with: pip install flexitac"
            )
        self._sensor_factory: Callable[[Any], Any] = sensor_factory or make_flexitac_sensor

        self._sensor: Any = None
        self._cond = threading.Condition()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._listeners: list[FrameListener] = []
        self.last_error: str | None = None
        self._last_warn = 0.0
        self._reset_frame_state()

        # process backend handles
        self._shm: shared_memory.SharedMemory | None = None
        self._meta_shm: shared_memory.SharedMemory | None = None
        self._shm_buf: NDArray[np.float32] | None = None
        self._shm_meta: NDArray[np.int64] | None = None
        self._shm_seen_seq = 0
        self._proc: Any = None
        self._proc_stop: Any = None
        self._proc_lock: Any = None
        self._proc_ready: Any = None
        self._err_q: Any = None
        self._worker_dead_logged = False

    def __str__(self) -> str:
        return f"FlexiTacTactile(port={self.config.port!r}, shape={self.config.tactile_shape})"

    # ---- state -------------------------------------------------------------

    def _reset_frame_state(self) -> None:
        with self._cond:
            self._latest: NDArray[np.float32] | None = None
            self._filtered: NDArray[np.float32] | None = None
            self._latest_mono = 0.0
            self._seq = 0
            self._consumed_seq = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def frame_count(self) -> int:
        """Number of frames ingested since the last ``connect()``."""
        return self._seq

    def latest_frame_age_ms(self) -> float | None:
        with self._cond:
            if self._latest is None:
                return None
            return (time.monotonic() - self._latest_mono) * 1000.0

    def add_frame_listener(self, listener: FrameListener) -> None:
        """Register ``listener(tactile_map, epoch_ns, seq)`` called for every ingested frame."""
        with self._cond:
            self._listeners.append(listener)

    def remove_frame_listener(self, listener: FrameListener) -> None:
        with self._cond:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def _ingest(self, data: NDArray[np.float32], mono_s: float, epoch_ns: int) -> None:
        alpha = self.config.temporal_alpha
        with self._cond:
            self._latest = data
            # EMA advances once per sensor frame, independent of how often callers read.
            self._filtered = data.copy() if self._filtered is None else alpha * data + (1.0 - alpha) * self._filtered
            self._latest_mono = mono_s
            self._seq += 1
            seq = self._seq
            listeners = list(self._listeners)
            self._cond.notify_all()
        for listener in listeners:
            try:
                listener(data, epoch_ns, seq)
            except Exception:
                logger.exception(f"{self}: frame listener raised")

    def _warn_throttled(self, msg: str) -> None:
        now = time.monotonic()
        if now - self._last_warn > _WARN_INTERVAL_S:
            logger.warning(msg)
            self._last_warn = now

    # ---- discovery -----------------------------------------------------------

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """List serial ports that could host a FlexiTac board.

        FlexiTac boards expose no identifying USB descriptor, so this returns every serial
        port (including robot arm buses). Use ``flexitac-find-port`` (unplug/replug) to
        identify the right one.
        """
        try:
            from serial.tools import list_ports
        except ImportError:
            return []
        return [
            {
                "type": "flexitac",
                "port": p.device,
                "description": p.description,
                "hwid": p.hwid,
                "vid": p.vid,
                "pid": p.pid,
            }
            for p in list_ports.comports()
        ]

    # ---- reader loops --------------------------------------------------------

    def _thread_read_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                data = frame_to_tactile_map(self._sensor.read_latest(), self._sensor, self.config)
            except TactileShapeError as exc:
                self.last_error = repr(exc)
                logger.error(f"{self}: {exc}; stopping reader (schema mismatch is not recoverable)")
                return
            except Exception as exc:
                self.last_error = repr(exc)
                self._warn_throttled(f"{self}: transient read error: {exc!r}")
                time.sleep(0.05)
                continue
            self._ingest(data, time.monotonic(), time.time_ns())

    def _drain_worker_errors(self) -> None:
        if self._err_q is None:
            return
        while True:
            try:
                msg = self._err_q.get_nowait()
            except (queue.Empty, OSError, ValueError, EOFError):
                return
            self.last_error = msg
            self._warn_throttled(f"{self}: worker reported {msg}")

    def _sync_from_shm(self) -> None:
        if self._shm_meta is None or self._shm_buf is None:
            return
        with self._proc_lock:
            seq = int(self._shm_meta[0])
            if seq <= self._shm_seen_seq:
                return
            data = self._shm_buf.copy()
            mono_ns = int(self._shm_meta[1])
            epoch_ns = int(self._shm_meta[2])
        self._shm_seen_seq = seq
        self._ingest(data, mono_ns / 1e9, epoch_ns)

    def _process_sync_loop(self) -> None:
        while not self._stop_evt.is_set():
            self._drain_worker_errors()
            self._sync_from_shm()
            if self._proc is not None and not self._proc.is_alive() and not self._worker_dead_logged:
                self._worker_dead_logged = True
                logger.error(f"{self}: reader process exited (last error: {self.last_error})")
            time.sleep(0.001)

    # ---- lifecycle -------------------------------------------------------------

    def connect(self, warmup: bool = True) -> None:
        if self._connected:
            raise RuntimeError(f"{self} is already connected.")
        self._reset_frame_state()
        self.last_error = None
        self._stop_evt.clear()

        try:
            if self.config.process_backend:
                self._start_process_backend()
                timeout_s = self.config.process_connect_timeout_s
            else:
                self._sensor = self._sensor_factory(self.config)
                self._sensor.open()
                self._thread = threading.Thread(target=self._thread_read_loop, name=f"{self}-reader", daemon=True)
                self._thread.start()
                timeout_s = self.config.warmup_timeout_s
            self._connected = True

            if warmup and not self._wait_for_frame_after(0, timeout_s):
                raise TimeoutError(
                    f"Timed out after {timeout_s:.1f}s waiting for the first frame from {self} "
                    f"(last error: {self.last_error})"
                )
        except BaseException:
            self.disconnect()
            raise
        logger.info(
            f"{self} connected ({'process' if self.config.process_backend else 'thread'} backend, "
            f"signal={self.config.signal}, render={self.config.render})."
        )

    def _start_process_backend(self) -> None:
        ctx = mp.get_context("spawn")
        self._shm = shared_memory.SharedMemory(create=True, size=self.rows * self.cols * 4)
        self._meta_shm = shared_memory.SharedMemory(create=True, size=3 * 8)
        self._shm_buf = np.ndarray((self.rows, self.cols), dtype=np.float32, buffer=self._shm.buf)
        self._shm_meta = np.ndarray((3,), dtype=np.int64, buffer=self._meta_shm.buf)
        self._shm_meta[:] = 0
        self._shm_seen_seq = 0
        self._worker_dead_logged = False

        # Keep references to every sync primitive passed to the child: if one is garbage
        # collected before the spawned child unpickles it, its semaphore is unlinked.
        self._proc_lock = ctx.Lock()
        self._proc_stop = ctx.Event()
        self._proc_ready = ctx.Event()
        self._err_q = ctx.Queue()
        cfg_dict = {
            name: getattr(self.config, name)
            for name in (
                "port",
                "baud_rate",
                "tactile_shape",
                "baseline",
                "init_frames",
                "threshold",
                "noise_scale",
                "read_timeout_s",
                "signal",
            )
        }
        self._proc = ctx.Process(
            target=_process_worker,
            args=(
                self._sensor_factory,
                cfg_dict,
                self._shm.name,
                self._meta_shm.name,
                self._proc_lock,
                self._proc_ready,
                self._proc_stop,
                self._err_q,
            ),
            name=f"{self}-worker",
            daemon=True,
        )
        self._proc.start()
        self._thread = threading.Thread(target=self._process_sync_loop, name=f"{self}-sync", daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        self._stop_evt.set()
        if self._proc_stop is not None:
            self._proc_stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        if self._proc is not None:
            self._proc.join(timeout=3.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=1.0)
        self._proc = None

        if self._sensor is not None:
            try:
                self._sensor.close()
            except Exception as exc:
                logger.warning(f"{self}: error closing serial port: {exc!r}")
            self._sensor = None

        self._shm_buf = None
        self._shm_meta = None
        for shm_obj in (self._shm, self._meta_shm):
            if shm_obj is not None:
                try:
                    shm_obj.close()
                    shm_obj.unlink()
                except (FileNotFoundError, BufferError, OSError) as exc:
                    logger.debug(f"{self}: shared memory cleanup: {exc!r}")
        self._shm = self._meta_shm = None
        if self._err_q is not None:
            self._err_q.close()
        self._err_q = self._proc_stop = self._proc_lock = self._proc_ready = None

        was_connected = self._connected
        self._connected = False
        if was_connected:
            logger.info(f"{self} disconnected.")

    # ---- read API ----------------------------------------------------------------

    def _check_connected(self) -> None:
        if not self._connected:
            raise ConnectionError(f"{self} is not connected.")

    def _wait_for_frame_after(self, seq: int, timeout_s: float) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: self._seq > seq, timeout=timeout_s)

    def _latest_locked(self, max_age_ms: float | None) -> NDArray[np.float32]:
        if self._latest is None:
            raise RuntimeError(f"{self} has not received any frame yet (last error: {self.last_error})")
        if max_age_ms is not None:
            age_ms = (time.monotonic() - self._latest_mono) * 1000.0
            if age_ms > max_age_ms:
                raise TimeoutError(
                    f"{self}: latest frame is {age_ms:.0f}ms old (> {max_age_ms}ms, last error: {self.last_error})"
                )
        return self._latest

    def _render_locked(self) -> NDArray[np.uint8]:
        source = self._filtered if self.config.render == "heatmap" else self._latest
        assert source is not None
        return encode_tactile_map(source, render=self.config.render, width=self.width, height=self.height)

    def read_latest_tactile(self, max_age_ms: float | None = None) -> NDArray[np.float32]:
        """Return a copy of the latest ``(rows, cols)`` float32 map.

        Raises ``TimeoutError`` if it is older than ``max_age_ms`` and ``RuntimeError`` if no
        frame has arrived yet. Pass ``max_age_ms`` in control loops so a dead sensor is
        detected instead of silently repeating the last frame.
        """
        self._check_connected()
        with self._cond:
            return self._latest_locked(max_age_ms).copy()

    def read(self) -> NDArray[np.uint8]:
        """Block until the next new frame arrives and return it rendered."""
        self._check_connected()
        with self._cond:
            start = self._seq
        timeout_s = self.config.read_timeout_s + 0.5
        if not self._wait_for_frame_after(start, timeout_s):
            raise TimeoutError(f"{self}: no new frame within {timeout_s:.1f}s (last error: {self.last_error})")
        with self._cond:
            return self._render_locked()

    def async_read(self, timeout_ms: float = 200) -> NDArray[np.uint8]:
        """Return the latest *unconsumed* frame, waiting up to ``timeout_ms`` for one."""
        self._check_connected()
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > self._consumed_seq, timeout=timeout_ms / 1000.0):
                raise TimeoutError(f"{self}: no new frame within {timeout_ms}ms (last error: {self.last_error})")
            self._consumed_seq = self._seq
            return self._render_locked()

    def read_latest(self, max_age_ms: int = 500) -> NDArray[np.uint8]:
        """Return the most recent frame rendered, without waiting (peek)."""
        self._check_connected()
        with self._cond:
            self._latest_locked(max_age_ms)
            return self._render_locked()
