"""Picklable fake FlexiTac hardware shared by thread- and process-backend tests.

Kept in its own top-level module so ``multiprocessing`` (spawn) workers can import it.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

SHAPE = (12, 32)
BASELINE = 20.0


def reference_raw() -> np.ndarray:
    """Deterministic raw ADC frame (uint8) used by every fake."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 200, size=SHAPE, dtype=np.uint8)


def reference_normalized() -> np.ndarray:
    return np.linspace(0.0, 1.0, SHAPE[0] * SHAPE[1], dtype=np.float32).reshape(SHAPE)


class FakeFlexiTacSensor:
    """Mimics the parts of ``flexitac.FlexiTacSensor`` the plugin uses."""

    def __init__(self, cfg=None, frame_period_s: float = 0.002):
        self.frame_period_s = frame_period_s
        self.opened = False
        self.closed = False
        self.fail_reads = False
        self.baseline = np.full(SHAPE, BASELINE, dtype=np.float32)
        self.raw = reference_raw()
        self.normalized = reference_normalized()
        self._seq = 0

    def open(self):
        self.opened = True
        return self

    def close(self) -> None:
        self.closed = True

    def read_latest(self) -> SimpleNamespace:
        if not self.opened:
            raise RuntimeError("serial port is not open")
        time.sleep(self.frame_period_s)  # real read_latest blocks until a frame arrives
        if self.fail_reads:
            raise TimeoutError("timed out waiting for a frame")
        frame = SimpleNamespace(
            seq=self._seq, timestamp_s=time.time(), raw=self.raw.copy(), normalized=self.normalized.copy()
        )
        self._seq += 1
        return frame


def make_fake_sensor(cfg) -> FakeFlexiTacSensor:
    """Module-level (picklable) factory for ``process_backend`` tests."""
    return FakeFlexiTacSensor(cfg)
