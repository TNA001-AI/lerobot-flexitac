"""Hardware-free unit tests for lerobot_tactile_flexitac."""

from __future__ import annotations

from types import SimpleNamespace
import time

import numpy as np
import pytest

from lerobot.cameras.configs import CameraConfig
from lerobot_tactile_flexitac import (
    FlexiTacTactile,
    FlexiTacTactileConfig,
    decode_tactile_image,
    encode_tactile_map,
)


class _FakeFlexiTacHardware:
    """Simulated FlexiTac serial device with injectable read failures."""

    def __init__(self, shape: tuple[int, int] = (12, 32)):
        self.shape = shape
        self.opened = False
        self.closed = False
        self.fail_reads = False
        self.current_map = np.linspace(0.0, 1.0, shape[0] * shape[1], dtype=np.float32).reshape(shape)

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def read_latest(self) -> SimpleNamespace:
        if self.fail_reads:
            raise OSError("Simulated transient USB serial glitch")
        time.sleep(0.002)
        return SimpleNamespace(normalized=self.current_map.copy())


def test_config_registration_and_defaults():
    assert CameraConfig.get_choice_class("flexitac") is FlexiTacTactileConfig

    cfg_raw = FlexiTacTactileConfig(port="/dev/ttyUSB9", render="raw")
    assert (cfg_raw.height, cfg_raw.width) == (12, 32)
    assert cfg_raw.fps == 30

    cfg_hm = FlexiTacTactileConfig(port="/dev/ttyUSB9", render="heatmap")
    assert (cfg_hm.height, cfg_hm.width) == (96, 256)

    # Verify LeRobot's Config -> Device class name stripping rule:
    assert FlexiTacTactileConfig.__name__.removesuffix("Config") == FlexiTacTactile.__name__


def test_raw_encoding_is_invertible_at_native_and_upscaled_resolutions():
    rng = np.random.default_rng(42)
    original = rng.uniform(0.0, 1.0, size=(12, 32)).astype(np.float32)

    # Native 12x32
    img_native = encode_tactile_map(original, render="raw")
    assert img_native.shape == (12, 32, 3)
    assert img_native.dtype == np.uint8
    recovered_native = decode_tactile_image(img_native, tactile_shape=(12, 32))
    # Max quantization error is 0.5 / 255 ~= 0.002
    np.testing.assert_allclose(recovered_native, original, atol=1.0 / 255.0 + 1e-5)

    # Upscaled 96x256 via nearest-neighbor
    img_upscaled = encode_tactile_map(original, render="raw", width=256, height=96)
    assert img_upscaled.shape == (96, 256, 3)
    recovered_upscaled = decode_tactile_image(img_upscaled, tactile_shape=(12, 32))
    np.testing.assert_allclose(recovered_upscaled, recovered_native, atol=1e-6)


def test_heatmap_encoding_shape_and_contrast():
    zeros = np.zeros((12, 32), dtype=np.float32)
    img_zero = encode_tactile_map(zeros, render="heatmap", width=256, height=96)
    assert img_zero.shape == (96, 256, 3)
    assert img_zero.dtype == np.uint8

    peak = zeros.copy()
    peak[6, 16] = 1.0
    img_peak = encode_tactile_map(peak, render="heatmap", width=256, height=96)
    assert not np.array_equal(img_zero, img_peak)


def test_lifecycle_and_last_valid_frame_fault_tolerance():
    hw = _FakeFlexiTacHardware(shape=(12, 32))
    cfg = FlexiTacTactileConfig(port="/dev/null", render="raw")
    sensor = FlexiTacTactile(cfg, sensor_factory=lambda _c: hw)

    sensor.connect(warmup=True)
    assert sensor.is_connected
    assert hw.opened

    # Normal read
    tactile_1 = sensor.read_latest_tactile()
    np.testing.assert_allclose(tactile_1, hw.current_map)

    frame_rgb = sensor.async_read(timeout_ms=200)
    assert frame_rgb.shape == (12, 32, 3)

    # Inject transient serial failure: must retain last valid frame, NOT zeros!
    hw.fail_reads = True
    time.sleep(0.02)
    tactile_during_glitch = sensor.read_latest_tactile()
    assert np.count_nonzero(tactile_during_glitch) > 0
    np.testing.assert_allclose(tactile_during_glitch, tactile_1)

    sensor.disconnect()
    assert not sensor.is_connected
    assert hw.closed

    with pytest.raises(ConnectionError):
        sensor.read_latest_tactile()
