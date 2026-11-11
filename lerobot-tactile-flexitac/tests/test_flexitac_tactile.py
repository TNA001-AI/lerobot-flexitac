"""Hardware-free tests for lerobot_tactile_flexitac."""

from __future__ import annotations

import inspect
import time

import numpy as np
import pytest
from _fake_flexitac import BASELINE, FakeFlexiTacSensor, make_fake_sensor, reference_normalized, reference_raw

from lerobot.cameras.configs import CameraConfig
from lerobot_tactile_flexitac import (
    FlexiTacTactile,
    FlexiTacTactileConfig,
    decode_tactile_image,
    encode_tactile_map,
    frame_to_tactile_map,
)


def _connected_sensor(**cfg_kwargs):
    hw = FakeFlexiTacSensor()
    cfg = FlexiTacTactileConfig(port="/dev/null", **cfg_kwargs)
    sensor = FlexiTacTactile(cfg, sensor_factory=lambda _c: hw)
    sensor.connect(warmup=True)
    return sensor, hw


def test_config_registration_and_defaults():
    assert CameraConfig.get_choice_class("flexitac") is FlexiTacTactileConfig
    # LeRobot resolves the device class by stripping "Config".
    assert FlexiTacTactileConfig.__name__.removesuffix("Config") == FlexiTacTactile.__name__

    cfg_raw = FlexiTacTactileConfig(port="/dev/ttyUSB9")
    assert (cfg_raw.height, cfg_raw.width, cfg_raw.fps) == (12, 32, 30)
    cfg_hm = FlexiTacTactileConfig(port="/dev/ttyUSB9", render="heatmap")
    assert (cfg_hm.height, cfg_hm.width) == (96, 256)

    with pytest.raises(ValueError):
        FlexiTacTactileConfig(port="/dev/ttyUSB9", signal="bogus")
    with pytest.raises(ValueError):
        FlexiTacTactileConfig(port="/dev/ttyUSB9", temporal_alpha=0.0)


def test_raw_encoding_is_invertible_at_native_and_upscaled_resolutions():
    original = np.random.default_rng(42).uniform(0.0, 1.0, size=(12, 32)).astype(np.float32)

    img_native = encode_tactile_map(original, render="raw")
    assert img_native.shape == (12, 32, 3) and img_native.dtype == np.uint8
    recovered_native = decode_tactile_image(img_native, (12, 32))
    np.testing.assert_allclose(recovered_native, original, atol=0.5 / 255.0 + 1e-6)

    img_up = encode_tactile_map(original, render="raw", width=256, height=96)
    assert img_up.shape == (96, 256, 3)
    np.testing.assert_allclose(decode_tactile_image(img_up, (12, 32)), recovered_native, atol=1e-6)


def test_heatmap_encoding_shape_and_contrast():
    zeros = np.zeros((12, 32), dtype=np.float32)
    img_zero = encode_tactile_map(zeros, render="heatmap", width=256, height=96)
    peak = zeros.copy()
    peak[6, 16] = 1.0
    img_peak = encode_tactile_map(peak, render="heatmap", width=256, height=96)
    assert img_zero.shape == (96, 256, 3)
    assert not np.array_equal(img_zero, img_peak)


def test_signal_modes():
    hw = FakeFlexiTacSensor()
    frame = hw.open().read_latest()

    cfg_norm = FlexiTacTactileConfig(port="/dev/null")
    np.testing.assert_allclose(frame_to_tactile_map(frame, hw, cfg_norm), reference_normalized())

    cfg_abs = FlexiTacTactileConfig(port="/dev/null", signal="baseline_subtracted", threshold=25.0)
    expected = np.clip(reference_raw().astype(np.float32) - BASELINE - 25.0, 0, 100) / 100.0
    np.testing.assert_allclose(frame_to_tactile_map(frame, hw, cfg_abs), expected, atol=1e-6)


def test_lifecycle_and_frame_flow():
    sensor, hw = _connected_sensor()
    try:
        assert sensor.is_connected and hw.opened
        np.testing.assert_allclose(sensor.read_latest_tactile(max_age_ms=500), reference_normalized())
        assert sensor.async_read(timeout_ms=500).shape == (12, 32, 3)
        assert sensor.read().shape == (12, 32, 3)
        assert sensor.read_latest().shape == (12, 32, 3)
        assert sensor.frame_count > 0
    finally:
        sensor.disconnect()
    assert not sensor.is_connected and hw.closed
    with pytest.raises(ConnectionError):
        sensor.read_latest_tactile()


def test_dead_sensor_is_reported_not_silently_frozen():
    sensor, hw = _connected_sensor()
    try:
        hw.fail_reads = True
        time.sleep(0.15)
        # Without an age bound the last frame is still available (callers decide policy)...
        assert sensor.read_latest_tactile() is not None
        # ...but any age bound surfaces the dead sensor instead of repeating stale data.
        with pytest.raises(TimeoutError):
            sensor.read_latest_tactile(max_age_ms=50)
        with pytest.raises(TimeoutError):
            sensor.read_latest(max_age_ms=50)
        sensor.async_read(timeout_ms=0)  # consume whatever is pending
        with pytest.raises(TimeoutError):
            sensor.async_read(timeout_ms=50)
        assert sensor.last_error is not None
    finally:
        sensor.disconnect()


def test_heatmap_ema_advances_per_frame_not_per_read():
    sensor, hw = _connected_sensor(render="heatmap")
    try:
        hw.fail_reads = True  # freeze the stream
        time.sleep(0.1)
        first = sensor.read_latest(max_age_ms=10_000)
        for _ in range(5):
            again = sensor.read_latest(max_age_ms=10_000)
        np.testing.assert_array_equal(first, again)
    finally:
        sensor.disconnect()


def test_frame_listener_receives_every_frame():
    received = []
    hw = FakeFlexiTacSensor()
    sensor = FlexiTacTactile(FlexiTacTactileConfig(port="/dev/null"), sensor_factory=lambda _c: hw)
    sensor.add_frame_listener(lambda data, epoch_ns, seq: received.append((seq, epoch_ns, data.shape)))
    sensor.connect()
    time.sleep(0.05)
    sensor.disconnect()
    seqs = [s for s, _, _ in received]
    assert len(seqs) >= 3
    assert seqs == list(range(1, len(seqs) + 1))
    assert all(shape == (12, 32) and ns > 0 for _, ns, shape in received)


def test_warmup_timeout_cleans_up():
    hw = FakeFlexiTacSensor()
    hw.fail_reads = True
    cfg = FlexiTacTactileConfig(port="/dev/null", warmup_timeout_s=0.2)
    sensor = FlexiTacTactile(cfg, sensor_factory=lambda _c: hw)
    with pytest.raises(TimeoutError):
        sensor.connect()
    assert not sensor.is_connected and hw.closed


def test_process_backend_shared_memory():
    cfg = FlexiTacTactileConfig(port="/dev/null", process_backend=True, process_connect_timeout_s=30.0)
    sensor = FlexiTacTactile(cfg, sensor_factory=make_fake_sensor)
    sensor.connect(warmup=True)
    try:
        np.testing.assert_allclose(sensor.read_latest_tactile(max_age_ms=1000), reference_normalized())
        assert sensor.async_read(timeout_ms=1000).shape == (12, 32, 3)
        assert sensor.frame_count > 0
    finally:
        sensor.disconnect()
    assert not sensor.is_connected


def test_real_flexitac_api_contract():
    """Guard against drift between this plugin and the real flexitac package."""
    flexitac = pytest.importorskip("flexitac")
    from lerobot_tactile_flexitac import make_flexitac_sensor

    params = inspect.signature(flexitac.FlexiTacSensor.__init__).parameters
    for name in ("port", "rows", "cols", "baud", "threshold", "noise_scale", "init_frames", "read_timeout_s", "baseline"):
        assert name in params, f"flexitac.FlexiTacSensor no longer accepts {name!r}"
    for method in ("open", "close", "read_latest"):
        assert callable(getattr(flexitac.FlexiTacSensor, method))
    frame_fields = set(flexitac.FlexiTacFrame.__dataclass_fields__)
    assert {"seq", "raw", "normalized"} <= frame_fields

    cfg = FlexiTacTactileConfig(port="/dev/null", baseline=20.0, signal="baseline_subtracted")
    real = make_flexitac_sensor(cfg)  # constructing does not open the port
    assert (real.rows, real.cols) == (12, 32) and real.baseline is not None

    raw = np.full((12, 32), 95, dtype=np.uint8)
    frame = flexitac.FlexiTacFrame(seq=0, timestamp_s=0.0, raw=raw, normalized=np.zeros((12, 32), np.float32))
    np.testing.assert_allclose(frame_to_tactile_map(frame, real, cfg), 0.5, atol=1e-6)
