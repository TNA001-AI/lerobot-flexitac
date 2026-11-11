"""Hardware-free tests for lerobot_robot_so_flexitac."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.robots.config import RobotConfig
from lerobot_robot_so_flexitac import (
    FlexiTacSensorEntryConfig,
    SOTactileFollower,
    SOTactileFollowerConfig,
    flatten_tactile_names,
    unflatten_tactile_state,
)

SHAPE = (12, 32)


class _FakeTactileSensor:
    def __init__(self, fail_connect: bool = False):
        self.is_connected = False
        self.fail_connect = fail_connect
        self.fail_reads = False
        self.listeners = []
        self.data = np.arange(SHAPE[0] * SHAPE[1], dtype=np.float32).reshape(SHAPE) / (SHAPE[0] * SHAPE[1])

    def connect(self, warmup: bool = True) -> None:
        if self.fail_connect:
            raise TimeoutError("no frame from fake sensor")
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def add_frame_listener(self, cb) -> None:
        self.listeners.append(cb)

    def remove_frame_listener(self, cb) -> None:
        self.listeners.remove(cb)

    def emit(self, seq: int) -> None:
        for cb in list(self.listeners):
            cb(self.data, time.time_ns(), seq)

    def read_latest_tactile(self, max_age_ms=None) -> np.ndarray:
        if self.fail_reads:
            raise TimeoutError("stale")
        return self.data.copy()


@pytest.fixture
def stub_arm(monkeypatch: pytest.MonkeyPatch):
    """Replace the SO follower arm with an in-memory stub (no serial bus)."""
    base = SOTactileFollower.__mro__[1]
    state = {"arm_disconnects": 0}

    def fake_init(self, cfg):
        self.config = cfg
        self.bus = MagicMock(motors={f"j{i}": i for i in range(6)}, is_connected=False)
        self.cameras = {}

    def fake_connect(self, calibrate=True):
        self.bus.is_connected = True

    def fake_disconnect(self):
        if not self.is_connected:  # mirrors upstream @check_if_not_connected
            raise RuntimeError("parent refused: not connected")
        self.bus.is_connected = False
        state["arm_disconnects"] += 1

    monkeypatch.setattr(base, "__init__", fake_init)
    monkeypatch.setattr(base, "connect", fake_connect)
    monkeypatch.setattr(base, "disconnect", fake_disconnect)
    monkeypatch.setattr(base, "get_observation", lambda self: {f"j{i}.pos": float(i) for i in range(6)})
    monkeypatch.setattr(base, "is_connected", property(lambda self: bool(self.bus.is_connected)))
    return state


def _robot(sensor, **entry_kwargs):
    cfg = SOTactileFollowerConfig(
        port="/dev/ttyACM0",
        tactile_sensors={"primary": FlexiTacSensorEntryConfig(port="/dev/ttyUSB0", **entry_kwargs)},
    )
    return SOTactileFollower(cfg, tactile_factory=lambda _n, _c: sensor)


def test_robot_config_registration_and_naming_convention():
    assert RobotConfig.get_choice_class("so_tactile_follower") is SOTactileFollowerConfig
    assert SOTactileFollowerConfig.__name__.removesuffix("Config") == SOTactileFollower.__name__


def test_flatten_and_unflatten_c_order_roundtrip():
    names = [f"j{i}.pos" for i in range(6)] + flatten_tactile_names("primary", SHAPE)
    assert names[6] == "tactile.primary.r00_c00" and names[-1] == "tactile.primary.r11_c31"

    rng = np.random.default_rng(7)
    maps = rng.uniform(0.0, 1.0, size=(4, *SHAPE)).astype(np.float32)
    state = np.concatenate([rng.normal(size=(4, 6)).astype(np.float32), maps.reshape(4, -1)], axis=-1)
    np.testing.assert_allclose(unflatten_tactile_state(names, state, "primary", SHAPE), maps)


def test_features_before_connect_and_observation_values(stub_arm):
    sensor = _FakeTactileSensor()
    robot = _robot(sensor)
    ft = robot.observation_features  # must work before connect() (lerobot-record order)
    assert len(ft) == 6 + 384 and ft["tactile.primary.r00_c00"] is float

    robot.connect(calibrate=False)
    assert robot.is_connected
    obs = robot.get_observation()
    keys = list(ft)
    state = np.array([obs[k] for k in keys], dtype=np.float32)
    np.testing.assert_allclose(unflatten_tactile_state(keys, state, "primary", SHAPE), sensor.data)

    sensor.is_connected = False  # tactile loss is reflected in robot connectivity
    assert not robot.is_connected
    sensor.is_connected = True
    robot.disconnect()
    assert stub_arm["arm_disconnects"] == 1 and not sensor.is_connected


def test_stale_tactile_is_bridged_briefly_then_raises(stub_arm):
    sensor = _FakeTactileSensor()
    robot = _robot(sensor, stale_timeout_s=0.1)
    robot.connect(calibrate=False)
    good = robot.get_observation()

    sensor.fail_reads = True
    bridged = robot.get_observation()  # within stale_timeout: reuse last valid frame
    assert bridged == good

    time.sleep(0.15)
    with pytest.raises(RuntimeError, match="no fresh frame"):
        robot.get_observation()
    robot.disconnect()


def test_connect_rolls_back_arm_when_tactile_fails(stub_arm):
    sensor = _FakeTactileSensor(fail_connect=True)
    robot = _robot(sensor)
    with pytest.raises(TimeoutError):
        robot.connect(calibrate=False)
    assert stub_arm["arm_disconnects"] == 1
    assert not robot.bus.is_connected and not sensor.is_connected


def test_full_rate_csv_sidecar(stub_arm, tmp_path: Path):
    sensor = _FakeTactileSensor()
    robot = _robot(sensor, record_raw_csv=True, raw_csv_dir=str(tmp_path), csv_flush_every=2)
    robot.connect(calibrate=False)
    for seq in range(1, 6):  # 5 sensor frames, independent of observation rate
        sensor.emit(seq)
    robot.disconnect()
    assert sensor.listeners == []

    lines = (tmp_path / "flexitac_primary.csv").read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("epoch_ns,seq,tactile.primary.r00_c00")
    assert len(lines) == 1 + 5
    assert [int(line.split(",")[1]) for line in lines[1:]] == [1, 2, 3, 4, 5]
